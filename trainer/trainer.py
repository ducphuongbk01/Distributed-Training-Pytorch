import os
import copy
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class Trainer(object):
    def __init__(self, 
                 max_epoch, 
                 batch_size,
                 pin_memory, 
                 have_validate=False, 
                 save_best_for=None, 
                 save_period=None, 
                 save_folder='.', 
                 snapshot_path=None, 
                 logger=None):
        # Logger
        self.log = lambda msg, log_type: logger.log(msg, log_type) if logger is not None else print(f"{log_type.upper()}: {msg}")
        
        # Save folder
        self.save_folder = save_folder
        self.save_weight_folder = os.path.join(self.save_folder, "weights")
        if not os.path.exists(self.save_weight_folder):
            os.makedirs(self.save_weight_folder)
        
        # Train definition
        self.save_best_for = save_best_for
        self.cur_epoch = 0
        self.max_epoch = max_epoch
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Load snapshot
        if snapshot_path is not None:
            self._load_snapshot(snapshot_path)

        # Distributed set up
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.model = self.model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # Dataloader set up
        self.batch_size = batch_size
        self.local_batch_size = batch_size//self.world_size
        train_dataset = self.build_train_dataset()
        self.train_dataloader = self.build_dataloader(train_dataset, 
                                                      self.local_batch_size,
                                                      pin_memory, 
                                                      collate_fn=train_dataset.collate_fn if callable(getattr(train_dataset, "collate_fn", None)) else None,
                                                      phase="train")
        self.have_validate = have_validate
        self.save_period = save_period
        if self.have_validate:
            val_dataset = self.build_val_dataset()
            self.val_dataloader = self.build_dataloader(val_dataset, 
                                                        self.local_batch_size, 
                                                        pin_memory, 
                                                        collate_fn=val_dataset.collate_fn if callable(getattr(val_dataset, "collate_fn", None)) else None,
                                                        phase="val")  

    # Initialize for DDP training
    @ staticmethod
    def ddp_setup(backend):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend=backend)

    # Clear all process for finished training
    @staticmethod
    def destroy_process():
        destroy_process_group()

    # Save model
    def _save_snapshot(self, epoch, name="last"):
        snapshot = dict(
            epoch=epoch,
            model_state_dict=self.model.module.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict= self.scheduler.state_dict()
        )
        torch.save(snapshot, os.path.join(self.save_weight_folder, f"{name}.pth"))
        self.log(f"Saved model at epoch {epoch}!", log_type="info")

    # Load model
    def _load_snapshot(self, path):
        snapshot = torch.load(path, map_location="cpu")
        self.cur_epoch = snapshot["epoch"]
        self.model.load_state_dict(snapshot["model_state_dict"])
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.scheduler.load_state_dict(snapshot["scheduler_state_dict"])

    # Training pipeline
    def train(self):
        # Best information
        if self.have_validate:
            best_fitness = dict(epoch=None, value=None, metrics = None)
        
        # Training pipeline
        for epoch in range(self.cur_epoch, self.max_epoch):
            self.cur_epoch = epoch

            # Validate
            if self.have_validate and (epoch%self.save_period==0):
                if self.world_rank==0:
                    self.model.eval()
                    metrics = self.validate()
                    # Check for save the best model
                    if (best_fitness["epoch"] is None) or \
                    (metrics[self.save_best_for[0]]>=best_fitness["value"] if self.save_best_for[1] == "geq" else metrics[self.save_best_for[0]]<=best_fitness["value"]):
                        best_fitness["epoch"] = epoch
                        best_fitness["value"] = metrics[self.save_best_for[0]]
                        best_fitness["metrics"] = copy.deepcopy(metrics)
                        self._save_snapshot(epoch, name="best")
                    # Log best
                    self.log(msg=100*'=', log_type="info")
                    log_msg = f"The BEST model is at EPOCH {best_fitness["epoch"]} and has "
                    for k, v in best_fitness["metrics"].items():
                        log_msg += f" | {k.upper()} = {v} | "
                    self.log(log_msg, log_type="info")
                    # World rank 0 run into the barrier unitl finishing validation
                    torch.distributed.barrier()
                else:
                    # Other world ranks wait at barrier
                    torch.distributed.barrier()
            
            # Train
            self.model.train()
            loss_local = None
            self.train_dataloader.sampler.set_epoch(epoch)
            self.log(msg=100*'=', log_type="info")
            self.log(msg=f"[GPU{self.world_rank}] Epoch {epoch+1}/{self.max_epoch}", log_type="info")
            loop = tqdm(self.train_dataloader)
            for i, batch in enumerate(loop):
                # Train
                loss = self.train_step(batch)
                # Log
                loop.set_postfix(loss)
                # Collect loss
                if loss_local is None:
                    loss_local = dict()
                    for k, v in loss.items():
                        loss_local[k] = [v]
                else:
                    for k, v in loss.items():
                        loss_local[k].append(v)

            # Update scheduler
            self.scheduler.step()
            self.log(f"THE NEXT LEARNING RATE VALUE IS {self.scheduler.get_last_lr()[0]}", log_type="info")
            
            # Check to save model
            if self.world_rank==0:
                if self.have_validate:
                    self._save_snapshot(epoch+1, name="last")
                elif epoch%self.save_period==0:
                    self._save_snapshot(epoch+1, name=f"checkpoint_epoch_{epoch+1}")
                # World rank 0 run into the barrier unitl finishing saving
                torch.distributed.barrier()
            else:
                # Other world ranks wait at barrier
                torch.distributed.barrier()
            
            # Aggregate & log
            log_msg = f"TOTAL LOCAL TRAINING LOSS: "
            for k, v in loss_local.items():
                log_msg += f" | {k} = {np.mean(v)} | "
            self.log(log_msg, log_type="info")
        
        # Finish
        self.log("Finished!", log_type="info")
  
    # Validate for training
    def validate(self):
        avg_metrics = None
        loop = tqdm(self.val_dataloader)
        for batch in loop:
            batch_metrics = self.validate_step(batch)
            if avg_metrics is None:
                avg_metrics = dict()
                for k, v in batch_metrics.items():
                    avg_metrics[k] = [batch_metrics[k]]
            else:
                for k, v in batch_metrics.items():
                    avg_metrics[k].append(v)
            # Show info
            loop.set_postfix(batch_metrics)
        # Aggregate
        for k, v in avg_metrics.items():
            avg_metrics[k] = np.mean(v)
        # For logging
        log_msg = "VALIDATE RESULTS: "
        for k, v in avg_metrics.items():
            log_msg += f" | {k} = {v} | "
        self.log(log_msg, log_type="info")
        return avg_metrics

    # Get dataloader
    def build_dataloader(self, dataset, batch_size, pin_memory, collate_fn=None, phase="train"):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
            sampler=DistributedSampler(dataset, shuffle=True) if phase=="train" else None,
            collate_fn=collate_fn
        )

    # Get train dataset
    def build_train_dataset(self):
        raise NotImplementedError("Please implement the build_train_dataset method before calling")
    
    # Get validate dataset
    def build_val_dataset(self):
        raise NotImplementedError("Please implement the build_val_dataset method before calling")
    
    # Get model
    def build_model(self):
        raise NotImplementedError("Please implement the build_model method before calling")
    
    # Get objective (loss) function
    def build_criterion(self):
        raise NotImplementedError("Please implement the build_criterion method before calling")
    
    # Get opimizer 
    def build_optimizer(self):
        raise NotImplementedError("Please implement the build_optimizer method before calling")

    # Get scheduler
    def build_scheduler(self):
        raise NotImplementedError("Please implement the build_scheduler method before calling")
    
    # Design for batch preprocessing
    def preprocess_batch(self):
        raise NotImplementedError("Please implement the preprocess_batch method before calling")
    
    # Design forward, backward and update process
    def train_step(self):
        raise NotImplementedError("Please implement the train_step method before calling")

    # Validate for each batch
    def validate_step(self):
        raise NotImplementedError("Please implement the validate_step method before calling")
