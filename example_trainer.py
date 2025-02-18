import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainer.trainer import Trainer
from dataset.example_dataset import ExampleDataset
from model.vgg16 import VGG16


class ExampleTrainer(Trainer):
    def __init__(self, 
                 train_path,
                 val_path,
                 labels,
                 height,
                 width,
                 max_epoch, 
                 batch_size, 
                 pin_memory, 
                 have_validate=False, 
                 save_best_for=None, 
                 save_period=None, 
                 save_folder='.', 
                 snapshot_path=None, 
                 logger=None):
        self.train_path = train_path
        self.val_path = val_path
        self.labels = labels
        self.height = height
        self.width = width
        super().__init__(max_epoch, 
                         batch_size,
                         pin_memory, 
                         have_validate, 
                         save_best_for, 
                         save_period, 
                         save_folder, 
                         snapshot_path, 
                         logger)
    
    # Get train dataset
    def build_train_dataset(self):
        return ExampleDataset(self.train_path, self.labels, self.height, self.width, phase="train")
    
    # Get validate dataset
    def build_val_dataset(self):
        return ExampleDataset(self.train_path, self.labels, self.height, self.width, phase="val")
    
    # Get model
    def build_model(self):
        return VGG16(3, 3, True)
    
    # Get objective (loss) function
    def build_criterion(self):
        criterion = lambda x, y: F.cross_entropy(F.softmax(x, dim=-1), y, reduction="mean")
        return criterion
    
    # Get opimizer 
    def build_optimizer(self):
        return optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Get scheduler
    def build_scheduler(self):
        step = self.max_epoch//4
        return optim.lr_scheduler.MultiStepLR(self.optimizer, [step, 2*step, 3*step], gamma=0.1)
    
    # Design for batch preprocessing
    def preprocess_batch(self, batch):
        return [e.to(f"cuda:{self.local_rank}") for e in batch]
    
    # Design forward, backward and update process
    def train_step(self, batch):
        # Preprocess & Un-patch
        batch = self.preprocess_batch(batch)
        img_batch, lb_batch = batch
        # Clear gradient from optimizer
        self.optimizer.zero_grad()
        # Turn on gradient calculation flag
        with torch.set_grad_enabled(True):
            # Forward
            out = self.model(img_batch)
            # Loss Calculation
            loss = self.criterion(out, lb_batch)
            # Calculate gradient (backward)
            loss.backward()
            # Update weights
            self.optimizer.step()
        return dict(ce_loss=loss.item())

    # Validate for each batch
    def validate_step(self, batch):
        ## Preprocess & Un-patch
        batch = self.preprocess_batch(batch)
        img_batch, lb_batch = batch
        # Turn off gradient calculation flag
        with torch.set_grad_enabled(False):
            # Predict (Forward)
            preds = F.softmax(self.model(img_batch), dim=-1)
            pred_lbs = preds.argmax(dim=-1)
            acc = (pred_lbs==lb_batch).sum()/pred_lbs.shape[0]
        return dict(accuracy=acc.item())

