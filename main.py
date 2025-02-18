from example_trainer import ExampleTrainer
from utils.logger import Logger

if __name__=="__main__":
    logger = Logger("VGG16", file="./runs/logfile.log")

    ExampleTrainer.ddp_setup(backend="nccl")
    
    trainer = ExampleTrainer(train_path="./data/train",
                             val_path="./data/val",
                             labels=["cat", "dog", "snake"],
                             height=224,
                             width=224,
                             max_epoch=100,
                             batch_size=16,
                             pin_memory=True,
                             have_validate=True,
                             save_best_for=("accuracy", "geq"),
                             save_period=5,
                             save_folder="./runs",
                             snapshot_path=None,
                             logger=logger)

    trainer.train()

    ExampleTrainer.destroy_process()
