import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import Namespace

import os, sys, glob
CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) 
sys.path.append(CODE_DIR)

from dgld.data.data_interface import NodeLevelAnomalyDataset
from dgld.modules.pretrain import GGD, GRACE

def pretrain_feature_extractor(graph, trainer_args, model_args, load_exist_ckpt=False):
    ckpt_dir = trainer_args['ckpt_dir']
    
    if load_exist_ckpt:
        # Check if there is a ckpt file in the directory
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if len(ckpt_files) > 0:
            # Load the latest checkpoint file with the same model name and hyperparameters
            ckpt_file = max(ckpt_files, key=os.path.getctime)
            model = eval(f"{model_args['model_name']}").load_from_checkpoint(ckpt_file, graph=graph.clone(), **model_args)
            print(f"Loaded checkpoint from {ckpt_file}")
            return model
    
    # Create a new model with the given parameters
    model = eval(f"{model_args['model_name']}")(graph.clone(), **model_args)
    
    # Set up checkpoint callbacks to save model after each epoch
    checkpoint_callbacks = ModelCheckpoint(dirpath=trainer_args['ckpt_dir'], filename="{epoch}_pretrain")
    
    # Train the model
    trainer = Trainer.from_argparse_args(args=Namespace(**trainer_args), callbacks=[checkpoint_callbacks])
    trainer.logger._default_hp_metric = None
    trainer.fit(model)
    
    return model
    
if __name__ == '__main__':
    
    dataset = NodeLevelAnomalyDataset('Amazon', 'natural', raw_dir=os.path.join(CODE_DIR, 'dgld/data/downloads'))
    graph = dataset[0]
    
    trainer_args = {
        'max_epochs': 300,
        "accelerator": "gpu", 
        "devices":1, 
        "reload_dataloaders_every_n_epochs":1,
        'ckpt_dir': os.path.join(CODE_DIR, '../result/checkpoints/unarchived')
    }
    
    model_args = {
        "model_name": "GGD",
        "embedding_dim": 256,
        "n_layers": 2,
        "dropout": 0,
        "batch_norm": False,
        "encoder_name": 'sage',
        "lr": 1e-3,
        "weight_decay": 0,
        "batch_size": 0,
    }
    
    pretrain_feature_extractor(graph, trainer_args, model_args)