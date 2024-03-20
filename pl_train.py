import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler
from contextlib import contextmanager

from datasets import data_loaders
from trainer.pl_trainer import PL_Trainer
from utils import read_json


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config', type=str, help='main config path')
    # parser.add_argument(
    #     '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--n_gpus', type=int, default=1, help='number of cuda devices')
    parser.add_argument(
        '--max_epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument(
        '--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument(
        '--accumulate_grad_batches', type=int, default=2, help='accumulate gradient')
    parser.add_argument(
        '--val_batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--num_views', type=int, default=3)
    parser.add_argument(
        '--high_res', action="store_true")
    parser.add_argument(
        '--random_image_scale', action="store_true")
    parser.add_argument(
        '--reload_dataloaders_every_n_epochs', type=int, default=0)

    # parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


def get_list_dataloaders(args, config, mode="train"):
    list_dataloaders = []
    for info in config['data_loader']:
        dataloader_name, dataset_args = info['type'], dict(info['args'])
        
        dl_args = dataset_args.copy()
        dl_args["data_list"] = dataset_args[f"{mode}_data_list"]
        del dl_args["train_data_list"], dl_args["val_data_list"]
    
        if mode == "train":
            dl_args.update({"batch_size": args.batch_size, "num_workers": args.num_workers, "num_srcs": args.num_views,
                            "num_stages": config["arch"]["args"]["num_stages"], 
                            "high_res": args.high_res, "random_image_scale": args.random_image_scale})
        elif mode == "val":
            dl_args.update({"mode": "val", "num_srcs": 5, "shuffle": False, 
                            "batch_size": args.val_batch_size, "num_stages": config["arch"]["args"]["num_stages"]})

        module = getattr(data_loaders, dataloader_name)()
        dataloader = module.get(**dl_args)
        list_dataloaders.append(dataloader)

    return list_dataloaders

class DataModule(pl.LightningDataModule):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config

    def setup_(self, mode):
        list_dataloaders = []
        for info in self.config['data_loader']:
            dataloader_name, dataset_args = info['type'], dict(info['args'])
        
            dl_args = dataset_args.copy()
            dl_args["data_list"] = dataset_args[f"{mode}_data_list"]
            del dl_args["train_data_list"], dl_args["val_data_list"]
            if mode == "train":
                dl_args.update({"batch_size": self.args.batch_size, "num_workers": self.args.num_workers, 
                                "num_srcs": self.args.num_views, "num_stages": self.config["arch"]["args"]["num_stages"], 
                                "high_res": self.args.high_res, "random_image_scale": self.args.random_image_scale})
            else:
            # val_dl_args = dataset_args.copy()
            # val_dl_args["data_list"] = dataset_args[f"val_data_list"]
            # del val_dl_args["train_data_list"], val_dl_args["val_data_list"]
                dl_args.update({"mode": "val", "num_srcs": 5, "shuffle": False, "batch_size": self.args.val_batch_size, 
                                "num_stages": self.config["arch"]["args"]["num_stages"]})

            module = getattr(data_loaders, dataloader_name)()
            dataloader = module.get(**dl_args)
            list_dataloaders.append(dataloader)

            # val_dataloader = module.get(**val_dl_args)
            # self.list_val_dataloaders.append(val_dataloader)
        return list_dataloaders

    def train_dataloader(self):
        return CombinedLoader(self.setup_("train"), mode="min_size")

    def val_dataloader(self):
        return CombinedLoader(self.setup_("val"), mode="sequential")


class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time. 
    """

    def __init__(self):
        super().__init__()
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)

    @contextmanager
    def profile(self, action_name: str):
        try:
            torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            torch.cuda.synchronize()
            self.stop(action_name)


def build_profiler(name):
    if name == 'inference':
        return InferenceProfiler()
    elif name == 'pytorch':
        from pytorch_lightning.profilers import PyTorchProfiler
        return PyTorchProfiler(use_cuda=True, profile_memory=True, row_limit=100)
    elif name is None:
        return PassThroughProfiler()
    else:
        raise ValueError(f'Invalid profiler: {name}')


def main():
    # parse arguments
    # parser = get_parser()
    args = parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = read_json(args.config) # ConfigParser.from_args(parser)
    # pprint(dict(config))

    pl.seed_everything(19951209)  # reproducibility
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_Trainer(config, profiler=None, ckpt_path=args.ckpt_path)
    loguru_logger.info(f"Model LightningModule initialized!")
    
    # lightning data
    # train_dataloaders = CombinedLoader(get_list_dataloaders(args, config, mode="train"), mode="min_size")
    # val_dataloaders = CombinedLoader(get_list_dataloaders(args, config, mode="val"), mode="sequential")
    datamodule = DataModule(args, config)
    loguru_logger.info(f"Model DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=config["trainer"]["save_dir"], default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='id0_abs_depth_error', verbose=True, save_top_k=5, mode='min',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{id0_abs_depth_error:.3f}-{id0_prec@2mm:.3f}-{id0_prec@4mm:.3f}-{id0_prec@8mm:.3f}')
    # lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [] # StochasticWeightAveraging(swa_lrs=1e-2)]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu", devices=args.n_gpus, precision=32,
        check_val_every_n_epoch=1,
        # log_every_n_steps=7000,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_val_batches=1., num_sanity_val_steps=10,
        benchmark=True,
        max_epochs=args.max_epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=True,
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs, 
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    # trainer.fit(model, train_dataloaders, val_dataloaders) #, ckpt_path=args.ckpt_path)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
