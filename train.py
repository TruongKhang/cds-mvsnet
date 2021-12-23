import argparse, collections
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import datasets.data_loaders as module_data
import models.model as module_arch
import models.losses as module_loss
from trainer import Trainer
from parse_config import ConfigParser


SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True


def main(config):
    logger = config.get_logger('train')

    # data_loader = config.init_obj('data_loader', module_data)
    train_data_loaders, valid_data_loaders = [], []
    for dl_params in config['data_loader']:
        dl_name, dl_args = dl_params['type'], dict(dl_params['args'])
        train_dl_args = dl_args.copy()
        train_dl_args['data_list'] = dl_args['train_data_list']
        del train_dl_args['train_data_list'], train_dl_args['val_data_list']
        data_loader = getattr(module_data, dl_name)(**train_dl_args)
        train_data_loaders.append(data_loader)
        # setup valid_data_loader instances
        
        val_kwags = {
            "data_list": dl_args['val_data_list'],
            "mode": "val",
            "num_srcs": 5,
            "shuffle": False,
            "batch_size": 2 if 'dtu' in dl_args['data_path'] else 5
        }
        val_dl_args = train_dl_args.copy()
        val_dl_args.update(val_kwags)
        val_data_loader = getattr(module_data, dl_name)(**val_dl_args)
        valid_data_loaders.append(val_data_loader)

    # build models architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    mvsnet_params = filter(lambda p: p.requires_grad, model.parameters())
    mvsnet_optimizer = config.init_obj('optimizer', torch.optim, mvsnet_params)
    optimizer = mvsnet_optimizer
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    writer = SummaryWriter(config.log_dir)

    trainer = Trainer(model, criterion, optimizer, config=config, data_loader=train_data_loaders,
                      valid_data_loader=valid_data_loaders, lr_scheduler=lr_scheduler, writer=writer)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
