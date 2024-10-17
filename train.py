import os
import sys
import argparse
import torch

from config.config import get_config_dict
from core.engine import Trainer

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='DL Template for Pytorch', add_help=add_help)
    #
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--train_path', default='./data/training', type=str)
    parser.add_argument('--valid_path', default='./data/testing', type=str)
    #
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--channel', default=3, type=int)
    #
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--tb_step', default=10, type=int)
    #
    return parser


if __name__ == '__main__':
    # Get arguments & configuration
    args = get_args_parser().parse_args()
    config = get_config_dict()

    # Set device
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    # Get Trainer
    trainer = Trainer(args, config, device=device)

    # Start train
    trainer.start_train()
