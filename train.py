import os
import sys
import argparse

from config.config import get_config_dict
from core.engine import Trainer

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='DL Template for Pytorch', add_help=add_help)
    parser.add_argument('--train_path', default='./data', type=str)
    #
    parser.add_argument('--height', default=52, type=int)
    parser.add_argument('--width', default=52, type=int)
    parser.add_argument('--channel', default=3, type=int)
    #
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    #
    return parser


if __name__ == '__main__':
    # Get arguments & configuration
    args = get_args_parser().parse_args()
    config = get_config_dict()

    # Get Trainer
    trainer = Trainer(args, config)

    # Start train
    trainer.start_train()
