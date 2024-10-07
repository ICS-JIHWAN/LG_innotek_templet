import os
import sys
import argparse
import numpy as np

import torch

from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score,
    precision_recall_curve, roc_auc_score,
    confusion_matrix
)
from torch.utils.data import DataLoader

from config.config import get_config_dict
from dataset.data_loader import data_class
from cam_algorithms import GradCAM

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='DL Template for Pytorch', add_help=add_help)
    #
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--test_path', default='./data/testing', type=str)
    parser.add_argument('--save_path', default='./rums/resnet/best_model.pth', type=str)
    #
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--channel', default=3, type=int)
    #
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    #
    return parser


if __name__ == '__main__':
    # Get arguments & configuration
    args = get_args_parser().parse_args()
    cfg = get_config_dict()

    # Set device
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    # data load
    test_object = data_class(
        path=args.test_path,
        height=args.height,
        width=args.width,
        task='train'
    )
    test_loader = DataLoader(
        test_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=data_class.collate_fn
    )
    num_class = test_object.num_class

    # model load
    if cfg['model']['name'] == 'lenet':
        from model.lenet import lenet
        model = lenet(num_class=num_class).to(device)
    elif cfg['model']['name'] == 'alexnet':
        from model.alexnet import alexnet
        model = alexnet(num_class=num_class).to(device)
    elif cfg['model']['name'] == 'resnet':
        from model.resnet import resnet
        model = resnet(num_class=num_class).to(device)
    else:
        print('Model load fail')
        raise NotImplementedError

    try:
        model.load_state_dict(torch.load(args.save_path))
    except OSError:
        print('cannot open : ', args.save_path)

    # cam
    target_layer = model.conv5_x[-1].residual_function[-2]
    cam = GradCAM(model, target_layer=target_layer)

    # test start
    print(f'\nTest start')
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))

    # metrix score for one epoch
    y_true_int_list = []
    y_true_onehot_list = []
    y_prob_list = []
    y_pred_list = []
    cm = np.zeros((num_class, num_class))
    #
    correct = 0
    total = 0
    #
    model.eval()
    with torch.no_grad():
        for step, batch_data in pbar:
            # Data info
            images = batch_data[0].to(device)  # Image
            labels = batch_data[1].to(device)
            int_labels = batch_data[2].to(device)
            cls_names = batch_data[3]
            #
            # Forward propagation
            output = model(images)
            #
            # Prediction info
            _, predict = torch.max(output, 1)
            predict_cls_names = test_loader.dataset.le.inverse_transform(predict.cpu())
            #
            y_true_int_list.append(int_labels.cpu().numpy())
            y_true_onehot_list.append(labels.cpu().numpy())
            y_prob_list.append(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())
            y_pred_list.append(predict.cpu().numpy())
            #
            #
            y_true_int_list.append(int_labels.cpu().numpy())
            y_true_onehot_list.append(labels.cpu().numpy())
            y_prob_list.append(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())
            y_pred_list.append(predict.cpu().numpy())
            #
            # Get statistics
            cm += confusion_matrix(cls_names, predict_cls_names, labels=test_loader.dataset.le.classes_)
            #
            total += labels.size(0)
            correct += (predict == int_labels).sum().item()
            #
            pbar.set_postfix(acc=round(correct / total, 2))

    # PR curve print
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_int_list)
    y_pred_prob = np.concatenate(y_prob_list)
    y_true_onehot = np.concatenate(y_true_onehot_list)
    #
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
