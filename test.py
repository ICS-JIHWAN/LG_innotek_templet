import os
import sys
import cv2
import argparse
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
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
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--test_path', default='./data/test_dataset', type=str)
    parser.add_argument('--save_path', default='/home/jhchoi/inno/LG_innotek_templet/runs/resnet/best_model.pth', type=str)
    parser.add_argument('--num_class', default=3, type=int)
    #
    parser.add_argument('--height', default=224, type=int)
    parser.add_argument('--width', default=224, type=int)
    parser.add_argument('--channel', default=3, type=int)
    #
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    #
    return parser


if __name__ == '__main__':
    #
    # Get arguments & configuration
    args = get_args_parser().parse_args()
    cfg = get_config_dict()
    #
    # Set device
    if args.gpu_id is not None:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    #
    # data load
    test_object = data_class(
        path=args.test_path,
        height=args.height,
        width=args.width,
        task='test'
    )
    test_loader = DataLoader(
        test_object,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    num_class = args.num_class
    #
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
        best_model = torch.load(args.save_path, weights_only=False)
        le = best_model['lb_encoder']
        model.load_state_dict(best_model['model'])
    except OSError:
        print('cannot open : ', args.save_path)
    #
    # cam
    target_layer = model.conv5_x[-1].residual_function[-2]
    cam = GradCAM(model, target_layer=target_layer)
    #
    # test output make dir
    for cls in le.classes_:
        os.makedirs(os.path.join(os.getcwd(), 'data/test_output/{}'.format(cls)), exist_ok=True)
    #
    # test start
    print(f'\nTest start')
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    #
    #
    model.eval()
    with torch.no_grad():
        for step, input_img in pbar:
            # Data info
            images = input_img[0][0].to(device)
            paths = input_img[1][0]
            #
            # Forward propagation
            output = model(images)
            #
            # Prediction info
            _, predict = torch.max(output, 1)
            predict_cls_names = le.inverse_transform(predict.cpu())
            #
            pbar.set_postfix(predict_class=predict_cls_names[0])
            #
            cv2.imshow('image', images[0].permute(1, 2, 0).cpu().numpy())
            cv2.waitKey(0)
            #
            # Test output save
            try:
                im = cv2.imread(paths)
                assert im is not None, f"opencv cannot r ead image correctly or {paths} not exists"
            except:
                im = cv2.cvtColor(np.asarray(Image.open(paths)), cv2.COLOR_RGB2BGR)
                assert im is not None, f"Image Not Found {paths}, workdir: {os.getcwd()}"

            cv2.imwrite(os.path.join(os.getcwd(), 'data/test_output/{}'.format(predict_cls_names[0]), os.path.basename(paths)), im)

