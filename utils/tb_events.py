#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import shutil
import numpy as np
import matplotlib.cm as cm

import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont


def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)


def write_scalar(tblogger, items, step, title_text=None):
    tblogger.add_scalar(f"{title_text}", items, step + 1)


def write_tbPR(tblogger, TP, FP, FN, epoch, task):
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    for defect_idx in range(TP.shape[0]):
        tblogger.add_scalar('precision/{}/{}'.format(defect_idx, task), P[defect_idx], epoch + 1)
        tblogger.add_scalar('recall/{}/{}'.format(defect_idx, task), R[defect_idx], epoch + 1)


def write_tbimg(tblogger, imgs, step, cam_imgs=None, real_classes=None, pred_classes=None, task='train'):
    _, c, h, w = imgs.shape
    if cam_imgs is not None:
        cam_colormaps = cm.jet(cam_imgs)  # Apply 'jet' colormap
        cam_colormaps = (cam_colormaps[:, :, :, :3] * 255).astype(np.uint8)  # 컬러맵을 [0, 255] 범위로 변환 (RGB)

    for i in range(len(imgs)):
        print_img = transforms.ToPILImage()(imgs[i])
        if cam_imgs is not None:
            pil_cam_imgs = Image.fromarray(cam_colormaps[i])
            pil_cam_imgs = pil_cam_imgs.resize((h, w), Image.BILINEAR)
            print_img = Image.blend(print_img, pil_cam_imgs, 0.5)
        #
        draw = ImageDraw.Draw(print_img)
        #
        font = ImageFont.load_default()
        text = f'pred : {pred_classes[i]}'
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        #
        margin = 5
        x, y = margin, margin
        draw.rectangle([x, y, x + text_width + margin, y + text_height + margin], fill="black")
        #
        draw.text((x + margin, y + margin), text, font=font, fill="white")
        #
        tblogger.add_image(
            f'{task}_imgs/class_{real_classes[i]}',
            transforms.ToTensor()(print_img),
            step + 1,
            dataformats='CHW'
        )
