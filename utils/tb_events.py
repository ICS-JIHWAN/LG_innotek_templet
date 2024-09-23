#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
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

    for i in range(len(imgs)):
        print_img = transforms.ToPILImage()(imgs[i])
        if cam_imgs is not None:
            heatmap = cv2.applyColorMap(np.uint8(cam_imgs[i]), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (h, w))
            heatmap = np.float32(heatmap) / 255

            cam = 0.7 * heatmap + 0.3 * np.array(imgs[i].permute(1, 2, 0))

            print_img = Image.fromarray((cam * 255).astype(np.uint8))
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
