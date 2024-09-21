#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import shutil

import torchvision.transforms as transforms

from PIL import ImageDraw, ImageFont

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


def write_tbimg(tblogger, imgs, step, real_classes=None, pred_classes=None, task='train'):
    for i in range(len(imgs)):
        print_img = transforms.ToPILImage()(imgs[i])
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
        try:
            tblogger.add_image(
                f'{task}_imgs/class_{real_classes[i]}',
                transforms.ToTensor()(print_img),
                step + 1,
                dataformats='CHW'
            )
        except:
            None
