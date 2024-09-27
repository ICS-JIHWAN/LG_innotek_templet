#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import io
import logging
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch

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


def write_tbPR(tblogger, precision, recall, class_names, step, task):
    for i, class_name in enumerate(class_names):
        tblogger.add_scalar(f'{task}_Precision/{class_name}', precision[i], global_step=step)
        tblogger.add_scalar(f'{task}_Recall/{class_name}', recall[i], global_step=step)


def write_tbimg(tblogger, imgs, step, cam_imgs=None, real_classes=None, pred_classes=None, task='train'):
    _, c, h, w = imgs.shape

    for i in range(len(imgs)):
        print_img = transforms.ToPILImage()(imgs[i])
        if cam_imgs is not None:
            heatmap = cv2.resize(cam_imgs[i], (h, w))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            print_img = (255 * imgs[i]).permute(1, 2, 0).numpy().astype(np.uint8)
            print_img = cv2.addWeighted(print_img, 0.5, heatmap, 0.5, 0, dtype=cv2.CV_8U)

            print_img = Image.fromarray(print_img)
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


def write_tbCM(tblogger, cm, class_names, step, task='train'):
    figure = plot_confusion_matrix(cm, class_names)
    cm_image = plot_to_image(figure)

    tblogger.add_image(
        f'{task}_cm/confusion matrix',
        cm_image,
        step + 1,
        dataformats='CHW'
    )


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    #
    # PIL 이미지를 numpy 배열로 변환
    image = Image.open(buf)
    image = np.array(image)

    image = image / 255.0

    # 이미지를 3D tensor로 변환 (C, H, W 형식으로 변환)
    image = torch.tensor(image).permute(2, 0, 1)
    return image
