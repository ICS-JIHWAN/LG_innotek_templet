import os
import cv2
import glob
import numpy as np
import pickle
import time

import torch
import torchvision.transforms as transforms

from PIL import Image
from dataset.make_dataset import image_to_pkl
from torch.utils.data import Dataset


class data_class(Dataset):
    def __init__(self, path, width=52, height=52, augmentation=False, task='train'):
        super().__init__()
        # Data loader 초기화 시작
        start = time.time()
        #
        assert task == 'train' or task == 'val' or task == 'test', 'Invalid task...'
        #
        self.img_size = (height, width)
        self.augmentation = augmentation
        self.task = task
        #
        # 이미지 .pkl 파일로 변경
        if task == 'train' or task == 'val':
            self.le, self.num_class = image_to_pkl(path)
            self.data_paths = sorted(glob.glob(os.path.join(path, '*/*.pkl')))
        elif task == 'test':
            self.data_paths = sorted(glob.glob(os.path.join(path, '*')))
        self.fn_transform = self.get_transformer()
        #
        end = time.time()
        # logger 출력
        if task == 'train':
            print(f'Data loader 초기화 완료 !! {end - start:.5f} sec...\n'
                  f'총 train 데이터 수 : {self.__len__()}\n')
        elif task == 'val' or task == 'test':
            print(f'Data loader 초기화 완료 !! {end - start:.5f} sec...\n'
                  f'총 {task} 데이터 수 : {self.__len__()}\n')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        if self.task == 'train' or self.task == 'val':
            with open(self.data_paths[idx], 'rb') as f:
                data_dict = pickle.load(f)

            image = data_dict['image']

            image_rgb_pil = Image.fromarray(
                image.astype(dtype=np.uint8)
            ).convert('RGB')

            image_tensor = self.fn_transform(image_rgb_pil).unsqueeze(dim=0)

            int_label = torch.tensor(data_dict['integer_label']).unsqueeze(dim=0)
            one_hot_label = torch.tensor(data_dict['one_hot_label']).unsqueeze(dim=0)
            class_name = data_dict['class']
            paths = data_dict['original_path']

            return image_tensor, one_hot_label, int_label, class_name, paths
        elif self.task == 'test':
            try:
                im = cv2.imread(self.data_paths[idx])
                assert im is not None, f"opencv cannot read image correctly or {self.data_paths[idx]} not exists"
            except:
                im = cv2.cvtColor(np.asarray(Image.open(self.data_paths[idx])), cv2.COLOR_RGB2BGR)
                assert im is not None, f"Image Not Found {self.data_paths[idx]}, workdir: {os.getcwd()}"

            image_rgb_pil = Image.fromarray(
                im.astype(dtype=np.uint8)
            ).convert('RGB')

            image_tensor = self.fn_transform(image_rgb_pil).unsqueeze(dim=0)

            return image_tensor

    def get_transformer(self):
        if self.augmentation:
            if self.task == 'train':
                fn_trans = transforms.Compose(
                    [
                        transforms.Resize(self.img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                    ]
                )
            else:
                fn_trans = transforms.Compose(
                    [
                        transforms.Resize(self.img_size),
                        transforms.ToTensor(),
                    ]
                )
        else:
            fn_trans = transforms.Compose(
                [
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ]
            )

        return fn_trans

    @staticmethod
    def collate_fn(batch):
        inputs, one_hot_labels, int_labels, class_names, paths = zip(*batch)

        inputs         = torch.cat(inputs, dim=0)
        one_hot_labels = torch.cat(one_hot_labels, dim=0)
        int_labels     = torch.cat(int_labels, dim=0)
        class_names    = list(class_names)
        paths          = list(paths)

        return inputs, one_hot_labels, int_labels, class_names, paths


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    object = data_class(
        path='../data',
        height=512,
        width=512,
        augmentation=True,
        task='train'
    )

    img, label = object.__getitem__(10)

    loader = DataLoader(
        object,
        batch_size=5,
        shuffle=True,
        collate_fn=data_class.collate_fn
    )

    for batch_id, data in enumerate(loader):
        image, label = data[0], data[1]
