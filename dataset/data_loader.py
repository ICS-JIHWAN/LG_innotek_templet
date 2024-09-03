import os
import glob
import numpy as np
import time

import torch
import torchvision.transforms as transforms

from PIL import Image
from scipy import io
from dataset.make_dataset import image_to_mat
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
        # 이미지 .mat 파일로 변경
        self.num_class = image_to_mat(path)
        self.data_paths = sorted(glob.glob(os.path.join(path, '*/*.mat')))
        #
        self.fn_transform = self.get_transformer()
        #
        end = time.time()
        # logger 출력
        print(f'Data loader 초기화 완료 !! {end - start:.5f} sec...\n'
              f'총 학습 데이터 수 : {self.__len__()}\n')

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_dict = io.loadmat(self.data_paths[idx])

        image = data_dict['image']

        image_rgb_pil = Image.fromarray(
            image.astype(dtype=np.uint8)
        ).convert('RGB')

        image_tensor = self.fn_transform(image_rgb_pil).unsqueeze(dim=0)

        label_numpy = torch.tensor(data_dict['one_hot_label'])

        return image_tensor, label_numpy

    def get_transformer(self):
        if self.augmentation:
            if self.task == 'train':
                fn_trans = transforms.Compose(
                    [
                        transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomRotation(degrees=(0, 90)),
                        transforms.ToTensor(),
                    ]
                )
            else:
                fn_trans = transforms.Compose(
                    [
                        transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.ToTensor(),
                    ]
                )
        else:
            fn_trans = transforms.Compose(
                [
                    transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                ]
            )

        return fn_trans

    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return inputs, labels


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
