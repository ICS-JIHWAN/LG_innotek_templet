import time
import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_loader import data_class
from utils.events import write_tbimg, write_tbloss, write_tbPR


class Trainer:
    def __init__(self, config, device=torch.device('cpu')):
        super().__init__()
        self.cfg = config
        self.device = device
        #
        self.save_path = self.make_save_path()
        #
        self.tblogger = SummaryWriter(self.save_path)

        # data loader
        self.train_loader = self.get_dataloader()

        # model
        self.model = self.build_model()

        # optimizer
        self.optimizer = self.build_optimizer()

        # scheduler
        self.scheduler = self.build_scheduler()

        # loss
        self.compute_loss = self.build_criterion()

        # parameters
        self.max_epoch = self.cfg['solver']['max_epoch']
        self.max_stepnum = len(self.train_loader)

    def get_dataloader(self):
        height, width = self.cfg['dataset']['height'], self.cfg['dataset']['width']
        batch_size = self.cfg['dataset']['batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        #
        train_path = self.cfg['dataset']['train_path']
        train_object = data_class(
            path=train_path,
            height=height,
            width=width,
            task='train'
        )
        train_loader = DataLoader(
            train_object,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=data_class.collate_fn
        )

        return train_loader

    def make_save_path(self):
        # save base path~ name
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name'])
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def build_model(self):
        model_name = self.cfg['model']['name']
        num_class = self.cfg['model']['num_class']
        if model_name == 'lenet':
            from model.lenet import lenet
            model = lenet(num_class=num_class).to(self.device)
        elif model_name == 'alexnet':
            from model.alexnet import alexnet
            model = alexnet(num_class=num_class).to(self.device)
        elif model_name == 'resnet':
            from model.resnet import resnet
            model = resnet(num_class=num_class).to(self.device)
        else:
            print('Model load fail')
            raise NotImplementedError
        return model

    def build_optimizer(self):
        from solver.fn_optimizer import build_optimizer
        optimizer = build_optimizer(self.cfg, self.model)
        return optimizer

    def build_scheduler(self):
        if self.cfg['scheduler']['name'] == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                gamma=0.9,
                step_size=5
            )
        else:
            raise NotImplementedError
        return scheduler

    def build_criterion(self):
        return torch.nn.BCEWithLogitsLoss()

    def start_train(self):
        try:
            for epoch in range(self.max_epoch):
                self.train_one_epoch(epoch)
        except:
            print('Error in training loop...')
            raise

    def train_one_epoch(self, epoch):
        pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum)
        #
        TP = np.zeros(8)
        FP = np.zeros(8)
        FN = np.zeros(8)
        TN = np.zeros(8)
        #
        for step, batch_data in pbar:
            images = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            #
            output = self.model(images)
            #
            loss = self.compute_loss(output, labels.float())
            #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Get statistics
            TP, FP, FN = self.get_statistics(
                self.model.predict(output.detach()), labels,
                TP, FP, FN
            )
            #
            if step % 2 == 0:
                write_tbloss(self.tblogger, loss.detach().cpu(), (epoch * self.max_epoch + step))

        write_tbPR(self.tblogger, TP, FP, FN, epoch, 'train')

        self.scheduler.step()

        print('finish one epoch')

    @staticmethod
    def get_statistics(pred, true, TP, FP, FN):
        for defect_idx in range(pred.shape[1]):
            pred_per_defect = pred[:, defect_idx].cpu().detach().numpy()
            true_per_defect = true[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(pred_per_defect * true_per_defect)
            FP[defect_idx] += np.sum(pred_per_defect * (1 - true_per_defect))
            FN[defect_idx] += np.sum((1 - pred_per_defect) * true_per_defect)

        return TP, FP, FN
