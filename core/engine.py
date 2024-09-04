import os
import sys
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_loader import data_class
from utils.events import write_tbimg, write_scalar, write_tbPR


class Trainer:
    def __init__(self, args, config, device=torch.device('cpu')):
        super().__init__()
        self.cfg = config
        self.args = args
        self.device = device
        #
        self.save_path = self.make_save_path()
        #
        self.tblogger = SummaryWriter(self.save_path)

        # data loader
        self.train_loader, self.valid_loader = self.get_dataloader()

        # model
        self.model = self.build_model()

        # optimizer
        self.optimizer = self.build_optimizer()

        # scheduler
        self.scheduler = self.build_scheduler()

        # loss
        self.compute_loss = self.build_criterion()

        # parameters
        self.max_epoch = self.args.epoch
        self.max_stepnum = len(self.train_loader)
        #
        self.best_score = 0.0
        self.best_model = None

    def get_dataloader(self):
        # Data loader configurations
        height, width = self.args.height, self.args.width
        batch_size = self.args.batch_size
        num_workers = self.args.num_workers
        #
        train_path = self.args.train_path
        train_object = data_class(
            path=train_path,
            height=height,
            width=width,
            task='train'
        )
        train_loader = DataLoader(
            train_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_class.collate_fn
        )
        #
        val_path = self.args.valid_path
        valid_object = data_class(
            path=val_path,
            height=height,
            width=width,
            task='val'
        )
        valid_loader = DataLoader(
            valid_object,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=data_class.collate_fn
        )
        #
        self.num_class = train_object.num_class
        #
        return train_loader, valid_loader

    def make_save_path(self):
        # save base path~ name
        save_path = os.path.join(self.cfg['path']['save_base_path'],
                                 self.cfg['model']['name'])
        os.makedirs(save_path, exist_ok=True)
        return save_path

    def build_model(self):
        # Model Configurations
        model_name = self.cfg['model']['name']
        num_class = self.num_class
        #
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
        #
        print(f'Model 초기화 완료 !!\n'
              f'Model : {model_name}\n')
        #
        return model

    def build_optimizer(self):
        #
        from solver.fn_optimizer import build_optimizer
        optimizer = build_optimizer(self.cfg, self.model)
        #
        return optimizer

    def build_scheduler(self):
        sched_name = self.cfg['scheduler']['name']
        if sched_name == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                gamma=0.9,
                step_size=5
            )
        elif sched_name is None:
            scheduler = None
        else:
            raise NotImplementedError
        print(f'Scheduler 초기화 완료 !!\n'
              f'Scheduler : {sched_name}\n')
        return scheduler

    def build_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def start_train(self):
        try:
            for epoch in range(self.max_epoch):
                self.train_one_epoch(epoch)
                self.valid_one_epoch(epoch)
        except:
            print('Error in training loop...')
            raise

    def train_one_epoch(self, epoch):
        #
        print(f'\nTrain start : {epoch} / {self.max_epoch}')
        pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum)
        #
        TP = np.zeros(self.num_class)
        FP = np.zeros(self.num_class)
        FN = np.zeros(self.num_class)
        TN = np.zeros(self.num_class)
        #
        correct = 0
        total = 0
        #
        self.model.train()
        for step, batch_data in pbar:
            images = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            int_labels = batch_data[2].to(self.device)
            cls_names = batch_data[3]
            #
            output = self.model(images)
            #
            loss = self.compute_loss(output, labels.float())
            #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #
            _, predict = torch.max(output, 1)
            #
            # Get statistics
            TP, FP, FN = self.get_statistics(
                self.model.predict(output.detach()), labels,
                TP, FP, FN
            )
            #
            total += labels.size(0)
            correct += (predict == int_labels).sum().item()
            #
            pbar.set_postfix(loss=round(loss.item(), 2), acc=round(correct / total, 2))
            if step % 2 == 0:
                # Scalar print
                write_scalar(   # loss
                    self.tblogger, loss.detach().cpu(), (epoch * self.max_epoch + step), title_text='training/loss'
                )
                write_scalar(  # accuracy
                    self.tblogger, (correct / total), (epoch * self.max_epoch + step), title_text='training/accuracy'
                )
                # Image print
                write_tbimg(
                    self.tblogger, images.detach().cpu(),
                    step, cls_names, self.train_loader.dataset.le.inverse_transform(predict.cpu())
                )
        # PR curve print
        write_tbPR(self.tblogger, TP, FP, FN, epoch, 'train')
        #
        if self.scheduler is not None:
            self.scheduler.step()
        #

    def valid_one_epoch(self, epoch):
        print(f'\nValidation start : {epoch} / {self.max_epoch}')
        pbar = tqdm(enumerate(self.valid_loader), total=self.max_stepnum)
        #
        correct = 0
        total = 0
        #
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in pbar:
                images = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                int_labels = batch_data[2].to(self.device)
                cls_names  = batch_data[3]
                #
                outputs = self.model(images)
                #
                _, predict = torch.max(outputs, 1)
                #
                total += labels.size(0)
                correct += (predict == int_labels).sum().item()
                #
                pbar.set_postfix(acc=round(correct / total, 2))
                if step % 2 == 0:
                    # Scalar print
                    write_scalar(  # accuracy
                        self.tblogger, (correct / total), (epoch * self.max_epoch + step),
                        title_text='validation/accuracy'
                    )
                    # Image print
                    write_tbimg(
                        self.tblogger, images.detach().cpu(),
                        step, cls_names, self.valid_loader.dataset.le.inverse_transform(predict.cpu())
                    )

        val_acc = correct / total
        if val_acc > self.best_score:
            self.best_score = val_acc
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))

    @staticmethod
    def get_statistics(pred, true, TP, FP, FN):
        for defect_idx in range(pred.shape[1]):
            pred_per_defect = pred[:, defect_idx].cpu().detach().numpy()
            true_per_defect = true[:, defect_idx].cpu().detach().numpy()

            TP[defect_idx] += np.sum(pred_per_defect * true_per_defect)
            FP[defect_idx] += np.sum(pred_per_defect * (1 - true_per_defect))
            FN[defect_idx] += np.sum((1 - pred_per_defect) * true_per_defect)

        return TP, FP, FN
