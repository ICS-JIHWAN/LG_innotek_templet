import os
import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score,
    precision_recall_curve, roc_auc_score,
    confusion_matrix
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.data_loader import data_class
from cam_algorithms import GradCAM
from utils import (
    write_tbimg, write_scalar, write_tbPR, write_tbCM,
    show_cam_on_image
)

# This should work for any tensorflow > 1.14
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
# or you can use below codes
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

class Trainer:
    def __init__(self, args, config, device=torch.device('cpu')):
        super().__init__()
        # configuration
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

        # GradCAM 설정
        target_layer = self.model.conv5_x[-1].residual_function[-2]  # Target layer 설정
        self.cam = GradCAM(self.model, target_layer=target_layer)

        # optimizer
        self.optimizer = self.build_optimizer()

        # scheduler
        self.scheduler = self.build_scheduler()

        # loss
        self.compute_loss = self.build_criterion()

        #
        self.df_mismatch = pd.DataFrame()
        #
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
            #
            # mismatch data save
            self.df_mismatch.to_excel(os.path.join(self.save_path, 'mismatch_data.xlsx'))
        except:
            print('Error in training loop...')
            raise

    def train_one_epoch(self, epoch):
        #
        print(f'\nTrain start : {epoch} / {self.max_epoch}')
        pbar = tqdm(enumerate(self.train_loader), total=self.max_stepnum)
        #
        # mismatch dataframe for one epoch
        mismatch_data_paths = []
        mismatch_data_label = []
        mismatch_data_pred  = []
        #
        # metrix score for one epoch
        y_true_int_list    = []
        y_true_onehot_list = []
        y_prob_list = []
        y_pred_list = []
        cm = np.zeros((self.num_class, self.num_class))
        #
        losses = 0
        correct = 0
        total = 0
        #
        self.model.train()
        for step, batch_data in pbar:
            # Data info
            images     = batch_data[0].to(self.device)  # Image
            labels     = batch_data[1].to(self.device)
            int_labels = batch_data[2].to(self.device)
            cls_names  = batch_data[3]
            paths      = batch_data[4]
            #
            # Forward propagation
            output = self.model(images)
            #
            loss = self.compute_loss(output, labels.float())
            #
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #
            # Prediction info
            _, predict = torch.max(output, 1)
            predict_cls_names = self.train_loader.dataset.le.inverse_transform(predict.cpu())
            #
            y_true_int_list.append(int_labels.cpu().numpy())
            y_true_onehot_list.append(labels.cpu().numpy())
            y_prob_list.append(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())
            y_pred_list.append(predict.cpu().numpy())
            #
            # Mismatch data info
            mis_match_data = (int_labels != predict).cpu().numpy()
            mismatch_data_paths.extend(np.array(paths)[mis_match_data])
            mismatch_data_label.extend(np.array(cls_names)[mis_match_data])
            mismatch_data_pred.extend(predict_cls_names[mis_match_data])
            #
            # Get statistics
            cm += confusion_matrix(cls_names, predict_cls_names, labels=self.train_loader.dataset.le.classes_)
            #
            losses += loss.item()
            total += labels.size(0)
            correct += (predict == int_labels).sum().item()
            #
            pbar.set_postfix(loss=round(loss.item(), 2), acc=round(correct / total, 2))
            if step % self.args.tb_step == 0: # loss, accuracy, image 스텝마다 보고 싶은 경우
            #     # Scalar print
            #     write_scalar(  # Loss
            #         self.tblogger, loss.detach().cpu(), (epoch * self.max_stepnum + step), title_text='training/loss'
            #     )
            #     write_scalar(  # Accuracy
            #         self.tblogger, (correct / total), (epoch * self.max_stepnum + step), title_text='training/accuracy'
            #     )
                # Image print
                write_tbimg(    # Original Image
                    self.tblogger, images.detach().cpu(),
                    (epoch * self.max_stepnum + step), real_classes=cls_names, pred_classes=predict_cls_names
                )
        #
        # Mismatch dataframe
        df_mismatch_epoch = pd.DataFrame({
            'epoch': [epoch+1]*len(mismatch_data_paths),
            'path': mismatch_data_paths,
            'true_label': mismatch_data_label,
            'pred_label': mismatch_data_pred
        })
        self.df_mismatch = pd.concat((self.df_mismatch, df_mismatch_epoch))
        #
        # Confusion Matrix
        write_tbCM(self.tblogger, cm, class_names=self.train_loader.dataset.le.classes_, step=epoch, task='train')
        #
        # PR curve print
        y_pred = np.concatenate(y_pred_list)
        y_true = np.concatenate(y_true_int_list)
        y_pred_prob   = np.concatenate(y_prob_list)
        y_true_onehot = np.concatenate(y_true_onehot_list)
        #
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class    = recall_score(y_true, y_pred, average=None)
        #
        write_tbPR( # Precision & Recall per classes
            self.tblogger, precision_per_class, recall_per_class,
            class_names=self.train_loader.dataset.le.classes_, step=epoch, task='train'
        )
        #
        for i in range(self.num_class):
            self.tblogger.add_pr_curve(
                f'train/PR Curve/{self.train_loader.dataset.le.classes_[i]}',
                y_true_onehot[:, i], y_pred_prob[:, i], global_step=epoch
           )
        #
        # Scalar print
            write_scalar(  # Loss
                self.tblogger, (losses/total), epoch, title_text='training/loss'
            )
            write_scalar(  # Accuracy
                self.tblogger, (correct / total), epoch, title_text='training/accuracy'
            )
        #
        # Scheduler update
        if self.scheduler is not None:
            self.scheduler.step()
        #

    def valid_one_epoch(self, epoch):
        print(f'\nValidation start : {epoch} / {self.max_epoch}')
        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        #
        # metrix score for one epoch
        y_true_int_list = []
        y_true_onehot_list = []
        y_prob_list = []
        y_pred_list = []
        cm = np.zeros((self.num_class, self.num_class))
        #
        correct = 0
        total = 0
        #
        self.model.eval()
        with torch.no_grad():
            for step, batch_data in pbar:
                # Data info
                images = batch_data[0].to(self.device)  # Image
                labels = batch_data[1].to(self.device)
                int_labels = batch_data[2].to(self.device)
                cls_names = batch_data[3]
                #
                # Forward propagation
                output = self.model(images)
                #
                # Prediction info
                _, predict = torch.max(output, 1)
                predict_cls_names = self.train_loader.dataset.le.inverse_transform(predict.cpu())
                #
                y_true_int_list.append(int_labels.cpu().numpy())
                y_true_onehot_list.append(labels.cpu().numpy())
                y_prob_list.append(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy())
                y_pred_list.append(predict.cpu().numpy())
                #
                # Get statistics
                cm += confusion_matrix(cls_names, predict_cls_names, labels=self.train_loader.dataset.le.classes_)
                #
                total += labels.size(0)
                correct += (predict == int_labels).sum().item()
                #
                pbar.set_postfix(acc=round(correct / total, 2))
                if step % self.args.tb_step == 0:
                    # # Scalar print
                    # write_scalar(  # Accuracy
                    #     self.tblogger, (correct / total), (epoch * self.max_epoch + step),
                    #     title_text='validation/accuracy'
                    # )
                    # Image print
                    write_tbimg(    # Original Image
                        self.tblogger, images.detach().cpu(),
                        step, real_classes=cls_names,
                        pred_classes=self.valid_loader.dataset.le.inverse_transform(predict.cpu()),
                        task='validation'
                    )

        #
        # Confusion Matrix
        write_tbCM(self.tblogger, cm, class_names=self.train_loader.dataset.le.classes_, step=epoch,
                   task='validation')
        #
        # PR curve print
        y_pred = np.concatenate(y_pred_list)
        y_true = np.concatenate(y_true_int_list)
        y_pred_prob = np.concatenate(y_prob_list)
        y_true_onehot = np.concatenate(y_true_onehot_list)
        #
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        #
        write_tbPR(  # Precision & Recall per classes
            self.tblogger, precision_per_class, recall_per_class,
            class_names=self.train_loader.dataset.le.classes_, step=epoch, task='validation'
        )
        #
        write_scalar(  # Accuracy
            self.tblogger, (correct / total), epoch,
            title_text='validation/accuracy'
        )
        #
        for i in range(self.num_class):
            self.tblogger.add_pr_curve(
                f'validation/PR Curve/{self.train_loader.dataset.le.classes_[i]}',
                y_true_onehot[:, i], y_pred_prob[:, i], global_step=epoch
            )
        #
        # best model save
        val_acc = correct / total
        if val_acc > self.best_score:
            self.best_score = val_acc
            save_dict = {
                'model': self.model.state_dict(),
                'lb_encoder': self.train_loader.dataset.le
            }
            torch.save(save_dict, os.path.join(self.save_path, 'best_model.pth'))

