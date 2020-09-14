#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model import MNIST
from dataset import decode_idx1_ubyte, decode_idx3_ubyte, RoadData
from options import Options
from utils import save_checkpoints
from metrics import MetricTracker


class Trainer():
    def __init__(self, args):

        # arg parameters
        self.device = args.device
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Network
        self.model = MNIST()

        # Criterion
        self.criterion = nn.BCELoss()

        # optim
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5)

        # Dataset
        train_image = decode_idx3_ubyte('./train-images.idx3-ubyte')
        train_label = decode_idx1_ubyte('./train-labels.idx1-ubyte')
        valid_image = decode_idx3_ubyte('./t10k-images.idx3-ubyte')
        valid_label = decode_idx1_ubyte('./t10k-labels.idx1-ubyte')

        self.train_set = RoadData(train_image, train_label)
        self.valid_set = RoadData(valid_image, valid_label)

        # DataLoader
        self.train_load = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True)
        self.valie_load = DataLoader(
            self.valid_set, batch_size=1, shuffle=False)

        # resume
        if args.resume:
            if os.path.isfile(args.resume):
                print(f'Resuming training,loading {args.resume} ...')
                self.model.load_state_dict(torch.load(args.resume)['model'])
            else:
                print('input resume error', args.resume)

    def train(self):
        tt = time.time()
        best_loss = 1
        self.model.train()
        self.model.to(self.device)

        self.writer = SummaryWriter(comment='MNIST')
        print("*******Start training*******")
        for epoch in range(self.epochs):
            print(f"---Epoch:{epoch+1}/{self.epochs}---")
            train_metrics = self._train(epoch)
            valid_metrics = self._valid(epoch)
            self.model.train()

            # save model
            is_best = valid_metrics['valid_loss'] < best_loss
            state = {'epoch': epoch,
                     'model': self.model.state_dict(),
                     'optimozer': self.optimizer.state_dict(),
                     'best_loss': best_loss}
            save_checkpoints(state, is_best, save_dir='MNIST')
        time_ellapser = time.time()-tt
        self.writer.close()
        print(
            f'Training complete in {time_ellapsed//60}m {time_ellapsed % 60}s')

    def _train(self, epoch):
        step = 0
        metric = MetricTracker()
        for idx, sample in enumerate(self.train_load):
            step += 1
            self.optimizer.zero_grad()
            img = sample['img'].to(self.device)
            lab = sample['lab'].float()
            lab.to(self.device)
            out = self.model(img)
            loss = self.criterion(out, lab)

            # backward
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # updata acc&avg
            metric.update_avg(loss)
            metric.update_acc(out, lab)

            print(f"train--step:{step}/epoch:{epoch+1}--",
                  f"train_loss: {metric.avg:.4f}",
                  f"acc:{metric.acc:.4f}",
                  f"lr: {self.lr_scheduler.get_lr()[0]: .2f}")
            # tensorboard
            self.writer.add_scalar('train_loss', metric.avg, step)
        print(f'---Metrics in {epoch+1}/{self.epochs}---',
              f'Training Loss : {metric.avg}',
              f'Acc : {metric.acc}')

        return {'loss': metric.avg, 'acc': metric.acc}

    def _valid(self, epoch):
        step = 0
        metric = MetricTracker()
        self.model.eval()
        for idx, sample in enumerate(self.valie_load):
            step += 1
            img = sample['img'].to(self.device)
            lab = sample['lab'].float()
            lab.to(self.device)
            out = self.model(img)
            loss = self.criterion(out, lab)

            # update acc&avg
            metric.update_avg(loss)
            metric.update_acc(out, lab)

            if step % 500 == 0:
                print(f"valid--step:{step}/epoch:{epoch+1}--",
                      f"valid_loss:{metric.avg:.4f}",
                      f"acc:{metric.acc:.4f}")

            self.writer.add_scalar('valid_loss', metric.avg, step)
        print(f'----Valid---',
              f'Valid_loss:{metric.avg}',
              f'Acc:{metric.acc}')
        return {'valid_loss': metric.avg, 'acc': metric.acc}


if __name__ == '__main__':
    args = Options().parse()
    trainer = Trainer(args)
    trainer.train()
