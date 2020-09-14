#!/usr/bin/env python3
# _*_ coding:utf-8 _*_
# author:CT
    
import argparse
import torch
import torch.nn as nn


class Options():

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--no_cuda',
                            action='store_true',
                            default=False,
                            help='Disables cuda training')
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='learning rate')
        parser.add_argument('--epochs',
                            type=int,
                            default=100,
                            help='epochs for train')
        parser.add_argument('--batch_size',
                            type=int,
                            default=150,
                            help='batch_size')
        parser.add_argument('--resume',
                            type=str,
                            default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        return args


if __name__ == '__main__':
    args = Options().parse()
    print(args.lr)
