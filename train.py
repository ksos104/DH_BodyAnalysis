#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   train.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import timeit
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils import data

import networks
import utils.schp as schp
# from datasets.datasets import LIPDataSet
# from datasets.pascal_loader_orig import PascalPartsPerson
from datasets.datasets_PPP import PascalPartSegmentation
from datasets.datasets_SURREAL_v2 import SURREAL
from datasets.target_generation import generate_edge_tensor
from utils.transforms import BGR2RGB_transform
from utils.criterion import CriterionAll
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.warmup_scheduler import SGDRScheduler

import neptune.new as neptune

from networks.AugmentCE2P_t2t_vit import t2tnet

npt = neptune.init(
    project="kaist-cilab/DH-FaceAnalysis",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4OTQ2MGY0Yi0zMTM2LTQ5ZmEtYjlmOS1lNmQxMTliOTE0MjkifQ==",
)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet18')
    # Data Preference
    # parser.add_argument("--data-dir", type=str, default='/mnt/server8_hard3/msson/datasets/Pascal Part Person')
    parser.add_argument("--data-dir", type=str, default='/mnt/server14_hard0/msson/datasets/SURREAL/data/ref_cmu')
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=str, default='473,473')
    # parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Training Strategy
    parser.add_argument("--learning-rate", type=float, default=7e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    # parser.add_argument("--gpu", type=str, default='0,1,2')
    parser.add_argument("--gpu", type=str, default='2,7')
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-epochs", type=int, default=10)
    parser.add_argument("--imagenet-pretrain", type=str, default='./pretrain_model/resnet18-imagenet.pth')
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log/checkpoint.pth.tar')
    parser.add_argument("--schp-start", type=int, default=100, help='schp start epoch')
    parser.add_argument("--cycle-epochs", type=int, default=10, help='schp cyclical epoch')
    parser.add_argument("--schp-restore", type=str, default='./log/schp_checkpoint.pth.tar')
    parser.add_argument("--lambda-s", type=float, default=1, help='segmentation loss weight')
    parser.add_argument("--lambda-e", type=float, default=1, help='edge loss weight')
    parser.add_argument("--lambda-c", type=float, default=0.1, help='segmentation-edge consistency loss weight')
    return parser.parse_args()


'''
    전체 dataset에 대해 i와 u 합산하여 mIoU 계산
'''
epsilon = 1e-4
NUM_CLASSES = 7

def cal_miou_total(result, gt):                ## resutl.shpae == gt.shape == [batch_size, 512, 512]    
    tensor1 = torch.Tensor([1]).to(gt.device)
    tensor0 = torch.Tensor([0]).to(gt.device)

    res_i = torch.zeros((NUM_CLASSES)).to(result.device)
    res_u = torch.zeros((NUM_CLASSES)).to(result.device)

    for idx in range(NUM_CLASSES):
        u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
        i = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()

        res_i[idx] += i
        res_u[idx] += u

    return res_i, res_u


def main():
    args = get_arguments()
    print(args)

    start_epoch = 0
    cycle_n = 0

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)

    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    input_size = list(map(int, args.input_size.split(',')))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Model Initialization
    AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    # AugmentCE2P = t2tnet(num_classes=args.num_classes, pretrained='pretrain_model/81.7_T2T_ViTt_14.pth.tar')
    model = DataParallelModel(AugmentCE2P)
    model.cuda()

    IMAGE_MEAN = AugmentCE2P.mean
    IMAGE_STD = AugmentCE2P.std
    INPUT_SPACE = AugmentCE2P.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))

    restore_from = args.model_restore
    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))
        checkpoint = torch.load(restore_from)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    SCHP_AugmentCE2P = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=args.imagenet_pretrain)
    # SCHP_AugmentCE2P = t2tnet(num_classes=args.num_classes, pretrained='pretrain_model/81.7_T2T_ViTt_14.pth.tar')
    schp_model = DataParallelModel(SCHP_AugmentCE2P)
    schp_model.cuda()

    if os.path.exists(args.schp_restore):
        print('Resuming schp checkpoint from {}'.format(args.schp_restore))
        schp_checkpoint = torch.load(args.schp_restore)
        schp_model_state_dict = schp_checkpoint['state_dict']
        cycle_n = schp_checkpoint['cycle_n']
        schp_model.load_state_dict(schp_model_state_dict)

    # Loss Function
    criterion = CriterionAll(lambda_1=args.lambda_s, lambda_2=args.lambda_e, lambda_3=args.lambda_c,
                             num_classes=args.num_classes)
    criterion = DataParallelCriterion(criterion)
    criterion.cuda()

    # Data Loader
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    elif INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # train_dataset = LIPDataSet(args.data_dir, 'train', crop_size=input_size, transform=transform)
    # train_dataset = PascalPartSegmentation(root=args.data_dir, split='train', resize_size=input_size)
    # train_dataset = PascalPartSegmentation(root=args.data_dir, split='train', crop_size=input_size,
    #                                    scale_factor=0.25,
    #                                    rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
    #                                    void_pixels=3, return_edge=False)
    train_dataset = SURREAL(root=args.data_dir, split='train', crop_size=input_size,
                                       scale_factor=0.25,
                                       rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
                                       void_pixels=3, return_edge=False)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size * len(gpus),
                                   num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    print('Total training samples: {}'.format(len(train_dataset)))

    # Optimizer Initialization
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_scheduler = SGDRScheduler(optimizer, total_epoch=args.epochs,
                                 eta_min=args.learning_rate / 100, warmup_epoch=10,
                                 start_cyclical=args.schp_start, cyclical_base_lr=args.learning_rate / 2,
                                 cyclical_epoch=args.cycle_epochs)

    total_iters = args.epochs * len(train_loader)
    start = timeit.default_timer()
    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step(epoch=epoch)
        lr = lr_scheduler.get_lr()[0]

        model.train()
        losses= 0
        avg_miou = 0
        total_i = torch.zeros((NUM_CLASSES)).cuda()
        total_u = torch.zeros((NUM_CLASSES)).cuda()
        for i_iter, batch in enumerate(train_loader):
            i_iter += len(train_loader) * epoch

            # images, labels, _ = batch
            images, labels = batch
            labels = labels.cuda(non_blocking=True)
            # labels = labels.squeeze(dim=1)

            edges = generate_edge_tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            edges = edges.type(torch.cuda.LongTensor)

            preds = model(images)

            # Online Self Correction Cycle with Label Refinement
            if cycle_n >= 1:
                with torch.no_grad():
                    soft_preds = schp_model(images)
                    soft_parsing = []
                    soft_edge = []
                    for soft_pred in soft_preds:
                        # soft_parsing.append(soft_pred[0][-1])
                        # soft_edge.append(soft_pred[1][-1])
                        soft_parsing.append(soft_pred[0][-1].to(torch.device('cuda:0')))
                        soft_edge.append(soft_pred[1][-1].to(torch.device('cuda:0')))
                    soft_preds = torch.cat(soft_parsing, dim=0)
                    soft_edges = torch.cat(soft_edge, dim=0)
                    # soft_preds = torch.nn.parallel.gather(soft_parsing, model.device_ids)
                    # soft_edges = torch.nn.parallel.gather(soft_edge, model.device_ids)
            else:
                soft_preds = None
                soft_edges = None

            loss = criterion(preds, [labels, edges, soft_preds, soft_edges], cycle_n)
            losses = losses + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses = losses.detach().cpu()

            if i_iter % 100 == 0:
                print('iter = {} of {} completed, lr = {}, loss = {}'.format(i_iter, total_iters, lr,
                                                                             loss.data.cpu().numpy()))
        
        avg_loss = losses / i_iter
        # avg_miou = torch.sum(total_i / (total_u+epsilon)).item() / NUM_CLASSES

        npt["train/loss"].log(avg_loss)
        # npt["train/mIoU"].log(avg_miou)
        
        if (epoch + 1) % (args.eval_epochs) == 0:
            schp.save_schp_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, False, args.log_dir, filename='checkpoint_{}.pth.tar'.format(epoch + 1))

        # Self Correction Cycle with Model Aggregation
        if (epoch + 1) >= args.schp_start and (epoch + 1 - args.schp_start) % args.cycle_epochs == 0:
            print('Self-correction cycle number {}'.format(cycle_n))
            schp.moving_average(schp_model, model, 1.0 / (cycle_n + 1))
            cycle_n += 1
            schp.bn_re_estimate(train_loader, schp_model)
            schp.save_schp_checkpoint({
                'state_dict': schp_model.state_dict(),
                'cycle_n': cycle_n,
            }, False, args.log_dir, filename='schp_{}_checkpoint.pth.tar'.format(cycle_n))

        torch.cuda.empty_cache()
        end = timeit.default_timer()
        print('epoch = {} of {} completed using {} s'.format(epoch, args.epochs,
                                                             (end - start) / (epoch - start_epoch + 1)))

    end = timeit.default_timer()
    print('Training Finished in {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
