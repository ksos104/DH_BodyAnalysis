#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_log/%j_out.txt
#SBATCH -e slurm_log/%j_err.txt
#SBATCH --gres=gpu:2

CUDA_VISIBLE_DEVICES=1,2 python train.py --gpu 1,2 --arch resnet18 --imagenet-pretrain ./pretrain_model/resnet18-imagenet.pth
