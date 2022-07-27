#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from configparser import Interpolation
import glob
import os
import pdb
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import io
from torch.utils import data
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
    
class PascalPartsPerson(data.Dataset):
    def __init__(self, root, split, resize_size=(224,224), pose_classes=14):
        self.root = root
        self.resize_size = resize_size
        self.split = split
        self.file_names = os.listdir(os.path.join(self.root,self.split,'image'))
        self.pose_classes = pose_classes

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        item = self.file_names[index]

        im_path = os.path.join(self.root, self.split , 'image', item)
        # seg_path = os.path.join(self.root, self.split , 'segmentation', item)
        seg_path = os.path.join(self.root, self.split , 'segmentation_ref', item)

        image = self.image_resize(Image.open(im_path))
        seg = self.label_resize(Image.open(seg_path))
        # image = Image.open(im_path)
        # seg = Image.open(seg_path)

        if self.split == 'train':
            image = self.augmentation(image)
            seg = self.augmentation(seg) * 255
        elif self.split == 'val':
            image = self.totensor(image)
            seg = self.totensor(seg) * 255
    
        image = self.rgb_normalize(image)
        seg = seg.long()

        return image, seg
    
    def image_resize(self,input_image):
        return transforms.Resize(self.resize_size)(input_image)

    def label_resize(self,input_image):
        return transforms.Resize(self.resize_size, interpolation=Image.NEAREST)(input_image)

    def augmentation(self,input_image): # Fix here
        return transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(45),
            transforms.ToTensor(), # Do not delete this line
            ])(input_image)

    def totensor(self,input_image):
        return transforms.ToTensor()(input_image)

    def rgb_normalize(self,input_image):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(input_image)
                                
    def array2image(self,pose):
        pose_image = torch.zeros((self.resize_size[0]+500, self.resize_size[1]+500))
        for i in range(pose.shape[0]):
            pose_image[int(torch.round(pose[i,0])),int(torch.round(pose[i,1]))]= i+1
        pose_image = pose_image[:self.resize_size[0],:self.resize_size[1]]
        pose_image = pose_image.numpy()
        pose_image = transforms.ToPILImage()(pose_image)
        return pose_image

    def image2array(self, pose_image):
        pose_image = pose_image.squeeze()                
        nonzero_indices = torch.nonzero(pose_image)
        pose_array = torch.zeros(self.pose_classes,2)
        for num in range(nonzero_indices.shape[0]):
            joint_index = pose_image[nonzero_indices[num][0],nonzero_indices[num][1]] - 1
            pose_array[joint_index.long()] = nonzero_indices[num]
        return pose_array

if __name__ == '__main__':
    # args
    resize_size = (224, 224)
    seg_classes = 7
    pose_classes = 14

    train_dst = PascalPartsPerson(root='./PPP', split='train', resize_size = resize_size)
    val_dst = PascalPartsPerson(root='./PPP', split='val', resize_size = resize_size)

    train_loader = data.DataLoader(train_dst, batch_size=32, shuffle=False, num_workers=2)
    val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False, num_workers=2)

    for i, (image, label) in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()

    pdb.set_trace()        