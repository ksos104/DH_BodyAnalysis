#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import glob
import os
import random
import sys

import cv2
# from utils import add_void
# from utils.ext_transforms import get_affine_transform
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils import data

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pdb

# from ext_transforms import get_affine_transform
from torchvision.transforms import transforms
from utils.ext_transforms import get_affine_transform
from utils.utils import add_void

val_old = np.arange(0,25)

# 14 labels
# val_new = np.array([0,
#                     2, 12, 9, 2, 13, 10,
#                     2, 14, 11, 2, 14, 11,
#                     2, 2, 2, 1, 6, 3,
#                     7, 4, 8, 5, 8, 5])

# 7 labels
val_new = np.array([0, 
                    2, 5, 5, 2, 6, 6,
                    2, 6, 6, 2, 6, 6,
                    1, 2, 2, 1, 3, 3,
                    4, 4, 4, 4, 4, 4])



class SURREAL(data.Dataset):
    def __init__(self, root, split, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=None, void_pixels=0, return_edge=False):
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = flip_prob
        self.transform = transform
        self.dataset = split
        self.void_pixels = void_pixels
        self.return_edge = return_edge
        
        self.file_names = os.listdir(os.path.join(self.root,self.dataset,'image'))

    def __len__(self):
        return len(self.file_names)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        item = self.file_names[index]

        im_path = os.path.join(self.root, self.dataset , 'image', item)
        parsing_anno_path = os.path.join(self.root, self.dataset , 'segmentation', item)


        im = cv2.imread(im_path, cv2.IMREAD_COLOR)[..., ::-1]
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

            
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        # Get pose annotation
        # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
        if self.dataset != 'no_gt':
            parsing_anno = np.array(Image.open(parsing_anno_path))

        arr = np.empty(len(val_new), dtype=val_new.dtype)
        arr[val_old] = val_new
        if self.dataset != 'no_gt':
            parsing_anno = arr[parsing_anno]
            parsing_anno = np.uint8(parsing_anno)

        if self.dataset == 'train' or self.dataset == 'trainval':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

            if random.random() <= self.flip_prob:
                im = im[:, ::-1, :]
                parsing_anno = parsing_anno[:, ::-1]
                person_center[0] = im.shape[1] - person_center[0] - 1
                # right_idx = [15, 17, 19]
                # left_idx = [14, 16, 18]
                # for i in range(0, 3):
                #     right_pos = np.where(parsing_anno == right_idx[i])
                #     left_pos = np.where(parsing_anno == left_idx[i])
                #     parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                #     parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.return_edge:
            width = 10
            # label_edge = cv2.Canny(parsing_anno, 0, 0)  # all parts edge
            label_edge = cv2.Canny(((parsing_anno > 0) * 1).astype('uint8'), 0, 0)  # whole human edge
            label_edge = cv2.dilate(label_edge, np.ones((width, width)))
            input_edge = cv2.Canny(input, 100, 200, L2gradient=True)

            edge_parsing = cv2.warpAffine(
                label_edge,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)

            edge_parsing = (edge_parsing.astype(int) * input_edge.astype(int)).clip(max=255).astype('uint8')
            edge_parsing = cv2.dilate(edge_parsing, np.ones((3, 3))).clip(max=1)

            edge_parsing = torch.from_numpy(edge_parsing)

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset != 'no_gt':
            if self.void_pixels > 0:
                parsing_anno = add_void(parsing_anno, width=self.void_pixels, void_value=self.ignore_label)
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                # borderValue=(self.ignore_label)
                )
            label_parsing = torch.from_numpy(label_parsing)

        if self.return_edge:
            return input, (label_parsing, edge_parsing)

        if self.dataset == 'test' or self.dataset == 'sample':
            return input, meta, label_parsing
        elif self.dataset == 'no_gt':
            return input, meta
        else:
            return input, label_parsing


if __name__ == '__main__':
    from tqdm import tqdm
    num_classes = 7

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return_edge = False # if 'edge' in opts.model else False
    train_dst = SURREAL(root='../../datasets/SURREAL/data/ref_cmu', split='train', crop_size=[512, 512], scale_factor=0.25,
                                       rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
                                       void_pixels=3, return_edge=return_edge)

    train_loader = data.DataLoader(train_dst, batch_size=1, shuffle=False, num_workers=2)

    val_dst = SURREAL(root='../../datasets/SURREAL/data/ref_cmu', split='test', crop_size=[512, 512], scale_factor=0.25,
                                       rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
                                       void_pixels=3, return_edge=return_edge)

    val_loader = data.DataLoader(train_dst, batch_size=1, shuffle=False, num_workers=2)


    weights = np.zeros(num_classes)
    for (images, labels) in tqdm(val_loader):
        pdb.set_trace()

        labels = labels.numpy()
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(labels == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weight = []
        for i in range(num_classes):
            weight.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weight = np.array(weight, dtype=np.float)
        weights += weight
        # weights = torch.from_numpy(weights).float().to(masks.device)
    print(weights / len(train_loader))
