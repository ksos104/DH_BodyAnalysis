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

import matplotlib.pyplot as plt
# from ext_transforms import get_affine_transform
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image
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

COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_parsing(pred_labels):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    # pred_labels = labels[:num_images].clone().cpu().data
    # if is_pred:
    #     pred_labels = torch.argmax(pred_labels, dim=1)
    n, h, w = pred_labels.size()

    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]

    return labels_color


def gaussian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class SURREAL(data.Dataset):
    def __init__(self, root, split, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=None, void_pixels=0, return_edge=False, num_keypoints=24, sigma=3):
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
        self.num_keypoints = num_keypoints
        self.sigma = sigma

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

        pose_anno_path = os.path.join(self.root, self.dataset , 'pose_png', item)

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)[..., ::-1]
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.compat.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            # parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = np.array(Image.open(parsing_anno_path))

            arr = np.empty(len(val_new), dtype=val_new.dtype)
            arr[val_old] = val_new
            parsing_anno = arr[parsing_anno]
            parsing_anno = np.uint8(parsing_anno)

            pose_anno = np.array(Image.open(pose_anno_path))

            if self.dataset == 'train' or self.dataset == 'trainval':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    
                    pose_anno = pose_anno[:, ::-1]

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

        if self.dataset == 'test':
            return input, meta
        else:
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

            
            label_pose = cv2.warpAffine(
                pose_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                )

            # convert to coordinate
            label_pose = torch.from_numpy(label_pose)

            kpt_coord = torch.zeros(self.num_keypoints,2)
            for i in range(self.num_keypoints):
                try:
                    kpt_coord[i,0] = int(torch.where(label_pose==(i+1))[0].float().mean().item())
                    kpt_coord[i,1] = int(torch.where(label_pose==(i+1))[1].float().mean().item())
                except:
                    print(torch.where(label_pose==(i+1)))
                    kpt_coord[i,0] = -1
                    kpt_coord[i,1] = -1
            heatmap = np.zeros((label_parsing.shape[0], label_parsing.shape[1], self.num_keypoints+1), dtype=np.float32)

            for i in range(len(kpt_coord)):
                # resize from 368 to 46
                x = int(kpt_coord[i][1])
                y = int(kpt_coord[i][0])
                if x < 0 and y < 0:
                    continue
                heat_map = gaussian_kernel(size_h=label_parsing.shape[0],size_w=label_parsing.shape[1], center_x=x, center_y=y, sigma=self.sigma)
                heat_map[heat_map > 1] = 1
                heat_map[heat_map < 0.0099] = 0
                heatmap[:, :, i + 1] = heat_map
            heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)
            heatmap = torch.from_numpy(heatmap).permute(2,0,1)

            
            # if self.return_edge:
            #     return input, (label_parsing, edge_parsing), label_pose

            return input, label_parsing, heatmap, meta

class SURREAL_Test(data.Dataset):
    def __init__(self, root, split='test', crop_size=[473, 473], transform=None, num_keypoints=24, sigma=3):
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.dataset = split
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.num_keypoints = num_keypoints
        self.sigma = sigma

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
        # Load training image

        im_path = os.path.join(self.root, self.dataset , 'image', item)
        parsing_anno_path = os.path.join(self.root, self.dataset , 'segmentation', item)
        pose_anno_path = os.path.join(self.root, self.dataset , 'pose_png', item)

        # im_path = os.path.join(self.root, self.dataset + '_images', val_item + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        # Get person center and scale

        parsing_anno = np.array(Image.open(parsing_anno_path))
        arr = np.empty(len(val_new), dtype=val_new.dtype)
        arr[val_old] = val_new
        parsing_anno = arr[parsing_anno]
        parsing_anno = np.uint8(parsing_anno)

        pose_anno = np.array(Image.open(pose_anno_path))

        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        input = self.transform(input)

        label_parsing = cv2.warpAffine(
            parsing_anno,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            # borderValue=(self.ignore_label)
            )
        label_parsing = torch.from_numpy(label_parsing)

        label_pose = cv2.warpAffine(
            pose_anno,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            )

        # convert to coordinate
        label_pose = torch.from_numpy(label_pose)

        kpt_coord = torch.zeros(self.num_keypoints,2)
        for i in range(self.num_keypoints):
            try:
                kpt_coord[i,0] = int(torch.where(label_pose==(i+1))[0].float().mean().item())
                kpt_coord[i,1] = int(torch.where(label_pose==(i+1))[1].float().mean().item())
            except:
                print(torch.where(label_pose==(i+1)))
                kpt_coord[i,0] = -1
                kpt_coord[i,1] = -1

        heatmap = np.zeros((label_parsing.shape[0], label_parsing.shape[1], self.num_keypoints+1), dtype=np.float32)
        for i in range(len(kpt_coord)):
            # resize from 368 to 46
            x = int(kpt_coord[i][1])
            y = int(kpt_coord[i][0])
            if x < 0 and y < 0:
                continue

            heat_map = gaussian_kernel(size_h=label_parsing.shape[0],size_w=label_parsing.shape[1], center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)
        heatmap = torch.from_numpy(heatmap).permute(2,0,1)

        meta = {
            'name': item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return input, label_parsing, heatmap, meta


if __name__ == '__main__':
    from tqdm import tqdm
    num_classes = 7

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return_edge = False # if 'edge' in opts.model else False
    train_dst = SURREAL(root='/mnt/server14_hard0/msson/datasets/SURREAL/data/ref_cmu_full', split='train', crop_size=[512, 512], scale_factor=0.25,
                                       rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
                                       void_pixels=0, return_edge=return_edge)
    train_loader = data.DataLoader(train_dst, batch_size=1, shuffle=True, num_workers=2)

    test_dst = SURREAL_Test(root='/mnt/server14_hard0/msson/datasets/SURREAL/data/ref_cmu_full', split='test', crop_size=[512, 512], transform=transform)
    test_loader = data.DataLoader(test_dst, batch_size=1, shuffle=False, num_workers=2)

    for (images, labels, poses, _) in tqdm(train_loader):
        a = images
        b = labels.expand_as(images)
        c = poses[:,1:,:,:].sum(axis=1).expand_as(images)

    
        grid = make_grid(torch.cat((a,b,c),dim=0),nrow=3)
        save_image(grid, './ex.png')

        pdb.set_trace()

    for (images, labels, poses, _) in tqdm(test_loader):
        a = images
        # b = labels.expand_as(images)
        c = poses[:,1:,:,:].sum(axis=1).expand_as(images)

        # pdb.set_trace()
        b = decode_parsing(labels)


        grid = make_grid(torch.cat((a,b,c),dim=0),nrow=3)
        save_image(grid, './ex.png')
        pdb.set_trace()

