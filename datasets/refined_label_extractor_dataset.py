# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-

# """
# @Author  :   Peike Li
# @Contact :   peike.li@yahoo.com
# @File    :   dataset.py
# @Time    :   8/30/19 9:12 PM
# @Desc    :   Dataset Definition
# @License :   This source code is licensed under the license found in the
#              LICENSE file in the root directory of this source tree.
# """

# import os
# import cv2
# import numpy as np

# from torch.utils import data
# from utils.transforms import get_affine_transform


# class SimpleFolderDataset(data.Dataset):
#     def __init__(self, root, input_size=[512, 512], transform=None):
#         self.root = root
#         self.input_size = input_size
#         self.transform = transform
#         self.aspect_ratio = input_size[1] * 1.0 / input_size[0]
#         self.input_size = np.asarray(input_size)

#         self.file_list = os.listdir(self.root)

#     def __len__(self):
#         return len(self.file_list)

#     def _box2cs(self, box):
#         x, y, w, h = box[:4]
#         return self._xywh2cs(x, y, w, h)

#     def _xywh2cs(self, x, y, w, h):
#         center = np.zeros((2), dtype=np.float32)
#         center[0] = x + w * 0.5
#         center[1] = y + h * 0.5
#         if w > self.aspect_ratio * h:
#             h = w * 1.0 / self.aspect_ratio
#         elif w < self.aspect_ratio * h:
#             w = h * self.aspect_ratio
#         scale = np.array([w, h], dtype=np.float32)
#         return center, scale

#     def __getitem__(self, index):
#         img_name = self.file_list[index]
#         img_path = os.path.join(self.root, img_name)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)

#         input = img
#         input = self.transform(input)

#         return input, img_name

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
        # seg_path = os.path.join(self.root, self.split , 'segmentation_ref', item)

        image = self.image_resize(Image.open(im_path))
        # image = Image.open(im_path)
        # seg = Image.open(seg_path)

        if self.split == 'train':
            image = self.augmentation(image)
        elif self.split == 'val':
            image = self.totensor(image)
    
        image = self.rgb_normalize(image)

        return image, item
    
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