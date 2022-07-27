#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch

from torch.utils import data
from tqdm import tqdm, tqdm_gui
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
# from datasets.datasets import LIPDataValSet
# from datasets.pascal_loader_orig import PascalPartsPerson
from datasets.datasets_PPP import PascalPartSegmentation
from datasets.datasets_SURREAL_v2 import SURREAL
from utils.miou_SURREAL import compute_mean_ioU
# from utils.miou_PPP import compute_mean_ioU
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing

from networks.AugmentCE2P_t2t_vit import t2tnet


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
    parser.add_argument("--data-dir", type=str, default='/mnt/server14_hard0/msson/datasets/SURREAL/data/for_gifs')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default='./log_SURREAL_resnet18/checkpoint_100.pth.tar')
    # parser.add_argument("--model-restore", type=str, default='./log_SURREAL/schp_6_checkpoint.pth.tar')
    # parser.add_argument("--model_restore", type=str, default="./log/exp-schp-201908270938-pascal-person-part.pth")
    parser.add_argument("--gpu", type=str, default='2', help="choose gpu device.")
    parser.add_argument("--save-results", action="store_true", default=True, help="whether to save the results.")
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    return parser.parse_args()

label_to_color = {
    0: [128, 64,128],
    1: [244, 35,232],
    2: [ 70, 70, 70],
    3: [102,102,156],
    4: [190,153,153],
    5: [153,153,153],
    6: [250,170, 30],
    7: [220,220,  0],
    8: [107,142, 35],
    9: [152,251,152],
    10: [ 70,130,180]
    }

epsilon = 1e-4
NUM_CLASSES = 7

'''
    전체 dataset에 대해 i와 u 합산하여 mIoU 계산
'''
# def cal_miou_total(result, gt):                ## resutl.shpae == gt.shape == [batch_size, 512, 512]    
#     tensor1 = torch.Tensor([1]).to(gt.device)
#     tensor0 = torch.Tensor([0]).to(gt.device)

#     res_i = torch.zeros((NUM_CLASSES)).to(result.device)
#     res_u = torch.zeros((NUM_CLASSES)).to(result.device)

#     for idx in range(NUM_CLASSES):
#         u = torch.sum(torch.where((result==idx) + (gt==idx), tensor1, tensor0)).item()
#         i = torch.sum(torch.where((result==idx) * (gt==idx), tensor1, tensor0)).item()

#         res_i[idx] += i
#         res_u[idx] += u

#     return res_i, res_u


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output[0]
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    multi_scales = [float(i) for i in args.multi_scales.split(',')]
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled = True

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)
    # model = t2tnet(num_classes=args.num_classes, pretrained='pretrain_model/81.7_T2T_ViTt_14.pth.tar')
    

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),

        ])
    if INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Data loader
    # lip_test_dataset = LIPDataValSet(args.data_dir, 'val', crop_size=input_size, transform=transform, flip=args.flip)
    # ppp_test_dataset = PascalPartsPerson(root=args.data_dir, split='val', resize_size=input_size)
    # ppp_test_dataset = PascalPartSegmentation(root=args.data_dir, split='val', crop_size=input_size, # [512,512]
    #                                    scale_factor=0.25,
    #                                    rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
    #                                    void_pixels=3, return_edge=False)
    ppp_test_dataset = SURREAL(root=args.data_dir, split='test', crop_size=input_size, # [512,512]
                                       scale_factor=0.25,
                                       rotation_factor=30, ignore_label=255, flip_prob=0.5, transform=transform,
                                       void_pixels=3, return_edge=False)
    num_samples = len(ppp_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(ppp_test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load model weight
    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    sp_results_dir = os.path.join(args.log_dir, 'sp_results')
    if not os.path.exists(sp_results_dir):
        os.makedirs(sp_results_dir)

    palette = get_palette(args.num_classes)
    parsing_preds = []
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)
    with torch.no_grad():
        # avg_miou = 0
        # total_i = torch.zeros((NUM_CLASSES)).cuda()
        # total_u = torch.zeros((NUM_CLASSES)).cuda()
        for idx, batch in enumerate(tqdm(testloader)):
            # image, seg = batch
            # if torch.cuda.is_available():
            #     image = image.cuda()
            #     seg = seg.cuda()

            image, meta, seg = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            im_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            scales[idx, :] = s
            centers[idx, :] = c
            # input_size = seg.shape[-2:]
            parsing, logits = multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=args.flip,
                                                  multi_scales=multi_scales)

            # res_i, res_u = cal_miou_total(torch.tensor(parsing).to(seg.device), seg)
            # total_i += res_i
            # total_u += res_u

            if args.save_results:         
                '''
                    Save frames as gif.
                '''
                from torchvision.utils import save_image, make_grid
                save_pth = os.path.join('result_for_gifs')
                os.makedirs(save_pth, exist_ok=True)

                real_frames = []
                gt_frames = []
                pred_frames = []

                invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                                    std = [ 1., 1., 1. ]),])
                inv_imgs = invTrans(image.cpu().squeeze())

                parsing = torch.tensor(parsing)
                parsing = torch.stack([parsing, parsing, parsing], dim=-1).type(torch.uint8)
                seg_color = np.zeros(parsing.shape)
                for key in label_to_color.keys():
                    seg_color[parsing[:,:,0] == key] = label_to_color[key]
                seg_color = transforms.ToTensor()(seg_color.astype(np.uint8))

                seg = seg.squeeze().cpu()
                seg = np.stack([seg, seg, seg], axis=-1)
                gt_color = np.zeros(seg.shape)
                for key in label_to_color.keys():
                    gt_color[seg[:,:,0] == key] = label_to_color[key]
                gt_color = transforms.ToTensor()(gt_color.astype(np.uint8))

                file_name = os.path.splitext(im_name)[0]
                path = os.path.join(save_pth, file_name+'_{}.png'.format('real'))
                save_image(inv_imgs, path)
                path = os.path.join(save_pth, file_name+'_{}.png'.format('gt'))
                save_image(gt_color, path)
                path = os.path.join(save_pth, file_name+'_{}.png'.format('pred'))
                save_image(seg_color, path)

                # real_frames.append(inv_imgs)
                # gt_frames.append(gt_color)
                # pred_frames.append(seg_color)

            parsing_preds.append(parsing)
        # make_gif('real', real_frames, save_pth)
        # make_gif('gt', gt_frames, save_pth)
        # make_gif('pred', pred_frames, save_pth)

    assert len(parsing_preds) == num_samples
    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    print(mIoU)
    return

def make_gif(type, frames, save_pth):
    save_name = os.path.join(save_pth, "{}.gif".format(type))
    frame_one = frames[0]
    frame_one.save(save_name, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    main()
