from losses import GeneralizedDiceLoss, MaxDiceLoss
from evaluation import image_to_patches, one_hot_target, colors_to_labels
import os
import torch
import numpy as np
import cv2

# eval_path = os.path.join('/usr/matematik/fi5666wi/Documents/Datasets/Eval')
eval_path = os.path.join('/home/fi5666wi/Documents/Prostate images/WSI-Annotations/image_slic')
image_path = os.path.join(eval_path, 'cropped_image_slic.png')
gt_path = os.path.join(eval_path, 'cropped_image_slic-annotated.png')
segment = cv2.imread(image_path)
gt = cv2.imread(gt_path)

pred_ds = image_to_patches(segment)
gt_ds = image_to_patches(gt)

loss_fn = GeneralizedDiceLoss(labels=np.array([0, 1, 2, 3]))
for idx in range(len(pred_ds)):
    tar = colors_to_labels(gt_ds[idx])
    tar = np.expand_dims(one_hot_target(tar), axis=0)
    tar = torch.tensor(tar).to('cuda')
    zer = torch.zeros_like(tar).to('cuda')
    loss_value = loss_fn(tar, tar)
    print(loss_value)
