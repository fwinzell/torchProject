from losses import GeneralizedDiceLoss, MaxDiceLoss
from evaluation import image_to_patches
import os
import torch

eval_path = os.path.join('/usr/matematik/fi5666wi/Documents/Datasets/Eval')
image_path = os.path.join(eval_path, 'segmented_image_2022-05-12.png')
gt_path = os.path.join(eval_path, 'cropped_image_slic-annotated.png')
segment = cv2.imread(image_path)
gt = cv2.imread(gt_path)

pred_ds = image_to_patches(segment)
gt_ds = image_to_patches(gt)

loss_fn = MaxDiceLoss(labels=np.array([0,1,2,3]))
for idx in range(len(pred_ds)):
    loss_value = loss_fn(pred_ds[idx],gt_ds[idx])



