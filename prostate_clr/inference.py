from prostate_clr.models import UnetInfer, Unet
from hu_clr.networks.unet_con import SupConUnetInfer

import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse
import cv2
from skimage.io import imread
import math
import datetime
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

from unet.transformations import normalize_01, re_normalize
from unet.losses import GeneralizedDiceLoss
import segmentation_models_pytorch as smp

from augmentation.color_transformation import hsv_transformation

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--input_dim", nargs=3, type=int, default=[256, 256, 3])
parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--start_filters", type=int, default=64)
parser.add_argument("--model_path", type=str,
                    default='/home/fi5666wi/Documents/Python/saved_models/prostate_clr/unet/2022-10-03'
                            '/unet_fold_3_best_4_model.pth')
parser.add_argument("--no_of_pt_decoder_blocks", type=int, default=3)
parser.add_argument("--imagenet", type=bool, default=False)
parser.add_argument("--hu", type=bool, default=False)

# c.saved_model_path = os.path.abspath("output_experiment") + "/20210227-065712_Unet_mmwhs/" \
#                        + "checkpoint/" + "checkpoint_last.pth.tar"

config = parser.parse_args()
if config.num_classes == 3:
    color_dict = {
        'Background': [255, 255, 255],
        'EC + Stroma': [255, 0, 0],
        'Nuclei': [255, 0, 255]
    }
elif config.num_classes == 5:
    color_dict = {
        'Background': [255, 255, 255],
        'Stroma': [255, 255, 0],
        'Cytoplasm': [255, 0, 0],
        'Nuclei': [255, 0, 255],
        'Border': [0, 0, 0]
    }
else:
    color_dict = {
        'Background': [255, 255, 255],
        'Stroma': [255, 255, 0],
        'Cytoplasm': [255, 0, 0],
        'Nuclei': [255, 0, 255]
    }


# preprocess function
def preprocess(img: np.ndarray):
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    return img


# postprocess function
def postprocess(img: torch.tensor):
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    return img


def ind2segment(ind):
    cmap = [color_dict[name] for name in color_dict]
    segment = np.array(cmap, dtype=np.uint8)[ind.flatten()]
    segment = segment.reshape((ind.shape[0], ind.shape[1], 3))
    return segment


def predict(img,
            model,
            preprocess,
            postprocess,
            device,
            ):
    model.eval()
    img = preprocess(img)  # preprocess image
    x = torch.from_numpy(img).to(device)  # to torch, send to device
    with torch.no_grad():
        if config.hu:
            z, y = model(x)
        else:
            y = model(x)  # send through model/network

    out_softmax = torch.softmax(y, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs
    return result


def get_tiles(image, tile_sz=224, patch_sz=256):
    pad = int((patch_sz - tile_sz) / 2)
    # torch_img = torch.tensor((np.moveaxis(img, source=-1, destination=0)))
    padded = np.pad(image, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    (ydim, xdim) = padded.shape[:2]
    nbr_xpatches = math.ceil(image.shape[1] / tile_sz)
    nbr_ypatches = math.ceil(image.shape[0] / tile_sz)

    tiles = np.empty(shape=(nbr_xpatches * nbr_ypatches, patch_sz, patch_sz, 3))
    count = 0
    coords = []
    ycor = 0
    for i in range(nbr_ypatches):
        xcor = 0
        for j in range(nbr_xpatches):
            patch = padded[ycor:ycor + patch_sz, xcor:xcor + patch_sz, :]
            tiles[count] = patch
            coords.append([ycor + pad, xcor + pad])
            count += 1
            xcor = xcor + tile_sz
            if xcor > xdim - patch_sz:
                xcor = xdim - patch_sz
        ycor = ycor + tile_sz
        if ycor > ydim - patch_sz:
            ycor = ydim - patch_sz

    return tiles


def tiling(tiles, image_shape, tile_sz=224, patch_sz=256):
    image = np.zeros(shape=image_shape)
    pad = int((patch_sz - tile_sz) / 2)
    nbr_xpatches = math.ceil(image_shape[1] / tile_sz)
    nbr_ypatches = math.ceil(image_shape[0] / tile_sz)
    ycor = 0
    count = 0
    for i in range(nbr_ypatches):
        xcor = 0
        for j in range(nbr_xpatches):
            t = tiles[count, pad:pad + tile_sz, pad:pad + tile_sz]
            image[ycor:ycor + tile_sz, xcor:xcor + tile_sz] = tiles[count, pad:pad + tile_sz, pad:pad + tile_sz]

            """cv2.imshow('tile', ind2segment(t.astype(np.uint8)))
            newim = ind2segment(image.astype(np.uint8))
            cv2.rectangle(newim, (xcor, ycor), (xcor + tile_sz, ycor + tile_sz),
                          color=(0, 0, 0), thickness=2)
            newim = cv2.resize(newim, dsize=None, fx=0.25, fy=0.25)
            cv2.imshow('image', newim)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""

            count += 1
            xcor = xcor + tile_sz
            if xcor > image_shape[1] - tile_sz:
                xcor = image_shape[1] - tile_sz
        ycor = ycor + tile_sz
        if ycor > image_shape[0] - tile_sz:
            ycor = image_shape[0] - tile_sz

    return image


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    # model = FineTunedUnet(UnetSimCLR(config), config).to(device)
    # model = UnetInfer(config).to(device)
    if config.hu:
        model = SupConUnetInfer(num_classes=config.num_classes, in_channels=3).to(device)
        state_dict = torch.load(config.model_path)['model']
        model.load_state_dict(state_dict, strict=False)
    else:
        if config.imagenet:
            model = smp.Unet(encoder_name='resnet50', classes=config.num_classes).to(device)
        else:
            model = Unet(config, load_pt_weights=False).to(device)
        model_weights = torch.load(config.model_path)
        model.load_state_dict(model_weights, strict=True)

    # Image to segment
    image_path = '/home/fi5666wi/Documents/Prostate images/WSI-Annotations/image_cancer_3/cropped_image19PM18049-10_10x.png'  # '/usr/matematik/fi5666wi/Documents/Datasets/Eval'

    prostate_image = imread(image_path)
    hsv_image = hsv_transformation(prostate_image)
    img_ds = get_tiles(hsv_image)

    output = []
    for idx in range(len(img_ds)):
        img = img_ds[idx]
        pred = predict(img, model, preprocess, postprocess, device)
        output.append(pred)

    res = tiling(np.array(output), prostate_image.shape[:-1])
    res = ind2segment(res.astype(np.uint8))

    s_f = 1000/prostate_image.shape[0]
    disp = cv2.resize(res, dsize=None, fx=s_f, fy=s_f)
    orig = cv2.resize(prostate_image, dsize=None, fx=s_f, fy=s_f)
    hsv = cv2.resize(hsv_image, dsize=None, fx=s_f, fy=s_f)
    cv2.imshow('Final result', disp)
    cv2.imshow('Original', cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))
    cv2.imshow('HSV', cv2.cvtColor(hsv, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('End')
    # savepath = os.path.join(eval_path, 'segmented_image_' + str(datetime.date.today()) + '.png')
    # cv2.imwrite(savepath, res)
