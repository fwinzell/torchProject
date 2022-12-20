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

from unet.transformations import normalize_01
from unet.losses import GeneralizedDiceLoss
import segmentation_models_pytorch as smp

from augmentation.color_transformation import hsv_transformation

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--start_filters", type=int, default=64)
parser.add_argument("--model_path", type=str,
                    default='/home/fi5666wi/Documents/Python/saved_models/prostate_clr/unet/2022-12-05_1'
                            '/unet_best_4_model.pth')
parser.add_argument("--no_of_pt_decoder_blocks", type=int, default=3)
parser.add_argument("--imagenet", type=bool, default=False)
parser.add_argument("--hu", type=bool, default=False)
parser.add_argument("--use_hsv_norm", type=bool, default=False)
parser.add_argument("--thresh", type=int, default=240)
parser.add_argument("--use_grayscale", type=bool, default=False)

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
if config.use_grayscale:
    config.input_shape = [1, 256, 256]


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


def image_to_patches(image, size=256, overlap=10):
    (ydim, xdim) = image.shape[:2]
    step = size - overlap
    nbr_xpatches = math.ceil(xdim / step)
    nbr_ypatches = math.ceil(ydim / step)

    patch_ds = np.empty(shape=(nbr_xpatches * nbr_ypatches, size, size, 3))
    count = 0
    coords = []
    ycor = 0
    for i in range(nbr_ypatches):
        xcor = 0
        for j in range(nbr_xpatches):
            patch = image[ycor:ycor + size, xcor:xcor + size, :]
            patch_ds[count] = patch
            coords.append([ycor, xcor])
            count += 1
            xcor = xcor + step
            if xcor > xdim - size:
                xcor = xdim - size
        ycor = ycor + step
        if ycor > ydim - size:
            ycor = ydim - size

    return patch_ds


def patches_to_image(ds, image_size, overlap=10):
    image = np.zeros(shape=image_size)
    sz = ds.shape[1]
    step = sz - overlap
    nbr_xpatches = math.ceil(image_size[1] / step)
    nbr_ypatches = math.ceil(image_size[0] / step)
    ycor = 0
    count = 0
    for i in range(nbr_ypatches):
        xcor = 0
        for j in range(nbr_xpatches):
            image[ycor:ycor + sz, xcor:xcor + sz, :] = ds[count]
            count += 1
            xcor = xcor + step
            if xcor > image_size[1] - sz:
                xcor = image_size[1] - sz
        ycor = ycor + step
        if ycor > image_size[0] - sz:
            ycor = image_size[0] - sz

    return image


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
            image[ycor:ycor + tile_sz, xcor:xcor + tile_sz] = tiles[count, pad:pad + tile_sz, pad:pad + tile_sz]
            count += 1
            xcor = xcor + tile_sz
            if xcor > image_shape[1] - tile_sz:
                xcor = image_shape[1] - tile_sz
        ycor = ycor + tile_sz
        if ycor > image_shape[0] - tile_sz:
            ycor = image_shape[0] - tile_sz

    return image


def colors_to_labels(img):
    label_mask = np.zeros(shape=img.shape[:2], dtype='float32')
    for j, label_name in enumerate(color_dict):
        index = np.where(np.all(color_dict[label_name] == img, axis=2))
        label_mask[index] = float(j)
    return label_mask


def one_hot_target(tar):
    classes = np.unique(tar)
    hot_target = np.zeros(shape=(config.num_classes, tar.shape[0], tar.shape[1]))
    for k in classes:
        bw = np.where(tar == k, 1, 0)
        hot_target[int(k), :, :] = bw
    return hot_target


def dice(pred, gt, get_mean=False):
    labels = np.array(range(config.num_classes))
    scores = np.zeros(len(labels))
    for i, l in enumerate(labels):
        y_pred = np.zeros_like(pred)
        y_pred[pred == l] = 1
        y_true = np.zeros_like(gt)
        y_true[gt == l] = 1
        scores[i] = single_dice_coef(y_true, y_pred)

    if get_mean:
        return np.mean(scores)
    else:
        return scores


def single_dice_coef(y_true, y_pred_bin):
    intersection = np.sum(y_true * y_pred_bin)
    if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
        return 1
    return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))


def iou(pred, gt):
    labels = np.array(range(config.num_classes))
    scores = np.zeros(len(labels))
    for i, l in enumerate(labels):
        y_pred = np.zeros_like(pred)
        y_pred[pred == l] = 1
        y_true = np.zeros_like(gt)
        y_true[gt == l] = 1
        scores[i] = binary_iou(y_true, y_pred)
    return scores


def binary_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(np.ceil((y_true+y_pred)/2))
    return intersection/union


def test_loss_function(pred, gt, loss_fn=GeneralizedDiceLoss(labels=np.array(range(config.num_classes)))):
    loss_val = loss_fn(torch.tensor(pred).cuda(), torch.tensor(gt).cuda())
    return loss_val

def grayscale(im):
    from skimage import color
    gray = color.rgb2gray(im)
    gray = np.expand_dims(gray, axis=-1)
    return gray

def evaluation(config, display):
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
    eval_path = '/home/fi5666wi/Documents/Prostate images/Eval'  # '/usr/matematik/fi5666wi/Documents/Datasets/Eval'
    """image_paths = [os.path.join(eval_path, 'image_slic/cropped_image_slic.png'),
                   os.path.join(eval_path, 'image_5/cropped_image_5.png'),
                   os.path.join(eval_path, 'image_4/cropped_image10PM 30316-7_10x.png')]
    if config.num_classes == 4:
        gt_paths = [os.path.join(eval_path, 'image_slic/cropped_image_slic-annotated.png'),
                    os.path.join(eval_path, 'image_5/cropped_image_5-annotated.png'),
                    os.path.join(eval_path, 'image_4/cropped_image_4-annotated.png')]
    else:
        gt_paths = [os.path.join(eval_path, 'image_slic/cropped_image_slic-annotated_3c.png'),
                    os.path.join(eval_path, 'image_5/cropped_image_5-annotated_3c.png'),
                    os.path.join(eval_path, 'image_4/cropped_image_4-annotated_3c.png')]"""
    image_paths = [os.path.join(eval_path, 'image_cancer_1/cancer_1.png')]
    gt_paths = [os.path.join(eval_path, 'image_cancer_1/cancer_1_annotation.png')]

    result = []
    targets = []
    dice_scores = np.zeros((len(image_paths), config.num_classes))
    iou_scores = np.zeros((len(image_paths), config.num_classes))
    conf = np.zeros((config.num_classes, config.num_classes))
    for i in range(len(image_paths)):
        prostate_image = imread(image_paths[i])
        prostate_image = prostate_image[:,:,:-1]
        hsv_image = hsv_transformation(prostate_image, thresh_val=config.thresh)
        gt_image = cv2.imread(gt_paths[i])

        if config.use_hsv_norm:
            img_ds = get_tiles(hsv_image)
        else:
            img_ds = get_tiles(prostate_image)
        gt_ds = get_tiles(gt_image)

        # Compute prediction
        # predict the segmentation maps
        output = []
        loss_values = []
        # For each patch: make predicition
        for idx in range(len(img_ds)):
            img = img_ds[idx]
            target = colors_to_labels(gt_ds[idx])
            hot = np.expand_dims(one_hot_target(target), axis=0)
            if config.use_grayscale:
                gray = grayscale(img)
                pred = predict(gray, model, preprocess, postprocess, device)
                #cv2.imshow('Gray', np.uint8(gray))
                #cv2.imshow('pred', ind2segment(pred.astype(np.uint8)))
                #cv2.waitKey(0)
            else:
                pred = predict(img, model, preprocess, postprocess, device)
            output.append(pred)
        # loss_values.append(test_loss_function(pred, hot).cpu()) # loss_fn=MaxDiceLoss(labels=np.array([0, 1, 2, 3]))).cpu())
        # loss_values.append(test_loss_function(model_output, target, loss_fn=torch.nn.CrossEntropyLoss()).cpu())

        # cv2.imshow('Patch ' + str(idx), seg)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        label_target = colors_to_labels(gt_image)
        res = tiling(np.array(output), prostate_image.shape[:-1])
        dsc = dice(res, label_target)
        ious = iou(res, label_target)
        res = ind2segment(res.astype(np.uint8))
        print('Scores (Dice) (IoU) {}: '.format(i+1))
        for j, name in enumerate(color_dict):
            print(name + ": %.3f  %.3f" % (dsc[j], ious[j]))
        print('Mean DSC: ' + str(np.mean(dsc)))
        print('Mean IoU: ' + str(np.mean(ious)))
        result.append(res)
        targets.append(gt_image)
        dice_scores[i, :] = dsc
        iou_scores[i, :] = ious

        # Confusion matrix
        label_res = colors_to_labels(res)
        conf = conf + confusion_matrix(label_target.flatten(), label_res.flatten())

    if display:
        for k in range(len(result)):
            res = cv2.resize(result[k], dsize=None, fx=0.5, fy=0.5)
            tar = cv2.resize(targets[k], dsize=None, fx=0.5, fy=0.5)
            cv2.imshow('Final result {}'.format(k+1), res)
            cv2.imshow('Target {}'.format(k+1), tar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print('Mean Dice Scores: ')
    for j, name in enumerate(color_dict):
        print(name + ": %.3f" % np.mean(dice_scores[:, j]))
    print('Total Non-bg Mean DSC: ' + str(np.mean(dice_scores[:, 1:])))

    print('Mean IoU Scores: ')
    for j, name in enumerate(color_dict):
        print(name + ": %.3f" % np.mean(iou_scores[:, j]))
    print('Total Non-bg Mean IoU: ' + str(np.mean(iou_scores[:, 1:])))
    # print('Mean loss value:')
    # print(np.mean(np.array(loss_values)))

    print('Confusion matrix')
    conf_sums = np.sum(conf, axis=1)
    conf = conf/conf_sums[:, np.newaxis]
    conf = np.around(conf, decimals=3)
    conf_table = PrettyTable()
    conf_table.add_column(" ", [name for name in color_dict])
    for i, name in enumerate(color_dict):
        conf_table.add_column(name, conf[:, i])
    print(conf_table)

    return np.mean(dice_scores, axis=0), np.mean(iou_scores, axis=0)
    # savepath = os.path.join(eval_path, 'segmented_image_' + str(datetime.date.today()) + '.png')
    # cv2.imwrite(savepath, res)


if __name__ == '__main__':
    do_crossval = True
    if not do_crossval:
        scores = evaluation(config, display=True)
    else:
        num_folds = 6
        model_dir = '/home/fi5666wi/Documents/Python/saved_models/prostate_clr/unet/2022-10-06'
        dice_mat = np.zeros((4, num_folds))
        iou_mat = np.zeros((4, num_folds))
        mdscs = np.zeros(num_folds)
        mious = np.zeros(num_folds)
        for i in range(6):
            model_name = 'unet_fold_{}_best_4_model.pth'.format(i)
            config.model_path = os.path.join(model_dir, model_name)
            print('Loading model: {}'.format(model_name))
            config.use_hsv_norm = True
            dice_mat[:, i], iou_mat[:, i] = evaluation(config, display=True)
            mdscs[i] = np.mean(dice_mat[1:, i])
            mious[i] = np.mean(iou_mat[1:, i])

        for j, name in enumerate(color_dict):
            print('---- ' + name + ' ----')
            print("Mean: %.3f  %.3f" % (np.mean(dice_mat[j, :]), np.mean(iou_mat[j, :])))
            print("St.Dev.: %.3f  %.3f" % (np.std(dice_mat[j, :]), np.std(iou_mat[j, :])))
        print('+----+----+----+ MDSC +----+----+----+')
        print("Mean: %.5f" % np.mean(mdscs))
        print("St.Dev.: %.5f" % np.std(mdscs))
        print('+----+----+----+ MIOU +----+----+----+')
        print("Mean: %.5f" % np.mean(mious))
        print("St.Dev.: %.5f" % np.std(mious))


    print("Babbaba")





