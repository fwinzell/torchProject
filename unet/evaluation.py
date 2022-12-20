import torch
from unet_model import UNet
import os
import numpy as np
import argparse
import cv2
import math
import datetime
from transformations import normalize_01, re_normalize
from losses import GeneralizedDiceLoss, MaxDiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--input_dim", nargs=3, type=int, default=[256, 256, 3])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--filters", type=int, default=64)
parser.add_argument("--model_name",
                    default='/home/fi5666wi/Documents/Python/saved_models/unet_model_2022-06-23')

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
        out = model(x)  # send through model/network

    out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
    result = postprocess(out_softmax)  # postprocess outputs
    return result, out


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


def test_loss_function(pred, gt, loss_fn=GeneralizedDiceLoss(labels=np.array(range(config.num_classes)))):
    loss_val = loss_fn(torch.tensor(pred).cuda(), torch.tensor(gt).cuda())
    return loss_val


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    model = UNet(config).to(device)
    model_weights = torch.load(os.path.join(config.model_name, 'model.pt'))
    model.load_state_dict(model_weights)

    # Image to segment
    eval_path = os.path.join('/usr/matematik/fi5666wi/Documents/Datasets/Eval')
    image_path = os.path.join(eval_path, 'cropped_image_slic.png')
    if config.num_classes == 4:
        gt_path = os.path.join(eval_path, 'cropped_image_slic-annotated.png')
    else:
        gt_path = os.path.join(eval_path, 'cropped_image_slic-annotated_3c.png')
    prostate_image = cv2.imread(image_path)
    gt_image = cv2.imread(gt_path)

    img_ds = image_to_patches(prostate_image)
    gt_ds = image_to_patches(gt_image)

    # Compute prediction
    # predict the segmentation maps
    output = []
    dsc = np.zeros(config.num_classes)
    loss_values = []
    for idx in range(len(img_ds)):
        img = img_ds[idx]
        target = colors_to_labels(gt_ds[idx])
        hot = np.expand_dims(one_hot_target(target), axis=0)
        pred, model_output = predict(img, model, preprocess, postprocess, device)
        scores = dice(pred, target)
        dsc = (dsc + scores) / 2
        # For displaying
        output.append(ind2segment(pred))
        loss_values.append(test_loss_function(model_output, hot).cpu()) # loss_fn=MaxDiceLoss(labels=np.array([0, 1, 2, 3]))).cpu())
        #loss_values.append(test_loss_function(model_output, target, loss_fn=torch.nn.CrossEntropyLoss()).cpu())

    print('Dice scores: ')
    for i, name in enumerate(color_dict):
        print(name + ": %.3f" % dsc[i])
    print('Mean DSC: ' + str(np.mean(dsc)))

    res = patches_to_image(np.array(output), prostate_image.shape)
    cv2.imshow('Final result', res)
    cv2.imshow('Target', gt_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Mean loss value:')
    print(np.mean(np.array(loss_values)))

    print('End')
    # savepath = os.path.join(eval_path, 'segmented_image_' + str(datetime.date.today()) + '.png')
    # cv2.imwrite(savepath, res)
