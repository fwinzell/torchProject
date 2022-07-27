import torch
import os
import numpy as np
import argparse
import cv2
import math
from model import FineTunedUnet, UnetSimCLR, Encoder, Decoder
from unet.transformations import normalize_01, re_normalize

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--input_dim", nargs=3, type=int, default=[256, 256, 3])
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--decoding_steps", type=int, default=3)
parser.add_argument("--filters", type=int, default=64)
parser.add_argument("--model_name",
                    default='/home/fi5666wi/Documents/Python/saved_models/simclr/simclr_model_2022-06-27')

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
    #img = re_normalize(img)  # scale it to the range [0-255]
    img = ind2segment(img)
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
    return result


def click_and_drag(event, x, y, flags, param):
    global refPt, drawing, img_small, cache

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        drawing = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        # draw a rectangle around the region of interest
        cv2.rectangle(img_small, refPt[0], refPt[1], (0, 255, 0), 2)
        drawing = False
        cv2.imshow('Selection', img_small)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        clone = img_small.copy()
        cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Selection', clone)

    elif event == cv2.EVENT_RBUTTONDOWN:
        img_small = cache.copy()
        refPt = []
        cv2.imshow('Selection', img_small)


def select_image_to_segment(img):
    scale_percent = 10  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    global img_small
    img_small = cv2.resize(img, dim)
    global cache
    cache = img_small.copy()
    global refPt
    refPt = []
    global drawing
    drawing = False

    cv2.namedWindow('Selection')
    cv2.setMouseCallback('Selection', click_and_drag)
    cv2.imshow('Selection', img_small)
    cv2.waitKey(0)

    if refPt:
        coords = np.int64(np.array(refPt) * (100 / scale_percent))
        cropped = img[coords[0, 1]:coords[1, 1], coords[0, 0]:coords[1, 0]]
        crop_dim = cropped.shape
        cv2.imshow('Selection ' + str(crop_dim[0]) + 'x' + str(crop_dim[1]), cropped)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped


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
    step = sz-overlap
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


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    # model = FineTunedUnet(UnetSimCLR(config), config).to(device)
    model = Decoder(Encoder(config), config).to(device)
    model_weights = torch.load(os.path.join(config.model_name, 'model.pt'))
    model.load_state_dict(model_weights)

    # Image to segment
    #eval_path = os.path.join('/usr/matematik/fi5666wi/Documents/Datasets/Eval')
    eval_path = os.path.join('/home/fi5666wi/Documents/Prostate images/2019-01-24')
    image_path = os.path.join(eval_path, '11PM 30667-1_10x.png')
    prostate_image = cv2.imread(image_path)

    image_to_segment = select_image_to_segment(prostate_image)
    patches = image_to_patches(image_to_segment)

    # Compute prediction
    # predict the segmentation maps
    output = [predict(img, model, preprocess, postprocess, device) for img in patches]
    """
    for j, out in enumerate(output):
        cv2.imshow('Result', out)
        cv2.imshow('Orginial', patches[j])
        cv2.waitKey(1000)
    """
    #cv2.destroyAllWindows()
    res = patches_to_image(np.array(output), image_to_segment.shape)
    cv2.imshow('Final result', res)
    cv2.imshow('Original', image_to_segment)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


