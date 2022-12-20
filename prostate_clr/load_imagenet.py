import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet50
import segmentation_models_pytorch as smp
from prettytable import PrettyTable
from skimage.io import imread
from hu_clr.datasets.transformations import *
from prostate_clr.models import Unet
import cv2


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


color_dict = {
        'Background': [255, 255, 255],
        'EC + Stroma': [255, 0, 0],
        'Nuclei': [255, 0, 255]
    }
config = Namespace(num_classes=3,
          depth=4,
          no_of_pt_decoder_blocks=0,
          start_filters=64,
          input_shape=[3, 256, 256])

def preprocess(img):
    transform = ComposeSingle([FunctionWrapperSingle(normalize_01),
                               FunctionWrapperSingle(np.moveaxis, source=-1, destination=0)
                               ])
    img = transform(img)
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    return torch.from_numpy(img).type(torch.float32)


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


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# Step 1: Initialize model with the best available weights
# weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(pretrained=True)
encoder = model.backbone
count_parameters(encoder)

image = imread("/home/fi5666wi/Documents/Prostate images/train_data_3_classes/gt_img_1/Patches/patch_60.png")
image = preprocess(image)

z = encoder(image)

x_0 = encoder.conv1(image)
x_0 = encoder.bn1(x_0)
x_1 = encoder.layer1(x_0)
x_2 = encoder.layer2(x_1)
x_3 = encoder.layer3(x_2)
x_4 = encoder.layer4(x_3)


# create segmentation model with pretrained encoder
unet = smp.Unet(
    encoder_name='resnet50',
    encoder_depth=4,
    encoder_weights='imagenet',
    classes=3,
    activation='softmax2d',
)
count_parameters(unet)
y = unet(image)
print(y.shape)

out = postprocess(y)
out = ind2segment(out)
cv2.imshow('Output', out)
cv2.waitKey(1000)

print(config.input_shape)
unet_model = Unet(config)
unet_model.load_state_dict(unet.state_dict(), strict=True)

