from prostate_clr.local_trainer import LocalCLRTrainer
from prostate_clr.simclr_trainer import SimCLRTrainer
from prostate_clr.unet_trainer import UnetTrainer
from unet.lr_finder import LearningRateFinder
import argparse
import os
from datetime import date


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("-e", "--epochs", type=int, default=25)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])

    # From hu_clr.yaml file
    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/train_data_4_classes")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/prostate_clr"
                                                        "/local_pt")
    parser.add_argument("--pretrained_model_path", type=str, default="/home/fi5666wi/Documents/Python/saved_models"
                                                                     "/prostate_clr/sim_clr/simclr_10_model.pth")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default='adam')

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--use_cosine_similarity", type=bool, default=True)
    parser.add_argument("--model_embed_dim", type=int, default=1024)
    parser.add_argument("--model_out_dim", type=int, default=256)

    # depth of unet encoder
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--start_filters", type=int, default=64)

    # temperature_scaling factor
    parser.add_argument('--temp_fac', type=float, default=0.1)
    # no. of local regions to consider in the feature map for local contrastive loss computation
    parser.add_argument('--no_of_local_regions', type=int, default=13)
    # no. of decoder blocks used. Here, 1 means 1 decoder block used, 2 is for 2 blocks,..., 5 is for all blocks aka
    # full decoder.
    parser.add_argument('--no_of_decoder_blocks', type=int, default=2)
    # local_reg_size - 1 for 3x3 local region size in the feature map. <local_reg> -> flat -> w*flat -> 128 bit z
    # vector matching; - 0 for 1x1 local region size in the feature map
    parser.add_argument('--local_reg_size', type=int, default=1)
    # wgt_en - 1 for having extra weight layer on top of 'z' vector from local region.
    #      - 0 for not having any weight layer.
    parser.add_argument('--wgt_en', type=int, default=1)
    # no. of neighbouring local regions sampled from the feature maps to act as negative samples in local contrastive
    # loss for a given positive local region - currently 5 local regions are chosen from each feature map.
    parser.add_argument('--no_of_neg_local_regions', type=int, default=5)
    # batch_size value for local_loss
    parser.add_argument('--bt_size', type=int, default=12)

    parser.add_argument('--no_of_pt_decoder_blocks', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_config()
    print(config)

    #local_pt = LocalCLRTrainer(config)
    #simclr = SimCLRTrainer(config)
    unet = UnetTrainer(config, load_pt_weights=False)
    lrf = LearningRateFinder(unet.model, unet.gd_loss, unet.optimizer, unet.device)
    lrf.fit(unet.train_loader, steps=500, max_lr=1)
    lrf.plot()