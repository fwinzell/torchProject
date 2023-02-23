import os
import argparse
import torch
from os.path import exists

from trixi.util import Config

from configs.Config import get_config
from datasets.prepare_dataset.create_splits import create_splits
from experiments.SegExperiment import SegExperiment
from datasets.downsanpling_data import downsampling_image

import datetime
import time

import matplotlib
import matplotlib.pyplot as plt

from datasets.prepare_dataset.rearrange_dir import rearrange_dir


def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="prostate")
    parser.add_argument("--train_sample", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--save_model_dir", type=str,
                        default='/home/fi5666wi/Documents/Python/saved_models/hu_clr/prostate_models/')
    parser.add_argument("--freeze_model", action='store_true',
                        help='whether freeze encoder of the segmenter')
    parser.add_argument("--load_saved_model", type=bool, default=True,
                        help="whether load saved model from saved_model_path")

    parser.add_argument("--update_from_argv", type=bool, default=True)

    # Train parameters
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--name", type=str, default='manly_unet')

    parser.add_argument("--device", type=str, default="cuda")  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

    # Logging parameters
    parser.add_argument("--plot_freq", type=int, default=10)  # How often should stuff be shown in visdom
    parser.add_argument("--append_rnd_string", type=bool, default=False)
    parser.add_argument("--start_visdom", type=bool, default=False)

    parser.add_argument("--do_instancenorm", type=bool, default=True) # Defines whether or not the UNet does a instance normalization in the contracting path
    parser.add_argument("--do_load_checkpoint", type=bool, default=False)

    # Adapt to your own path, if needed.
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--base_dir", type=str, default='/home/fi5666wi/Documents/Python/saved_models/hu_clr')  # Where to log the output of the experiment.
    # The path where the downloaded dataset is stored.
    parser.add_argument("--data_root_dir", type=str, default='/home/fi5666wi/Documents/Prostate images')
    parser.add_argument('--data_folder', type=str, default='train_data_with_gt_3_classes')

    args = parser.parse_args()
    args.data_dir = os.path.join(args.data_root_dir, args.data_folder)

    return args


def training(config):
    # config.saved_model_path = os.path.abspath('save') + '/SupCon/Hippocampus_models/' \
    #                     + 'SupCon_Hippocampus_resnet50_lr_0.0001_decay_0.0001_bsz_1_temp_0.7_trial_0_cosine/' \
    #                     + 'last.pth'

    exp = SegExperiment(config=config, name=config.name, n_epochs=config.n_epochs,
                        seed=42, append_rnd_to_name=config.append_rnd_string,
                        base_dir=os.path.abspath('output_experiment'))   # visdomlogger_kwargs={"auto_start": c.start_visdom}

    exp.run()
    # exp.run_test(setup=False)


def testing(config):

    c.do_load_checkpoint = True
    c.checkpoint_dir = c.base_dir + '/20210202-064334_Unet_mmwhs' + '/checkpoint/checkpoint_current'

    exp = SegExperiment(config=config, name='unet_test', n_epochs=config.n_epochs,
                        seed=42, globs=globals())
    exp.run_test(setup=True)


if __name__ == "__main__":
    args = parse_option()
    c = get_config()

    c.num_classes = args.num_classes
    c.dataset_name = args.data_folder

    c.fold = args.fold
    c.batch_size = args.batch_size
    c.train_sample = args.train_sample
    if args.load_saved_model:
        c.saved_model_path = args.save_model_dir \
                             + 'SupCon_prostate_adam_fold_0_lr_0.001_decay_0.0001_bsz_2_temp_0.07_train_0.0_stride_stride_4_pretrained/' \
                             + 'ckpt.pth'

    c.freeze = args.freeze_model
    print(c)
    training(config=c)


