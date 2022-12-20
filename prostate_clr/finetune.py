import os
from datetime import date
from prostate_clr.unet_trainer import UnetTrainer, CrossValTrainer
import argparse


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--num_folds", type=int, default=6)  # For cross-validation
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/train_data_4_classes")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/prostate_clr"
                                                        "/unet")
    parser.add_argument("--pretrained_model_path", type=str, default="/home/fi5666wi/Documents/Python/saved_models"
                                                                     "/prostate_clr/sim_clr"
                                                                     "/simclr_20_model.pth")
    parser.add_argument("--no_of_pt_decoder_blocks", type=int, default=3)
    parser.add_argument("--log_every_n_steps", type=int, default=10)

    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--img_size", type=int, default=64)

    # depth of unet encoder
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--start_filters", type=int, default=64)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # Preprocessing
    # 'norm' for normalization
    # 'aug' for augmentation
    parser.add_argument("--preprocess", type=str, default='aug')
    parser.add_argument("--hsv_factors", nargs=3, type=float, default=[3.5, 2.2, 1.2])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_config()
    save_path = os.path.join(config.save_dir, str(date.today()))
    num = 1
    while os.path.isdir(save_path):
        save_path = os.path.join(config.save_dir, "{}_{}".format(date.today(), num))
        num += 1
    os.mkdir(save_path)
    config.save_dir = save_path
    print(config)
    config.preprocess = 'augmap'

    #unet = UnetTrainer(config, load_pt_weights=False, use_imagenet=False, use_gdl=True)
    #unet.train()

    crossv = CrossValTrainer(config, load_pt_weights=False, use_imagenet=False, use_gdl=True)
    crossv.train()
