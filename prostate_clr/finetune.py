from prostate_clr.unet_trainer import UnetTrainer
import argparse


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/train_data_with_gt")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/prostate_clr"
                                                        "/unet")
    parser.add_argument("--pretrained_model_path", type=str, default="/home/fi5666wi/Documents/Python/saved_models"
                                                                     "/prostate_clr/sim_clr/sim.pth")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--img_size", type=int, default=64)

    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = parse_config()
    unet = UnetTrainer(config)
    unet.train()
