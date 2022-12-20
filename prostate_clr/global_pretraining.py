from prostate_clr.simclr_trainer import SimCLRTrainer
import argparse


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    # depth of unet encoder
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--start_filters", type=int, default=64)

    # From hu_clr.yaml file
    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/unlabeled-2")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/prostate_clr"
                                                        "/sim_clr")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--loss_temperature", type=float, default=0.5)
    parser.add_argument("--use_cosine_similarity", type=bool, default=True)
    parser.add_argument("--model_embed_dim", type=int, default=512)
    parser.add_argument("--model_out_dim", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_config()
    print(config)

    simclr = SimCLRTrainer(config)
    simclr.train()

