from prostate_clr.local_trainer import LocalCLRTrainer
import argparse


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("-e", "--epoch", type=int, default=50)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])

    # From hu_clr.yaml file
    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Documents/Prostate images/unlabeled")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/saved_models/prostate_clr"
                                                        "/sim_clr")
    parser.add_argument("--pretrained_model_path", type=str, default="/home/fi5666wi/Documents/Python/saved_models"
                                                                     "/prostate_clr/sim_clr/sim.pth")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--eval_every_n_epochs", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    # depth of unet encoder
    parser.add_argument("--depth", type=int, default=4)

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

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_config()
    print(config)

    localclr = LocalCLRTrainer(config)
    localclr.train()