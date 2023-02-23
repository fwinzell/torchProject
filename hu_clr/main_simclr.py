from hu_clr.experiments.simclr_prostate import SimCLR
import yaml
import argparse


def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("-e", "--epoch", type=int, default=50)
    parser.add_argument("-f", "--fold", type=int, default=1)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_option()
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epoch
    config['input_shape'] = args.input_shape
    print(config)

    simclr = SimCLR(config)
    simclr.train()
