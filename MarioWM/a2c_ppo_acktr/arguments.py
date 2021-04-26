import argparse
import toml


def parse_config(args):
    config = toml.load(args.config)
    for k in config.keys():
        args.__dict__[k.replace('-', '_')] = config[k]
    return args


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--exp-name',
        help='Name for experiment used to retrieve cofngiuration file'
    )
    parser.add_argument(
        '--load',
        help='Load pre-trained model',
        default=False
    )
    parser.add_argument(
        '--device',
        help='Device used to run experiments {cpu, gpu}'
    )

    parser.add_argument(
        '--config',
        default='',
        help='toml file containing configuration for run'
    )
    args = parser.parse_args()

    if args.config:
        parse_config(args)

    return args
