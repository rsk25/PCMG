import tqdm

from common.const.model import MDL_Q_LAYER, MDL_EQUATION
from learner import *
from train_model import read_arguments, build_configuration
from model import MODEL_CLS
from yaml import dump

from torch.autograd import detect_anomaly


def build_configuration_supervised(args):
    options = build_configuration(args)
    options[KEY_OPTIMIZER]['lr'] = args.opt_lr[0]
    options[KEY_SCHEDULER]['num_warmup_epochs'] = args.opt_warmup[0]
    options[KEY_MODEL][MDL_EQUATION][MDL_Q_LAYER] = args.equation_layer[0]
    options[KEY_MODEL][MODEL_CLS] = args.model[0]

    return options


if __name__ == '__main__':
    args = read_arguments()
    args.algorithm = 'supervised'
    if args.num_gpu == 0:
        torch.cuda.is_available = lambda: False

    if not Path(args.log_path).exists():
        Path(args.log_path).mkdir(parents=True)

    with detect_anomaly(args.detect_anomaly):
        algorithm = SupervisedTrainer(build_configuration_supervised(args))
        for i in tqdm.trange(args.max_iter):
            print('------------------------------------------')
            print(dump(algorithm.train()))

        algorithm.stop()
