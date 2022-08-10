import logging
from argparse import ArgumentParser, ArgumentTypeError
from os import cpu_count, environ
from sys import argv

from ray import tune, init, shutdown
from ray.tune.trial import Trial
from ray.tune.utils.util import is_nan_or_inf
from torch.cuda import device_count
from shutil import rmtree

from common.const.model import *
from common.trial import trial_dirname_creator_generator
from learner import *
from model import MODELS, MODEL_CLS


GPU_IDS = "0,1,2,3"
CPU_FRACTION = 1.0
GPU_FRACTION = 1.0

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.add_argument('--name', '-name', type=str, required=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    env.add_argument('--seed', '-seed', type=int, default=1)
    env.add_argument('--experiment-dir', '-exp', type=str, required=True)
    env.add_argument('--beam-expl', '-beamX', type=int, default=5)
    env.add_argument('--beam-expr', '-beamQ', type=int, default=3)

    env.add_argument('--max-iter', '-iter', type=int, default=100)
    env.add_argument('--stop-conditions', '-stop', type=str, nargs='*', default=[])

    model = parser.add_argument_group('Model')
    model.add_argument('--model', '-model', type=str, choices=MODELS.keys(), default=['EPT-G'], nargs='+')
    model.add_argument('--encoder', '-enc', type=str, default=DEF_ENCODER)
    model.add_argument('--ept-pretrained-path','-eptP', type=str, default=None)
    model.add_argument('--equation-hidden', '-eqnH', type=int, default=0)
    model.add_argument('--equation-intermediate', '-eqnI', type=int, default=0)
    model.add_argument('--equation-layer', '-eqnL', type=int, default=[6], nargs='+')
    model.add_argument('--equation-head', '-eqnA', type=int, default=0)
    model.add_argument('--keyword-shuffle', '-keyS', type=bool, default=True)
    model.add_argument('--kl-prior', '-klP', type=float, default=0.48)
    model.add_argument('--kl-coefficient', '-klC', type=float, default=0.01)

    log = parser.add_argument_group('Logger setup')
    log.add_argument('--log-path', '-log', type=str, default='./runs')
    log.add_argument('--detect-anomaly', '-da', choices=['true','false'], default='false')

    work = parser.add_argument_group('Worker setup')
    work.add_argument('--gpu-ids', '-ids', type=str, default=GPU_IDS)
    work.add_argument('--num-cpu', '-cpu', type=float, default=CPU_FRACTION)
    work.add_argument('--num-gpu', '-gpu', type=float, default=GPU_FRACTION)

    setup = parser.add_argument_group('Optimization setup')
    setup.add_argument('--float-type','-fp', type=int, choices=[16,32], default=32)
    setup.add_argument('--opt-lr', '-lr', type=float, default=[0.00176], nargs='+')
    setup.add_argument('--opt-beta1', '-beta1', type=float, default=0.9)
    setup.add_argument('--opt-beta2', '-beta2', type=float, default=0.999)
    setup.add_argument('--opt-eps', '-eps', type=float, default=1E-8)
    setup.add_argument('--opt-grad-clip', '-clip', type=float, default=10.0)
    setup.add_argument('--opt-warmup', '-warmup', type=float, default=[2], nargs='+')
    setup.add_argument('--batch-size', '-bsz', type=int, default=4)
    setup.add_argument('--starting-copy-ratio', '-cr', type=restricted_float, default=1.0)
    setup.add_argument('--copy-ratio-decrementer', '-crD', type=restricted_float, default=0.1)

    return parser.parse_args()


def build_experiment_config(args, exp_dir: str = None):
    exp_path = Path(args.experiment_dir if exp_dir is None else exp_dir)
    experiments = {}
    for file in exp_path.glob('*'):
        if not file.is_file():
            continue

        experiment_dict = {KEY_SPLIT_FILE: str(file.absolute())}
        if args.max_iter == 1:
            experiment_dict[KEY_EVAL_PERIOD] = args.max_iter
        elif file.name != KEY_TRAIN:
            experiment_dict[KEY_EVAL_PERIOD] = args.max_iter // 5 if file.name == KEY_DEV else args.max_iter

        experiments[file.name] = experiment_dict

    if KEY_DEV not in experiments:
        experiments[KEY_DEV] = experiments[KEY_TEST].copy()
        experiments[KEY_DEV][KEY_EVAL_PERIOD] = args.max_iter // 5
    return experiments


def build_configuration(args):
    return {
        KEY_SEED: args.seed,
        KEY_BATCH_SZ: args.batch_size,
        KEY_BEAM: args.beam_expr,
        KEY_BEAM_DESC: args.beam_expl,
        KEY_FP: args.float_type,
        KEY_DATASET: str(Path(args.dataset).absolute()),
        KEY_MODEL: {
            MODEL_CLS: tune.grid_search(args.model),
            MDL_ENCODER: args.encoder,
            MDL_EQUATION: {
                MDL_Q_PATH: args.ept_pretrained_path,
                MDL_Q_HIDDEN: args.equation_hidden,
                MDL_Q_INTER: args.equation_intermediate,
                MDL_Q_LAYER: tune.grid_search(args.equation_layer),
                MDL_Q_HEAD: args.equation_head
            },
            MDL_KEYWORD: {
                MDL_ENCODER: args.encoder,
                MDL_K_SHUFFLE_ON_TRAIN: args.keyword_shuffle,
                LOSS_KL_PRIOR: args.kl_prior,
                LOSS_KL_COEF: args.kl_coefficient
            },
            MDL_COPY_RATIO: args.starting_copy_ratio,
            MDL_DECREMENTER: args.copy_ratio_decrementer
        },
        KEY_RESOURCE: {
            KEY_GPU: args.num_gpu,
            KEY_CPU: args.num_cpu,
            KEY_IDS: args.gpu_ids
        },
        KEY_EXPERIMENT: build_experiment_config(args),
        KEY_GRAD_CLIP: args.opt_grad_clip,
        KEY_OPTIMIZER: {
            'type': 'lamb',
            'lr': tune.grid_search(args.opt_lr),
            'betas': (args.opt_beta1, args.opt_beta2),
            'eps': args.opt_eps,
            'debias': True
        },
        KEY_SCHEDULER: {
            'type': 'warmup-linear',
            'num_warmup_epochs': tune.grid_search(args.opt_warmup),
            'num_total_epochs': args.max_iter
        }
    }


def build_stop_condition(args):
    cond_dict = dict(training_iteration=args.max_iter)
    for condition in args.stop_conditions:
        key, value = condition.split('=')
        cond_dict[key] = float(value)

    return cond_dict


def get_experiment_name(args):
    from datetime import datetime
    now = datetime.now().strftime('%m%d%H%M%S')
    return f'{Path(args.dataset).stem}_{args.name}_{now}'


if __name__ == '__main__':
    args = read_arguments()
    if not Path(args.log_path).exists():
        Path(args.log_path).mkdir(parents=True)
            
    # Set GPU device
    environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids
    if args.num_gpu == 0:
        torch.cuda.is_available = lambda: False

    # Enable logging system
    file_handler = logging.FileHandler(filename=Path(args.log_path, 'meta.log'), encoding='UTF-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M:%S'))
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger('Hyperparameter Optimization')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info('========================= CMD ARGUMENT =============================')
    logger.info(' '.join(argv))
    init(num_cpus=cpu_count(), num_gpus=args.num_gpu)

    experiment_name = get_experiment_name(args)
    stop_condition = build_stop_condition(args)
    analysis = tune.run(SupervisedTrainer, name=experiment_name, stop=stop_condition,
                        config=build_configuration(args), local_dir=args.log_path, checkpoint_at_end=True,
                        checkpoint_freq=args.max_iter // 5, reuse_actors=True,
                        trial_dirname_creator=trial_dirname_creator_generator(), raise_on_failed_trial=False,
                        metric='dev_correct', mode='max')

    # Record trial information
    logger.info('========================= DEV. RESULTS =============================')
    logger.info('Hyperparameter search is finished!')
    trials: List[Trial] = analysis.trials
    best_scores = defaultdict(float)
    best_configs = {}
    best_trials = {}

    if len(trials) > 1:
        for trial in trials:
            if trial.status != Trial.TERMINATED:
                logger.info('\tTrial %10s (%-40s): FAILED', trial.trial_id, trial.experiment_tag)
                continue

            last_score = trial.last_result['dev_correct']
            logger.info('\tTrial %10s (%-40s): Correct %.4f on dev. set', trial.trial_id, trial.experiment_tag, last_score)

            if is_nan_or_inf(last_score):
                continue

            model_cls = trial.config[KEY_MODEL][MODEL_CLS]
            if best_scores[model_cls] < last_score:
                best_scores[model_cls] = last_score
                best_configs[model_cls] = trial.config
                best_trials[model_cls] = trial

    else:
        trial = trials[0]
        if trial.status != Trial.TERMINATED:
            logger.info('\tTrial %10s (%-40s): FAILED', trial.trial_id, trial.experiment_tag)

        last_score = trial.last_result['dev_correct']
        logger.info('\tTrial %10s (%-40s): Correct %.4f on dev. set', trial.trial_id, trial.experiment_tag, last_score)

        model_cls = trial.config[KEY_MODEL][MODEL_CLS]
        best_scores[model_cls] = last_score
        best_configs[model_cls] = trial.config
        best_trials[model_cls] = trial

    # Record the best configuration
    for cls, config in best_configs.items():
        logger.info('--------------------------------------------------------')
        logger.info('Found best configuration for %s (scored %.4f)', cls, best_scores[cls])
        logger.info(repr(config))
        logger.info('--------------------------------------------------------')

        # Save configuration as pickle and yaml
        bestpath = Path(best_trials[cls].logdir)
        chkpt_moveto = Path(bestpath.parent, 'best_%s' % cls)
        chkpt_moveto.mkdir(parents=True)
        with Path(chkpt_moveto, 'config.pkl').open('wb') as fp:
            pickle.dump(config, fp)
        with Path(chkpt_moveto, 'config.yaml').open('w+t') as fp:
            yaml_dump(config, fp, allow_unicode=True, default_style='|')

        # Save checkpoint files
        checkpoints = max(bestpath.glob('checkpoint_*'), key=lambda path: path.name)
        for file in checkpoints.glob('*.pt'):
            file.rename(chkpt_moveto / file.name)

    # Remove all checkpoints
    for chkpt_dir in Path(args.log_path).glob('**/checkpoint_*'):
        rmtree(chkpt_dir)

    shutdown()
