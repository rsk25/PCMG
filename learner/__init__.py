import math
import pickle
from collections import defaultdict
from functools import partial
from typing import Optional

from ray.tune.resources import Resources
from ray.tune.result import TIMESTEPS_THIS_ITER
from ray.tune.trainable import Trainable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from yaml import dump as yaml_dump
import time

from common.const.model import MDL_COPY_RATIO, MDL_DECREMENTER, MDL_ENCODER, MDL_KEYWORD, LOSS_KL_COEF, LOSS_KL_PRIOR
from common.const.pad import FLOAT_NAN, PAD_ID
from common.dataset import *
from common.tester import Tester
from common.torch.loss import SmoothedCrossEntropyLoss
from model import model_create, EPT, MODEL_CLS
from .const import *
from .util import *

SMOOTHED_CROSS_ENTROPY_LOSS = SmoothedCrossEntropyLoss(ignore_index=PAD_ID)
scaler = torch.cuda.amp.GradScaler()


class SupervisedTrainer(Trainable):
    def __init__(self, config=None, logger_creator=None):
        # Training config
        self._batch_size: int = 0
        # Dataset
        self._dataset: Optional[Dataset] = None
        # Model for learning
        self._model: Optional[EPT] = None
        # Tester
        self._tester: Tester = Tester()
        # Training/Evaluation configuration
        self._train_config: dict = {}
        self._eval_configs: dict = {}
        self._beam_eqn: int = 3
        self._beam_expl: int = 5
        # Optimization config
        self._optimizer: Optional[Optimizer] = None
        self._scheduler: Optional[LambdaLR] = None
        self._grad_clip: float = 0.0

        self._fp: int = 32

        self._kl_prior: float = 0.0
        self._kl_coef: float = 0.0

        self._copy_ratio: float = 1
        self._decrement: float = 0.1

        # Initialize Trainable
        super().__init__(config, logger_creator)

        # Store setup of the current experiment.
        with Path(self.logdir, 'trainer.log').open('w+t', encoding='UTF-8') as fp:
            fp.write('Initializing %s has been finished.\n' % self.__class__.__name__)
            fp.write('\n--------------------  System specification ---------------------\n')
            fp.write(read_system_spec())
            fp.write('\n-------------------- Trainer configuration ---------------------\n')
            fp.write(yaml_dump(config, allow_unicode=True))
            fp.write('\n--------------------   Model structure     ---------------------\n')
            fp.write(str(self._model))
            fp.write('\n--------------------   Model parameters    ---------------------\n')
            params = [(n, p.numel()) for n, p in self._model.named_parameters()]
            fp.write('\n'.join([f'{n}: {p}' for n, p in params]))
            fp.write('\nTOTAL: %s\n' % sum([x for _, x in params]))
            fp.write('\n-------------------- Dataset statistics    ---------------------\n')
            fp.write(yaml_dump(self._dataset.statistics, allow_unicode=True))

    @classmethod
    def default_resource_request(cls, config: dict) -> Resources:
        cls._validate_config(config)

        resource = config[KEY_RESOURCE]
        return Resources(
            cpu=resource[KEY_CPU],
            gpu=resource[KEY_GPU]
        )

    @classmethod
    def _validate_config(cls, config):
        assert KEY_DATASET in config
        assert KEY_MODEL in config
        assert KEY_OPTIMIZER in config
        assert KEY_SEED in config
        assert KEY_BATCH_SZ in config
        assert KEY_BEAM in config
        assert KEY_BEAM_DESC in config
        assert KEY_RESOURCE in config
        assert KEY_EXPERIMENT in config

        assert type(config[KEY_DATASET]) is str
        assert type(config[KEY_MODEL]) is dict
        assert type(config[KEY_OPTIMIZER]) is dict
        assert type(config[KEY_SEED]) is int
        assert type(config[KEY_BATCH_SZ]) is int
        assert type(config[KEY_BEAM]) is int
        assert type(config[KEY_BEAM_DESC]) is int
        assert type(config[KEY_RESOURCE]) is dict
        assert type(config[KEY_EXPERIMENT]) is dict

        assert config[KEY_BATCH_SZ] > 0

        assert KEY_CPU in config[KEY_RESOURCE]
        assert KEY_GPU in config[KEY_RESOURCE]

        if KEY_GRAD_CLIP in config:
            assert isinstance(config[KEY_GRAD_CLIP], (int, float))

        if KEY_SCHEDULER in config:
            assert type(config[KEY_SCHEDULER]) is dict

    @classmethod
    def get_trial_name(cls, config, trial_id):
        # Naming convention: [MODEL]-[ID]
        # Get model's trial name
        return '%s-%s' % (config[KEY_MODEL][MODEL_CLS], trial_id)

    def stop(self):
        super().stop()
        self._tester.close()

    def setup(self, config):
        super().setup(config)

        # Setup logging level of transformers to ERROR
        from transformers import logging
        logging.set_verbosity_error()
        self.reset_config(config)
    
    def reset_config(self, new_config):
        # Set seed
        set_seed(new_config[KEY_SEED])

        # Set batch size
        self._batch_size = new_config[KEY_BATCH_SZ]

        self._kl_prior = new_config[KEY_MODEL][MDL_KEYWORD][LOSS_KL_PRIOR]
        self._kl_coef = new_config[KEY_MODEL][MDL_KEYWORD][LOSS_KL_COEF]
        self._decrement = new_config[KEY_MODEL][MDL_DECREMENTER]
        self._copy_ratio = new_config[KEY_MODEL][MDL_COPY_RATIO]

        # Set beam size
        self._beam_eqn = new_config[KEY_BEAM]
        self._beam_expl = new_config[KEY_BEAM_DESC]

        # Read dataset
        if self._dataset is None:
            self._dataset = Dataset(path=new_config[KEY_DATASET], langmodel=new_config[KEY_MODEL][MDL_ENCODER],
                                    seed=new_config[KEY_SEED])
        else:
            self._dataset.reset_seed(new_config[KEY_SEED])

        # Store experiment setup
        experiments = new_config[KEY_EXPERIMENT]
        self._train_config = experiments.pop(KEY_TRAIN)
        self._eval_configs = experiments

        # Load training set
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])

        # Build models
        self._fp = new_config[KEY_FP]
        self._model = model_create(new_config[KEY_MODEL].copy())
        if torch.cuda.is_available():
            self._model.cuda()

        # Build or Re-build optimizer
        step_per_epoch = math.ceil(self._dataset.num_items / self._batch_size)
        self._set_optimizer(new_config[KEY_OPTIMIZER])
        self._set_grad_clip(new_config.get(KEY_GRAD_CLIP))  # This can be None.
        if KEY_SCHEDULER in new_config:
            new_config[KEY_SCHEDULER]['step_per_epoch'] = step_per_epoch
            self._set_scheduler(**new_config[KEY_SCHEDULER])
        else:
            self._set_scheduler()

        return True

    def step(self):
        # Prepare metrics
        report = dict()

        # Run scripts before updating
        report['before'] = self._before_update()

        # Run training
        report['train'] = self._update_module()
        report[TIMESTEPS_THIS_ITER] = report['train'][TIMESTEPS_THIS_ITER]

        # Run evaluation periodically
        executed_split = {}
        # if self._copy_ratio > 0:
        #     self._copy_ratio -= self._decrement ### for each step, decrease copy ratio of gold text
        iter_after_pretrain = self._iteration + 1

        for key, config in self._eval_configs.items():
            period = config[KEY_EVAL_PERIOD]
            split = config.get(KEY_SPLIT_FILE, '')
            if iter_after_pretrain % period == 0 and split:
                if split in executed_split:
                    # Avoid multiple running on the same split.
                    report[key] = report[executed_split[split]]
                else:
                    report[key] = self._evaluate(key, config)
                    executed_split[split] = key

        # Run scripts after updating
        report['after'] = self._after_update()

        # Add metric shortcut of development set
        report['dev_correct'] = report.get(KEY_DEV, {}).get('correct', FLOAT_NAN)

        return report

    def __getstate__(self) -> dict:
        return {
            KEY_MODEL: self._model.state_dict(),
            'rng': {
                'numpy': numpy.random.get_state(),
                'random': random.getstate(),
                'torch': {
                    'cpu': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                },
                'dataset': self._dataset.get_rng_state()
            },
            'iteration': self._iteration,
            'copy_ratio': self._copy_ratio,
            'optimizer': self._optimizer.state_dict(),
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None
        }

    def __setstate__(self, state: dict):
        # Load rng
        random_states = state['rng']
        numpy.random.set_state(random_states['numpy'])
        random.setstate(random_states['random'])
        self._dataset.set_rng_state(random_states['dataset'])

        torch.set_rng_state(random_states['torch']['cpu'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_states['torch']['cuda'])

        # Load iteration
        self._iteration = state['iteration']
        self._copy_ratio = state['copy_ratio']

        # Load policy
        self._model.load_state_dict(state[KEY_MODEL])
        if self._optimizer is not None and 'optimizer' in state:
            self._optimizer.load_state_dict(state['optimizer'])
        if self._scheduler is not None and 'scheduler' in state:
            self._scheduler.load_state_dict(state['scheduler'])

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir, 'chkpt')
        with checkpoint_path.open('wb') as fp:
            pickle.dump(self.__getstate__(), fp)

        # Save model & tokenizer
        self._model.save(tmp_checkpoint_dir)
        with Path(tmp_checkpoint_dir, 'tokenizer.pt').open('wb') as fp:
            torch.save(self._dataset.tokenizer, fp)

        rotate_checkpoint(tmp_checkpoint_dir, max_item=1)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint):
        with Path(checkpoint).open('wb') as fp:
            extra_data = pickle.load(fp)

        self.__setstate__(extra_data)

    def _set_grad_clip(self, param):
        assert param is None or param >= 0
        self._grad_clip = param

    def _set_optimizer(self, kwargs):
        name = kwargs.pop('type')
        if name == 'lamb':
            from torch_optimizer import Lamb
            cls = Lamb
        elif name == 'radam':
            from torch_optimizer import RAdam
            cls = RAdam
        elif name == 'adabound':
            from torch_optimizer import AdaBound
            cls = AdaBound
        elif name == 'yogi':
            from torch_optimizer import Yogi
            cls = Yogi
        elif name == 'adamw':
            from transformers import AdamW
            cls = AdamW
        elif name == 'rmsprop':
            from torch.optim.rmsprop import RMSprop
            cls = RMSprop
        else:
            from torch.optim.sgd import SGD
            cls = SGD

        self._optimizer = cls([params
                               for key, params in self._model.named_parameters()
                               if ('encoder.model.embeddings' not in key) and ('mwpsource_hidden.embeddings' not in key)\
                                and ('equation' not in key)],
                              **kwargs)

    def _set_scheduler(self, **kwargs):
        name = kwargs.pop('type') if 'type' in kwargs else ''
        if name == 'warmup-linear':
            from .scheduler import LinearWarmupLinearDecay
            self._scheduler = LinearWarmupLinearDecay(self._optimizer, **kwargs)
        elif name == 'warmup-constant':
            from .scheduler import LinearWarmupNoDecay
            self._scheduler = LinearWarmupNoDecay(self._optimizer, **kwargs)
        else:
            self._scheduler = None

    def _after_backprop(self):

        if self._fp == 16:
            scaler.unscale_(self._optimizer)

        if self._grad_clip is not None and self._grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)

        skip_lr_sched = False
        if self._fp == 16:
            scaler.step(self._optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())
        else:
            self._optimizer.step()
        if not skip_lr_sched:
            self._scheduler.step()
        
        self._model.zero_grad()


    def _after_update(self) -> dict:
        metric = {}
        if self._scheduler is not None:
            metric['lr'] = max(self._scheduler.get_last_lr())
        elif self._optimizer is not None:
            metric['lr'] = max(group['lr'] for group in self._optimizer.param_groups)

        weight_sizes = defaultdict(list)
        for key, weight in self._model.named_parameters():
            wkey = 'other'
            if 'encoder' in key:
                wkey = 'encoder'
            elif 'equation' in key:
                wkey = 'equation'
            elif 'explanation' in key:
                wkey = 'explanation'

            if '_embedding' in key:
                wkey += '_embed'
            if 'bias' in key:
                wkey += '_bias'

            weight_sizes[wkey].append(weight.detach().abs().mean().item())

        metric['weight'] = {key: sum(values) / len(values) if values else FLOAT_NAN
                            for key, values in weight_sizes.items()}

        return metric

    def _record_evaluation_output(self, experiment_name: str, output: dict):
        with Path(self.logdir, '%s.yaml' % experiment_name).open('w+t', encoding='UTF-8') as fp:
            fp.write('# Output of experiment %s in iteration %s.\n' % (experiment_name, self.iteration))
            fp.write('# Total %d items are tested.\n' % len(output['dump']))
            yaml_dump(output, fp, allow_unicode=True)

    def _train(self):
        raise ValueError('Trainer._train() should not be called!')

    def _save(self, tmp_checkpoint_dir):
        raise ValueError('Trainer._save() should not be called!')

    def _restore(self, checkpoint):
        raise ValueError('Trainer._restore() should not be called!')

    def _evaluate(self, name: str, configuration: dict) -> dict:
        self._dataset.select_items_with_file(configuration[KEY_SPLIT_FILE])
        self._model.eval()
        tokenizer = self._dataset.tokenizer

        output_pairs = []
        with torch.no_grad():
            for batch in self._dataset.get_minibatches(self._batch_size, for_testing=True):
                output = self._model.forward(copy_ratio=self._copy_ratio, 
                                             beam=self._beam_eqn,
                                             **batch.to(self._model.device).as_dict())

                for b in range(batch.batch_size):
                    item = batch.item_of_batch(b)
                    pairs = dict(
                        equation=(item.equation, output['equation'][b]),
                        mwp=(
                            tokenizer.decode(item.text.tokens.pad_fill(tokenizer.pad_token_id), skip_special_tokens=True).strip(),
                            tokenizer.decode(output['mwp'][b].pad_fill(tokenizer.pad_token_id), skip_special_tokens=True).strip()
                        )
                    )

                    output_pairs.append((item, pairs))

            results = self._tester.check(output_pairs, tokenizer=tokenizer)
            self._record_evaluation_output(name, results)
        # Remove 'dump' key before returning
        results.pop('dump')
        return results

    def _before_update(self) -> dict:
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])
        self._model.train()
        return {}

    def _update_module(self) -> dict:
        reports = []

        for batch in self._dataset.get_minibatches(self._batch_size):
            # ---- Input ----
            # text: Text [B, S]
            # equation: Equation [B, T]
            # explanation: B-List of Explanation [N/V, D]
            # ---- Output ----
            # equation: EquationPrediction [B, T]
            # num_expl?: B-List of Prediction [N, D]
            # var_expl?: B-List of Prediction [V, D] or Prediction [B, VD]
            # var_target?: Label [B, VD]
            
            if self._fp == 16:
                with torch.cuda.amp.autocast():
                    out_dict = self._model.forward(self._copy_ratio, iteration=self._iteration, **batch.to(self._model.device).as_dict())

                    # Compute loss
                    losses = batch.loss_calculation(self._kl_prior, self._kl_coef, **out_dict)
            else:
                out_dict = self._model.forward(self._copy_ratio, iteration=self._iteration, **batch.to(self._model.device).as_dict())

                # Compute loss
                losses = batch.loss_calculation(self._kl_prior, self._kl_coef, **out_dict)
            
            # Compute accuracy of tokens
            with torch.no_grad():
                report = batch.accuracy_of(**out_dict)

            # Build sum of losses
            total_loss = sum(losses.values())
            losses['total'] = total_loss
            report.update({'loss_' + key: value for key, value in losses.items()})

            # Add to report (cast to float to avoid memory leak)
            reports.append({key: float(value) for key, value in report.items() if key != 'seq'})

            # Run Backward prop.
            if self._fp == 16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            self._after_backprop()

        report = merge_reports(reports)
        report[TIMESTEPS_THIS_ITER] = len(reports)
        return report
