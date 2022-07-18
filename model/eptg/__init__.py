from typing import Tuple, List, Optional, Union
from functools import partial

import torch
from numpy.random import Generator, PCG64, randint

from common.const.model import *
from common.data import *
from common.const.operand import *
from common.const.operator import OPR_NEW_VAR_ID, OPR_NEW_EQN_ID
from common.const.pad import PAD_ID
from common.data import Text, Equation, Encoded, EquationPrediction, Label
from common.data.base import move_to
from common.data.text import text_tokenization, gather_number_toks, gather_text_toks
from model.base.util import init_weights, tie_lm_head_with_embed, logsoftmax
from model.base.beamsearch import beam_search
from model.ept import *
from preproc.num_parse import find_numbers
from .pg_head import PointerGeneratorHead
from .mwp_decoder import MWPDecoder


_OPERATOR_EXCLUDED = {OPR_NEW_EQN_ID, OPR_NEW_VAR_ID}


class MathWordProblemGenerator(EPT):
    def __init__(self, **config):
        super().__init__(**config)

        ### Need to add a new module that includes kw_linear and concats the keyword embeddings with equations and prompt
        self.mwpsource_hidden = MWPDecoder.create_or_load(**self.config[MDL_KEYWORD])
        # Head for predicting mwp
        self.mwp_pghead = PointerGeneratorHead(hidden_dim=self.equation.hidden_dim,
                                                embed_dim=self.mwpsource_hidden.embed_dim,
                                                vocab_size=self.encoder.model.config.vocab_size,
                                                init_factor=self.equation.init_factor)
        
        tie_lm_head_with_embed(self.mwp_pghead.generation_dist, self.mwpsource_hidden.embeddings.word_embeddings)

        # Variable counts (as regression problem)
        self.var_count_expand = torch.nn.Linear(self.equation.hidden_dim, self.equation.intermediate_dim)
        self.var_count_predict = torch.nn.Linear(self.equation.intermediate_dim, VAR_MAX)

        init_weights(self.var_count_expand, self.equation.init_factor)
        init_weights(self.var_count_predict, self.equation.init_factor)

        self._rng = Generator(PCG64(1))


    @property
    def _sep_token(self) -> int:
        return self.mwpsource_hidden.eos_id

    @property
    def _cls_token(self) -> int:
        return self.mwpsource_hidden.bos_id

    @property
    def _pad_token(self) -> int:
        return self.mwpsource_hidden.pad_id

    @property
    def _mask_token(self) -> int:
        return self.mwpsource_hidden.mask_id

    @property
    def _shuffle_on_training(self) -> bool:
        return self.config[MDL_KEYWORD].get(MDL_K_SHUFFLE_ON_TRAIN, True)

    def inter_values(self, type: str, is_log: bool=True) -> torch.Tensor:
        scores = torch.stack(self.mwp_pghead.intermediate_values[type])
        if is_log:
            return scores.exp()
        else:
            return scores

    def _equation_for_train(self, predict_last: bool = False,
                            **kwargs) -> Tuple[tuple, EquationPrediction]:
        # Exclude NEW_VAR operator
        return super()._equation_for_train(predict_last=predict_last, operator_excluded=_OPERATOR_EXCLUDED,
                                           **kwargs)

    def _equation_for_eval(self, **kwargs) -> Equation:
        # Exclude NEW_VAR operator
        return super()._equation_for_eval(**kwargs, excluded_operators={OPR_NEW_VAR_ID})

    def _decode_mwp_source(self, **kwargs) -> Tuple[Encoded, Encoded, tuple, int]:
        return self.mwpsource_hidden.forward(**kwargs)

    def _mwp_for_train(self, text_keywords: Label, text_equations: Label,
                            text_label: Label, 
                            text_enc: Optional[Encoded] = None, 
                            no_pred: bool = False, **kwargs) -> Tuple[Encoded, tuple, Optional[Prediction], torch.Tensor]:
        if 'cached' in kwargs and kwargs['cached'] is not None:
            # Reset cached keys
            cached = kwargs.pop('cached')
            kwargs['cached'] = cached[:-1]
            head_cache = cached[-1][0]
        else:
            head_cache = None

        # out: [B,D]
        mwp_enc, mwp_emb, key_value_cache, prefix_len, kw_logits = self._decode_mwp_source(
            text_keywords=text_keywords, text_equations=text_equations, text_label=text_label, text_enc=text_enc, **kwargs
        )

        if kwargs.get('no_pred', False):
            return mwp_enc, key_value_cache, None
        else:
            predicted, head_cache = \
                self.mwp_pghead.forward(
                    text=text_enc, 
                    text_label=text_label,
                    decoded=mwp_enc[:, prefix_len:],
                    decoder_embedding=mwp_emb[:, prefix_len:],
                    prev_key=head_cache,
                    pad_value=self._pad_token
                )

            # Append cache
            if key_value_cache is not None:
                key_value_cache = key_value_cache + (head_cache,)

            return mwp_enc, key_value_cache, Prediction(predicted), kw_logits
    

    def _mwp_for_eval(self, max_len: int = MWP_MAX, beam_size: int = 3, **kwargs) -> List[Label]:

        text_keywords: Label = kwargs['text_kw']
        text_equations: Label = kwargs['text_eq']
        # text_enc: [B,S]
        text_enc: Encoded = kwargs['text_enc']
        # text_label: [B,S]
        text_label: Label = kwargs['text_label']
        # keywords: Label = kwargs['keywords']

        batch_sz = text_enc.shape[0] if text_enc is not None else text_label.shape[0]

        def initialize_fn():
            # Initially we start with a single beam.
            beamscores = torch.zeros((batch_sz, 1))
            batch = [dict(text_keywords=text_keywords[b : b+1],
                          text_equations=text_equations[b : b+1],
                          text_enc=text_enc[b : b+1],  # [1, S]
                          text_label=text_label[b : b+1],  # [1, S]
                          cached=None)
                          for b in range(batch_sz)]
            return batch, beamscores

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M, T]
            _, kv_cache, mwp_pred, _ = self._mwp_for_train(**move_to(beams, self.device))
            # Shape [M]
            last_pred: Prediction = mwp_pred[:, -1].to('cpu')
            # Shape [M, T]
            target: Label = beams['text_label']
            # Assign cache
            beams['cached'] = move_to(kv_cache, 'cpu')

            scores = []
            for m_prev in range(target.shape[0]):
                if seq_len > 1 and target.indices[m_prev, -1].item() in {self._sep_token, PAD_ID}:
                    scores += [(0, m_prev, dict(text_label=[PAD_ID]))]
                    continue

                score_m, token_m = last_pred.log_prob[m_prev].topk(k=k + 1, dim=-1)
                for score, tok in zip(score_m.tolist(), token_m.tolist()):
                    if tok == self._sep_token and seq_len == 1:
                        continue

                    scores.append((score, m_prev, dict(text_label=[tok])))

            return scores

        def concat_next_fn(prev_beams: dict, beam_selected: List[int], list_of_next: dict):
            if prev_beams['text_label'].shape[0] == 1:
                # Before expanding beams.
                beamsz = len(beam_selected)
                for key in prev_beams:
                    if key in {'cached', 'text_label'} or prev_beams[key] is None:
                        continue
                    prev_beams[key] = prev_beams[key].repeat(beamsz)

            prev_beams['text_label'] = prev_beams['text_label'][beam_selected].extends_to(list_of_next['text_label'])

            # Select cache of selected beams. All have shape [M, N, ?, H], so we will shuffle only the first dim.
            prev_beams['cached'] = tuple(tuple(tensor[beam_selected] for tensor in pair)
                                         for pair in prev_beams['cached'])

            return prev_beams

        def is_all_finished(beams: dict):
            return all(f in {self._sep_token, PAD_ID}
                       for f in beams['text_label'].indices[:, -1].tolist())

        with torch.no_grad():
            # Execute beam search. List[Dict[str, ?]]
            batched_beams = beam_search(initialize_fn, compute_next_score_of_beam,
                                        concat_next_fn, is_all_finished, max_len, beam_size)

            # Select top-scored beam
            new_mwps = Label.build_batch(*[item['text_label'][0]
                                           for item in batched_beams])
            return new_mwps

    def _encode(self, text: Text) -> Tuple[Encoded, Encoded]:
        text_vec, num_enc =self.encoder(text)
        return text_vec, num_enc

    def encode_text_step101(self, text: Text) -> dict:
        text_vec, num_enc = self._encode(text.to(self.device))
        return dict(_text=text_vec, _number=num_enc)
    
    def generate_mwp_step102(self, text: Text, beam: int = 3, _text: Encoded = None, enforce_training: bool = False, **kwargs) -> dict:
        return_value = {}

        if self.training or enforce_training:
            # Case: Training

            # 1-3-2. Run prediction
            enc, _, pred, kw_logits = self._mwp_for_train(text_keywords=text.keywords, text_equations=text.prompt_eq, 
                                                          text_label=text.tokens, text_enc=_text)
            return_value.update({
                'mwp': pred,
                '_mwp_enc': enc,
                'kw_logits': kw_logits
            })
        else:
            # Case: Evaluation & generation required (by default)
            
            # 1-3-2. Run prediction
            print(f'generating MWP..\n')
            mwp = self._mwp_for_eval(text_kw=text.keywords, text_eq=text.prompt_eq, 
                                     text_label=text.tokens, text_enc=_text)
            return_value.update({
                'mwp': mwp,
                '_mwp_enc': mwp  #: Copy for internal use
            })

        return return_value

    def reconstruct_mwp_step201(self, copy_ratio: float, text: Text, mwp: Union[Prediction, Label]) -> Text:
        ### TODO: 100% -> 0% gold set에서 가져오도록 설계 (일종의 warmup)
        with torch.no_grad():
            concat_mwps = []
            concat_numbers = []
            tokenizer = self.mwpsource_hidden.tokenizer

            # text_tmp_batch = text.to_human_readable(tokenizer=partial(tokenizer.decode, skip_special_tokens=True))['tokens']
            raw_text_tmp_batch = text.to_human_readable(tokenizer=tokenizer)['raw']
            if self.training:
                mwp_tmp_batch = mwp.to_human_readable(converter=partial(tokenizer.decode, skip_special_tokens=True))['prediction']
            else:
                mwp_tmp_batch = [
                    mwp[b].flatten().to_human_readable(converter=partial(tokenizer.decode, skip_special_tokens=True))['target']
                    for b in range(mwp.shape[0])
                ]
                copy_ratio = 0

            for mwp_tmp_b, raw_text_tmp_b in zip(mwp_tmp_batch, raw_text_tmp_batch):
                # copy certain ratio from gold text
                if copy_ratio == 1:
                    combined_text = raw_text_tmp_b
                elif copy_ratio == 0 and mwp_tmp_b != '':
                    combined_text = mwp_tmp_b
                else:
                    raw_text_tmp_b_split = raw_text_tmp_b.split()
                    mwp_tmp_b_split = mwp_tmp_b.split()
                    orig_len = len(raw_text_tmp_b_split)
                    gen_len = len(mwp_tmp_b_split)
                    if gen_len == 0:
                        mwp_tmp_b = raw_text_tmp_b_split[randint(0, orig_len-1)]
                    if copy_ratio == 0:
                        combined_text = mwp_tmp_b
                    else:
                        from_orig = raw_text_tmp_b_split[:int(orig_len * copy_ratio)]
                        from_gen = mwp_tmp_b_split[int(gen_len * (1-copy_ratio)):]
                        combined_text = ' '.join(from_orig + from_gen)
                    
                numbers: dict = find_numbers(combined_text)
                spaced, orig_to_new_wid, tokens = text_tokenization(combined_text, tokenizer)
                tokens = gather_text_toks(tokens, tokenizer)
                token_nids = gather_number_toks(tokens, spaced, orig_to_new_wid, \
                                                numbers, tokenizer)

                assert len(tokens) == len(token_nids)

                concat_mwps.append(Label.from_list(tokens))
                concat_numbers.append(Label.from_list(token_nids))

        return Text(raw=None, tokens=Label.build_batch(*concat_mwps), numbers=Label.build_batch(*concat_numbers),
                    keywords=None, prompt_eq=None)
    

    def generate_eqn_step202(self, eqn_tgt: Equation, _text: Encoded = None, _number: Encoded = None, beam: int = 3, **kwargs) -> dict:
        return_value = {}
        eqn_kwargs = {} if _number is not None else {'text_label': kwargs['_label'],
                                                     'num_label': kwargs['_num_label']}

        if self.training:
            equation = self._equation_for_train(target=eqn_tgt, text=_text, number=_number, **eqn_kwargs)[-1]
            return_value['equation'] = equation
            return_value['equation_tgt'] = eqn_tgt
        else:
            equation = self._equation_for_eval(text=_text, number=_number, beam_size=beam, **eqn_kwargs)
            return_value['equation'] = equation

        return return_value


    def forward_mwp(self, text: Text, beam_mwp: int = 3, **kwargs) -> Tuple[dict, dict]:
        # Prepare kwargs
        return_value = {}
        
        return_value.update(self.encode_text_step101(text))
        
        # Generate math word problem
        return_value.update(self.generate_mwp_step102(text, beam=beam_mwp, **return_value))

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value


    def forward_equation(self, copy_ratio: float, text: Text, mwp: Union[Prediction, Label], equation: Equation = None, 
                         beam: int = 3, **kwargs) -> Tuple[dict, dict]:

        # (2-1) New Math Word Problem: Need to change Prediction class to Text class..
        new_text = self.reconstruct_mwp_step201(copy_ratio, text, mwp)

        # Compute MWP vector (Re-use step 1-1)
        encode_result = self.encode_text_step101(new_text.to(self.device))

        # (2-2) Read/Generate Equation
        return_value = self.generate_eqn_step202(eqn_tgt=equation, beam=beam, **encode_result)

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value


    def forward(self, copy_ratio: float, text: Text, **kwargs):
        # Ignore OPR_NEW_VAR_ID on training
        return_value = {'eqn_ignore': {OPR_NEW_VAR_ID}}

        """ (Phase 1) Generating math word problems """
        p1_external, p1_internal = self.forward_mwp(text, **kwargs)
        return_value.update(p1_external)

        """ (Phase 2) Building solution equations """
        p2_external, p2_internal = self.forward_equation(copy_ratio, text, **p1_internal, **kwargs)
        return_value.update(p2_external)

        return return_value


__all__ = ['MathWordProblemGenerator']
