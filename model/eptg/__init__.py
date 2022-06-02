from typing import Tuple, List, Optional
from model.ept import attention

import torch
from numpy.random import Generator, PCG64, randint

from common.const.model import *
from common.data import *
from common.const.operand import *
from common.const.operator import OPR_NEW_VAR_ID, OPR_NEW_EQN_ID
from common.const.pad import PAD_ID
from common.data import Text, Equation, Explanation, Encoded, EquationPrediction, Label
from common.data.base import move_to
from model.base.util import init_weights, tie_lm_head_with_embed, logsoftmax
from model.base.beamsearch import beam_search, EXPL_BEAM_SZ
from model.ept import *
from .pg_head import PointerGeneratorHead
from .kw_eq_decoder import KeywordEquationDecoder

SWAN_OPERATOR_EXCLUDED = {OPR_NEW_EQN_ID, OPR_NEW_VAR_ID}


class MathWordProblemGenerator(EPT):
    def __init__(self, **config):
        super().__init__(**config)

        ### Need to add a new module that includes kw_linear and concats the keyword embeddings with equations and prompt
        self.mwpsource_hidden = KeywordEquationDecoder.create_or_load(**self.config[MDL_KEYWORD])
        # Head for predicting mwp
        self.mwp_pghead = PointerGeneratorHead(hidden_dim=self.equation.hidden_dim,
                                                vocab_size=self.encoder.model.config.vocab_size,
                                                init_factor=self.equation.init_factor,
                                                debug=False)
        
        tie_lm_head_with_embed(self.mwp_pghead.generation_dist, self.mwpsource_hidden.embeddings.word_embeddings)

        # Variable counts (as regression problem)
        self.var_count_expand = torch.nn.Linear(self.equation.hidden_dim, self.equation.intermediate_dim)
        self.var_count_predict = torch.nn.Linear(self.equation.intermediate_dim, VAR_MAX)

        init_weights(self.var_count_expand, self.equation.init_factor)
        init_weights(self.var_count_predict, self.equation.init_factor)

        self._rng = Generator(PCG64(1))
   

    @property
    def _sep_token(self) -> int:
        return self.kw_model.eos_id

    @property
    def _cls_token(self) -> int:
        return self.kw_model.bos_id

    @property
    def _pad_token(self) -> int:
        return self.kw_model.pad_id

    @property
    def _mask_token(self) -> int:
        return self.kw_model.mask_id

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
        return super()._equation_for_train(predict_last=predict_last, operator_excluded=SWAN_OPERATOR_EXCLUDED,
                                           **kwargs)

    def _equation_for_eval(self, **kwargs) -> Equation:
        # Exclude NEW_VAR operator
        return super()._equation_for_eval(**kwargs, excluded_operators={OPR_NEW_VAR_ID})

    def _decode_mwp_source(self, **kwargs) -> Tuple[Encoded, Encoded, tuple, int]:
        return self.mwpsource_hidden.forward(**kwargs)

    def _mwp_for_train(self, text: Text,
                            text_enc: Optional[Encoded], 
                            text_label: Label, 
                            keyword_candidates: List[Label],
                            no_pred: bool = False, **kwargs) -> Tuple[Encoded, tuple, Optional[Prediction]]:
        if 'cached' in kwargs and kwargs['cached'] is not None:
            # Reset cached keys
            cached = kwargs.pop('cached')
            kwargs['cached'] = cached[:-1]
            head_cache = cached[-1][0]
        else:
            head_cache = None

        # out: [B,D]
        mwp_enc, mwp_emb, key_value_cache, prefix_len = self._decode_mwp_source(text= text, **kwargs)

        if kwargs.get('no_pred', False):
            return mwp_enc, key_value_cache, None
        else:
            predicted, head_cache = \
                self.explanation_pghead.forward(text=text_enc, text_label=text_label,
                                                prev_key=head_cache,
                                                pad_value=self._pad_token, decoded=mwp_enc[:, prefix_len:],
                                                decoder_embedding=mwp_emb[:, prefix_len:])

            # Append cache
            if key_value_cache is not None:
                key_value_cache = key_value_cache + (head_cache,)

            return mwp_enc, key_value_cache, Prediction(predicted)
    
    def _mwp_batched_for_train(self, text: Optional[Encoded], 
                                    text_label: Label, 
                                    keyword_candidates: List[Label],
                                    no_pred: bool = False) -> Tuple[List[Encoded], List[Prediction]]:
        encoded = []
        predictions = []

        for b, _ in enumerate(text):

            keywords_b = keyword_candidates[b]  # [N, T] : keyword candidates
            keyword_sz = keyword_candidates.shape[0]

            text_b = text[b:b + 1].repeat(keyword_sz) if text is not None else None  # [1, S]
            text_label_b = text_label[b:b + 1].repeat(keyword_sz)  # [1, S]

            kw_enc, _, mwp_pred = \
                self._mwp_for_train(keywords=keywords_b, text_label=text_label_b,
                                            target=text_b, no_pred=no_pred)
                # self._explanation_for_train(text=text_b, text_label=text_label_b, expl_label=num_snippet_b,
                #                             target=expl_b, no_pred=no_pred)

            encoded.append(kw_enc)
            predictions.append(mwp_pred)

        # Return encoded B-List of [N, D] and prediction B-List of [N, D]
        return encoded, predictions

    def _mwp_for_eval(self, max_len: int = EXPL_MAX, beam_size: int = 3, **kwargs) -> List[Label]:
        assert 'text' in kwargs
        # text: [B,S]
        text: Encoded = kwargs['text']
        # text_label: [B,S]
        text_label: Label = kwargs['text_label']
        # expl_label: B-list of [N,X]
        expl_label: List[Label] = kwargs['expl_label']

        batch_sz = len(expl_label)

        # out: [B,N,D]
        # beam: [B,N,M,D]. <-- [BN, M, D]
        lengths = [item.shape[0] for item in expl_label]

        def initialize_fn():
            # Initially we start with a single beam.
            flattened_items = []
            for b in range(batch_sz):
                text_b = text[b:b + 1] if text is not None else None  # [1, S]
                text_label_b = text_label[b:b + 1]  # [1, S]

                for n in range(lengths[b]):
                    expl_for_bn = expl_label[b][n:n + 1]
                    flattened_items.append(dict(text=text_b,  # [1, S]
                                                text_label=text_label_b,  # [1, S]
                                                expl_label=expl_for_bn,  # [1, 1] or [1, T]
                                                target=Label.from_list([[self._sep_token]])))  # [M=1, T=1]

            beamscores = torch.zeros((len(flattened_items), 1))
            return flattened_items, beamscores

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M, T]
            _, kv_cache, expl_pred = self._mwp_batched_for_train(**move_to(beams, self.device))
            # Shape [M]
            last_pred: Prediction = expl_pred[:, -1].to('cpu')
            # Shape [M, T]
            target: Label = beams['target']
            # Assign cache
            beams['cached'] = move_to(kv_cache, 'cpu')

            scores = []
            for m_prev in range(target.shape[0]):
                if seq_len > 1 and target.indices[m_prev, -1].item() in {self._sep_token, PAD_ID}:
                    scores += [(0, m_prev, dict(target=[PAD_ID]))]
                    continue

                score_m, token_m = last_pred.log_prob[m_prev].topk(k=k + 1, dim=-1)
                for score, tok in zip(score_m.tolist(), token_m.tolist()):
                    if tok == self._sep_token and seq_len == 1:
                        continue

                    scores.append((score, m_prev, dict(target=[tok])))

            return scores

        def concat_next_fn(prev_beams: dict, beam_selected: List[int], list_of_next: dict):
            if prev_beams['expl_label'].shape[0] == 1:
                # Before expanding beams.
                beamsz = len(beam_selected)
                for key in prev_beams:
                    if key in {'cached', 'target'} or prev_beams[key] is None:
                        continue
                    prev_beams[key] = prev_beams[key].repeat(beamsz)

            prev_beams['target'] = prev_beams['target'][beam_selected].extends_to(list_of_next['target'])

            # Select cache of selected beams. All have shape [M, N, ?, H], so we will shuffle only the first dim.
            prev_beams['cached'] = tuple(tuple(tensor[beam_selected] for tensor in pair)
                                         for pair in prev_beams['cached'])

            return prev_beams

        def is_all_finished(beams: dict):
            return all(f in {self._sep_token, PAD_ID}
                       for f in beams['target'].indices[:, -1].tolist())

        with torch.no_grad():
            # Execute beam search. List[Dict[str, ?]]
            batched_beams = beam_search(initialize_fn, compute_next_score_of_beam,
                                        concat_next_fn, is_all_finished, max_len, beam_size)

            # Select top-scored beam
            explanations = []
            for b, len_b in enumerate(lengths):
                if len_b > 0:
                    explanations.append(Label.build_batch(*[item['target'][0]
                                                            for item in batched_beams[:len_b]]))
                    batched_beams = batched_beams[len_b:]
                else:
                    # Add empty explanation, [0, 0]
                    explanations.append(Label(torch.full((0, 0), fill_value=PAD_ID, dtype=torch.long)))

            return explanations

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

            # 1-3-2. Run prediction for each target
            enc, _, pred = self._mwp_for_train(text=_text, text_label=text.tokens, keyword_candidates= text.keywords)
            return_value.update({
                'mwp': pred,
                '_mwp_enc': enc
            })
        else:
            # Case: Evaluation & generation required (by default)
            
            # 1-3-2. Run prediction for each target
            print(f'generating MWP..\n')
            mwp = self._mwp_for_eval(text=text, text_enc=_text, text_label=text.tokens, keyword_candidates= text.keywords)
            return_value.update({
                'mwp': mwp,
                '_mwp_enc': mwp  #: Copy for internal use
            })

        return return_value

    def reconstruct_problem_step201(self, text: Text, num_expl: List[Label], var_expl: List[Label], **kwargs) -> Text:
        with torch.no_grad():
            batch_sz = len(num_expl)
            concat_labels = []
            concat_numbers = []

            for b, (use_text, use_recon) in enumerate(self._get_recombine_policy(batch_sz)):
                assert use_text or use_recon
                text_b = text[b].tokens.indices
                textnum_b = text[b].numbers.indices
                num_max = num_expl[b].shape[0]

                if use_text:
                    concat_b = [tok for tok in text_b.tolist() if tok != PAD_ID]
                    if use_recon:
                        numpos_b = [PAD_ID] * len(concat_b)
                    else:
                        numpos_b = [tok for tok in textnum_b.tolist()][:len(concat_b)]
                else:
                    concat_b = [self._cls_token]
                    numpos_b = [PAD_ID]

                if use_recon:
                    expl_b = num_expl[b].indices.tolist() + var_expl[b].indices.tolist()
                    concat_set_b = []
                    numpos_set_b = []
                    for nid, expl_bn in enumerate(expl_b):
                        expl_bn = [tok for tok in expl_bn if tok not in self.explanation.tokens_ignored]
                        if expl_bn == self.explanation.empty_sequence:
                            continue

                        num_bn = [PAD_ID] * len(expl_bn)

                        if nid < num_max:
                            expl_bn += self._is_a_number
                            num_bn += [PAD_ID] * len(self._is_a_number)
                            num_bn[-2] = nid

                            num_tokens = text_b.masked_select(textnum_b.eq(nid)).tolist()
                            expl_bn = expl_bn[:-1] + num_tokens + expl_bn[-1:]
                            num_bn = num_bn[:-1] + ([nid] * len(num_tokens)) + num_bn[-1:]
                        else:
                            expl_bn = self._what_is[:-1] + expl_bn + self._what_is[-1:]
                            num_bn = [PAD_ID] * len(expl_bn)
                            num_bn[0] = nid

                        concat_set_b.append(expl_bn)
                        numpos_set_b.append(num_bn)

                        assert len(concat_set_b) == len(numpos_set_b) > 0
                        assert nid in num_bn

                    # Concatenate all explanations
                    concat_b += sum(concat_set_b, [])
                    numpos_b += sum(numpos_set_b, [])
                else:
                    # We should add variables as SWAN treats variables as written numbers.
                    for vid in range(var_expl[b].shape[0]):
                        var_text = self.explanation.var_labels[vid].indices.tolist()
                        concat_b += var_text
                        numpos_b += [vid + num_max] * len(var_text)

                concat_b.append(self._sep_token)
                numpos_b.append(PAD_ID)

                if len(concat_b) > 500:
                    concat_b = concat_b[-500:]
                    numpos_b = numpos_b[-500:]

                concat_labels.append(Label.from_list(concat_b))
                concat_numbers.append(Label.from_list(numpos_b))

            return Text(raw=None, tokens=Label.build_batch(*concat_labels), numbers=Label.build_batch(*concat_numbers),
                        snippets=None)

    def generate_eqn_step203(self, equation: Equation, _text: Encoded = None, _number: Encoded = None, beam: int = 3, **kwargs) -> dict:
        return_value = {}
        eqn_kwargs = {} if _number is not None else {'text_label': kwargs['_label'],
                                                     'num_label': kwargs['_num_label']}

        if self.training:
            number_len: List[int] = kwargs['number_len']
            eqn_tgt: Equation = equation.treat_variables_as_defined(number_len)
            if _number is None:
                _num_label: Label = kwargs['_num_label']
                eqn_tgt: Equation = eqn_tgt.treat_text_as_prev_result(_num_label)

            equation = self._equation_for_train(target=eqn_tgt, text=_text, number=_number, **eqn_kwargs)[-1]
            return_value['equation'] = equation
            return_value['equation_tgt'] = eqn_tgt
        else:
            equation = self._equation_for_eval(text=_text, number=_number, beam_size=beam, **eqn_kwargs)
            if _number is None:
                _num_label: Label = kwargs['_num_label']
                return_value['equation'] = equation.restore_numbers(_num_label)

            number_len: List[int] = kwargs['number_len']
            variable_len: List[int] = kwargs['variable_len']
            return_value['equation'] = equation.restore_variables(number_len, variable_len)

        return return_value

    def forward_mwp(self, text: Text, dont_generate_mwp: bool = False, beam_mwp: int = 3, **kwargs) -> Tuple[dict, dict]:
        # Prepare kwargs
        return_value = {}
        
        return_value.update(self.encode_text_step101(text))
        # Generate math word problem
        if self.training or not dont_generate_mwp:
            return_value.update(self.generate_mwp_step102(text, beam=beam_mwp, **return_value))

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value

    def forward_equation(self, text: Text, equation: Equation = None, beam: int = 3,
                         _num_expl: List[Label] = None, _var_expl: List[Label] = None,
                         **kwargs) -> Tuple[dict, dict]:

        # (2-1) Generate New Math Word Problem
        new_text = self.reconstruct_problem_step201(text, _num_expl, _var_expl)

        # (2-2) Compute MWP vector (Re-use step 1-1)
        encode_result = self.encode_text_step101(new_text.to(self.device))

        # (2-3) Read/Generate Equation
        # rsk: The length of numbers and variables should be changed so that they come directly from dataset
        number_len = [d.shape[0] for d in _num_expl]
        variable_len = [d.shape[0] for d in _var_expl]
        return_value = self.generate_eqn_step203(equation=equation, number_len=number_len,
                                                 variable_len=variable_len, beam=beam,
                                                 **encode_result)

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value

    def forward(self, text: Text, **kwargs):
        # Ignore OPR_NEW_VAR_ID on training
        return_value = {'eqn_ignore': {OPR_NEW_VAR_ID}}

        """ (Phase 1) Generating math word problems """
        p1_external, p1_internal = self.forward_mwp(text, **kwargs)
        return_value.update(p1_external)

        """ (Phase 2) Building solution equations """
        p2_external, p2_internal = self.forward_equation(text, **p1_internal, **kwargs)
        return_value.update(p2_external)

        return return_value


__all__ = ['MathWordProblemGenerator']
