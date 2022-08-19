from typing import Tuple, List, Optional, Union
from functools import partial
from copy import copy

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
from torch.autograd import variable
from .pg_head import PointerGeneratorHead
from .keyword_encoder import KeywordEncoder


_OPERATOR_EXCLUDED = {OPR_NEW_EQN_ID, OPR_NEW_VAR_ID}


class MathWordProblemGenerator(EPT):
    def __init__(self, **config):
        super().__init__(**config)

        ### Need to add a new module that includes kw_linear and concats the keyword embeddings with equations and prompt
        self.mwpsource_hidden = KeywordEncoder.create_or_load(**self.config[MDL_KEYWORD])
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

        self.register_buffer('_prefix_prompt', torch.LongTensor(self.mwpsource_hidden.tokenizer.encode('generate:')[:-1]))


    def _check_with_human_readable(self, obj) -> None:
        obj_copy = obj.copy()
        if isinstance(obj, Label):
            print(self.mwpsource_hidden.tokenizer.decode(obj_copy.pad_fill(self.mwpsource_hidden.tokenizer.pad_token_id), skip_special_tokens=True).strip())


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

    
    def prompt_gen(self, keywords: Label, equations: Label, target: Label, 
                    no_equations: bool = False, prefix: Label = None) -> Tuple[Label, int]:
        # Concatenate prefix and keyword & equation labels.  [P] + [B, T] + [B, Eq] -> [B, P+T+Eq]
        tmp = keywords.prepend(self._prefix_prompt)
        # Extend target with prefix. [B, D] -> [B, P+T+D]
        if no_equations:
            context_len = tmp.shape[-1]
            input_ids = Label.concat(prefix, tmp, target, dim=1)
        else:
            context_input = Label.concat(tmp, equations, dim=1)
            context_len = context_input.shape[-1]
            input_ids = Label.concat(prefix, context_input, target, dim=1)
        
        # self._check_with_human_readable(keywords[0])
        # self._check_with_human_readable(tmp[0])
        # self._check_with_human_readable(input_ids[0])
        return input_ids, context_len

    def build_context(self, embedding: Encoded, text: Encoded = None,
                      prev_key_value: tuple = None) -> Tuple[Encoded, tuple]:
        if (not self.training) and (prev_key_value is not None):
            # Cached: we need only the last token
            embedding = embedding[:, -1:]

        # Build attention masks
        # Note: we need full mask (raw_input_ids.attn_mask_float) even if we cached
        extended_attention_mask = self.mwpsource_hidden.extended_attention_mask(embedding.attn_mask_float,
                                                               embedding.shape, embedding.device)
        extended_text_mask = self.mwpsource_hidden.invert_attention_mask(text.attn_mask_float) if text is not None else None

        # Compute hidden states [B, P+D, H]
        outputs = self.mwpsource_hidden.encoder.forward(
            embedding.vector,
            attention_mask=extended_attention_mask,  # [B, H, P+D, P+D]
            head_mask=[None] * self.mwpsource_hidden.encoder.config.num_hidden_layers,
            encoder_hidden_states=text.vector if text is not None else None,
            encoder_attention_mask=extended_text_mask,  # [B, ?, ?, S]
            output_attentions=False,
            output_hidden_states=False,
            past_key_values=prev_key_value,
            return_dict=True,
            use_cache=not self.training  # Use caching if this is for evaluation
        )
        # Truncate the prefix [B, D]
        encoded = Encoded(outputs.last_hidden_state, embedding.pad)
        # outputs.last_hidden_state == outputs[0]
        # On evaluation, return cached output. otherwise None
        next_key_value = None if self.training else outputs.past_key_values

        return encoded, next_key_value


    def _mwp_for_train(self, selected_kws: Label, text_equations: Label,
                       text_label: Label,
                       target: Label,
                       selected_kw_enc: Optional[Encoded] = None,
                       no_pred: bool = False, **kwargs) -> Tuple[Encoded, tuple, Optional[Prediction]]:
        if 'cached' in kwargs and kwargs['cached'] is not None:
            # Reset cached keys
            cached = kwargs.pop('cached')
            kwargs['cached'] = cached[:-1]
            head_cache = cached[-1][0]
        else:
            head_cache = None

        # out: [B,D]
        # mwp_enc, mwp_emb, key_value_cache, prefix_len, kw_logits = self._decode_mwp_source(
        #     text_keywords=text_keywords, text_equations=text_equations, text_label=text_label, text_enc=text_enc, **kwargs
        # )

        # Whether key-value pair is cached or not
        cached = kwargs.get('cached', None)
        is_cached = (not self.training) and (cached is not None)

        input_ids, context_len = self.prompt_gen(selected_kws, text_equations, target, no_equations=False)
        assert input_ids.is_batched

        # for b in range(text_label.shape[0]):
        #     print(input_ids[b].flatten().to_human_readable(converter=partial(self.mwpsource_hidden.tokenizer.decode, skip_special_tokens=True))['target'])

        # Build token-type indices. [T] -> [1, T]
        token_type = torch.arange(input_ids.shape[-1]).ge(context_len).long().unsqueeze(0).to(self.device)


        # As we may add 'text_label' vector and do want to apply it after adding the vector,
        # we will explicitly call word_embedding here.
        word = self.mwpsource_hidden.embeddings.word_embeddings(input_ids.pad_fill(self.mwpsource_hidden.pad_id))

        # Compute entire embedding [B, P+D, H] or [B, 1, H]
        embeddings = self.mwpsource_hidden.embeddings(inputs_embeds=word, token_type_ids=token_type)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)

        # Wrap as Encoded instance
        mwp_emb = Encoded(word, input_ids.pad)
        full_emb = Encoded(embeddings, input_ids.pad)

        mwp_enc, key_value_cache = self.build_context(full_emb, selected_kw_enc, cached)

        if is_cached:
            # Cached: we need only the last token (encoded has already been cut)
            mwp_emb = mwp_emb[:, -1:]

        prefix_len = 0 if is_cached else context_len
        
        text_enc = self.mwpsource_hidden.encode(text_label)

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

            return mwp_enc, key_value_cache, Prediction(predicted)
    

    def _mwp_for_eval(self, max_len: int = MWP_MAX, beam_size: int = 3, **kwargs) -> List[Label]:

        selected_kws: Label = kwargs['selected_kws']
        text_equations: Label = kwargs['text_equations']
        text_label: Label = kwargs['text_label']
        selected_kw_enc: Encoded = kwargs['selected_kw_enc']

        batch_sz = selected_kw_enc.shape[0] if selected_kw_enc is not None else selected_kw_enc.shape[0]

        def initialize_fn():
            # Initially we start with a single beam.
            beamscores = torch.zeros((batch_sz, 1))
            batch = [dict(selected_kws=selected_kws[b : b+1],
                          text_equations=text_equations[b : b+1],
                          selected_kw_enc=selected_kw_enc[b : b+1],  # [1, S]
                          text_label=text_label[b : b+1],
                          target=Label.from_list([[self._sep_token]]),
                          cached=None)
                          for b in range(batch_sz)]
            return batch, beamscores

        def compute_next_score_of_beam(seq_len: int, beams: dict, k: int):
            # Shape [M, T]
            _, kv_cache, mwp_pred = self._mwp_for_train(**move_to(beams, self.device))
            # Shape [M]
            last_pred: Prediction = mwp_pred[:, -1].to('cpu')
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
            if prev_beams['text_label'].shape[0] == 1:
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
            new_mwps = Label.build_batch(*[item['target'][0]
                                           for item in batched_beams])
            return new_mwps


    def keyword_select_step101(self, text: Text):
        selected_kws_enc, selected_kws, kw_logits = self.mwpsource_hidden.forward(text)
        return dict(selected_kws_enc=selected_kws_enc, 
                    selected_kws=selected_kws, 
                    kw_logits=kw_logits)
    
    def generate_mwp_step102(self, text: Text, selected_kws_enc: Encoded, selected_kws: Label,
                            beam: int = 3, **kwargs) -> dict:
        return_value = {}

        if self.training:
            # Case: Training

            # 1-3-2. Run prediction
            enc, _, pred = self._mwp_for_train(selected_kws=selected_kws, text_equations=text.prompt_eq, 
                                               selected_kw_enc=selected_kws_enc, text_label=text.tokens,
                                               target=text.tokens)
            
            # for b in range(text.shape[0]):
            #     print(pred[b].to_human_readable(converter=partial(self.mwpsource_hidden.tokenizer.decode))['prediction'])
            
            return_value.update({
                'mwp': pred,
                '_mwp_enc': enc,
            })
        else:
            # Case: Evaluation & generation required (by default)
            
            # 1-3-2. Run prediction
            print(f'generating MWP..\n')
            mwp = self._mwp_for_eval(selected_kws=selected_kws, text_equations=text.prompt_eq, 
                                     selected_kw_enc=selected_kws_enc, text_label=text.tokens,
                                     **kwargs)

            return_value.update({
                'mwp': mwp,
                '_mwp_enc': mwp  #: Copy for internal use
            })

        return return_value

    def reconstruct_mwp_step201(self, copy_ratio: float, text: Text, mwp: Union[Prediction, Label]) -> Text:
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
    

    def _encode(self, text: Text) -> Tuple[Encoded, Encoded]:
        text_vec, num_enc =self.encoder(text)
        return text_vec, num_enc

    def encode_text_step202(self, text: Text) -> dict:
        text_vec, num_enc = self._encode(text.to(self.device))
        return dict(_text=text_vec, _number=num_enc)

    def _predict_var_count(self, encoded: Encoded, **kwargs) -> torch.Tensor:
        # Value should be at least 1.0
        return self.var_count_predict(self.var_count_expand(encoded.vector[:, 0]).relu())

    def predict_varcount_step203(self, _text: Encoded = None, variable_len: List[int] = None,
                                 **kwargs) -> dict:
        return_value = {}
        var_len_tensor = self._predict_var_count(_text, **kwargs)
        if self.training:
            return_value['var_len'] = Prediction(logsoftmax(var_len_tensor))  # [B, |V|]
            # Index should begin from 0
            return_value['var_len_tgt'] = Label(torch.tensor(variable_len, device=self.device, dtype=torch.long) - 1)
        else:
            # Override var_len_list variable with predicted result
            return_value['_var_lengths'] = [int(max_id) + 1  # Var count should begin with 1
                                            for max_id in var_len_tensor.argmax(dim=-1).tolist()]
        return return_value


    def generate_eqn_step204(self, mwp: Prediction, equation: Equation, _text: Encoded = None, 
                             _number: Encoded = None, beam: int = 3, **kwargs) -> dict:
        return_value = {}
        eqn_kwargs = {} if _number is not None else {'text_label': kwargs['_label'],
                                                     'num_label': kwargs['_num_label']}

        def gumbel_softmax_mwp(mwp_logits, pad):
            _softmax = torch.nn.Softmax(dim=-1)
            comparison = _softmax(mwp_logits)
            mwp_gumbel = torch.nn.functional.gumbel_softmax(mwp_logits, tau=1, hard=False, eps=1e-10, dim=-1)
            mwp_embed = torch.matmul(
                mwp_gumbel, 
                self.mwpsource_hidden.embeddings.word_embeddings.weight
            )
            return Encoded(mwp_embed, pad[:,:mwp_embed.shape[1]])
        
        if self.training:
            mwp_gumbel_embed = gumbel_softmax_mwp(mwp.log_prob, _text.pad)

            number_len: List[int] = kwargs['number_len']
            eqn_tgt: Equation = equation.treat_variables_as_defined(number_len)
            if _number is None:
                _num_label: Label = kwargs['_num_label']
                eqn_tgt: Equation = eqn_tgt.treat_text_as_prev_result(_num_label)

            
            equation = self._equation_for_train(target=eqn_tgt, text=mwp_gumbel_embed, number=_number, **eqn_kwargs)[-1]
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


    def forward_mwp(self, text: Text, beam_mwp: int = 3, **kwargs) -> Tuple[dict, dict]:
        # Prepare kwargs
        return_value = {}
        
        return_value.update(self.keyword_select_step101(text))
        
        # Generate math word problem
        return_value.update(self.generate_mwp_step102(text=text, beam=beam_mwp, **return_value))
        if self.training:
            print(f"prediction: {return_value['mwp'][0].to_human_readable(converter=partial(self.mwpsource_hidden.tokenizer.decode, skip_special_tokens=True))['prediction']}")
        else:
            print(f"prediction: {return_value['mwp'][0].flatten().to_human_readable(converter=partial(self.mwpsource_hidden.tokenizer.decode, skip_special_tokens=True))['target']}")

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value


    def forward_equation(self, copy_ratio: float, text: Text, mwp: Union[Prediction, Label], equation: Equation = None, 
                         beam: int = 3, **kwargs) -> Tuple[dict, dict]:

        return_value = {}

        _info: List[ExtraInfo] = kwargs['info']
        return_value.update(number_len=[len(info.numbers.keys()) for info in _info],
                            variable_len=[len(info.answers[0]) for info in _info])

        # (2-1) New Math Word Problem: Need to change Prediction class to Text class..
        new_text = self.reconstruct_mwp_step201(copy_ratio, text, mwp)

        # (2-2) Compute MWP vector
        return_value.update(self.encode_text_step202(new_text.to(self.device)))

        # (2-3) Predict var count
        return_value.update(self.predict_varcount_step203(**return_value))

        # (2-4) Read/Generate Equation
        return_value.update(self.generate_eqn_step204(mwp=mwp, equation=equation, beam=beam, **return_value))

        # Separate internal outputs
        external = {}
        for key in return_value:
            if not key.startswith('_'):
                external[key] = return_value[key]

        return external, return_value


    def forward(self, copy_ratio: float, text: Text, phase1_only: bool = False, **kwargs):
        # Ignore OPR_NEW_VAR_ID on training
        return_value = {'eqn_ignore': {OPR_NEW_VAR_ID}}

        """ (Phase 1) Generating math word problems """
        p1_external, p1_internal = self.forward_mwp(text, **kwargs)
        return_value.update(p1_external)

        if not phase1_only:
            """ (Phase 2) Building solution equations """
            p2_external, p2_internal = self.forward_equation(copy_ratio, text, **p1_internal, **kwargs)
            return_value.update(p2_external)

        return return_value


class MathWordProblemGenerator_GeneratorOnly(MathWordProblemGenerator):
    def __init__(self, **config):
        super().__init__(**config)

    def forward_equation(self, copy_ratio: float, text: Text, mwp: Union[Prediction, Label], equation: Equation = None, 
                         beam: int = 3, **kwargs) -> Tuple[dict, dict]:
        if self.training:
            # Do nothing for the equations
            return {}, {}
        else:
            # Return empty equation
            batchsz = text.shape[0]
            external = dict(equation=Equation.get_generation_base(batchsz))
            return external, {}


class MathWordProblemGenerator_NoPGN(MathWordProblemGenerator):
    def __init__(self, **config):
        super().__init__(**config)

    
    def _mwp_for_train(self, selected_kws: Label, text_equations: Label,
                       text_label: Label,
                       selected_kw_enc: Optional[Encoded] = None, 
                       no_pred: bool = False, **kwargs) -> Tuple[Encoded, tuple, Optional[Prediction]]:
        if 'cached' in kwargs and kwargs['cached'] is not None:
            # Reset cached keys
            cached = kwargs.pop('cached')
            kwargs['cached'] = cached[:-1]
            head_cache = cached[-1][0]
        else:
            head_cache = None

        # out: [B,D]
        # mwp_enc, mwp_emb, key_value_cache, prefix_len, kw_logits = self._decode_mwp_source(
        #     text_keywords=text_keywords, text_equations=text_equations, text_label=text_label, text_enc=text_enc, **kwargs
        # )

        # Whether key-value pair is cached or not
        cached = kwargs.get('cached', None)
        is_cached = (not self.training) and (cached is not None)

        input_ids, context_len = self.prompt_gen(selected_kws, text_equations, text_label, no_equations=False)
        assert input_ids.is_batched

        # print(f"prompt :{input_ids[0].flatten().to_human_readable(converter=partial(self.mwpsource_hidden.tokenizer.decode))['target']}")

        # Build token-type indices. [T] -> [1, T]
        token_type = torch.arange(input_ids.shape[-1]).ge(context_len).long().unsqueeze(0).to(self.device)


        # As we may add 'text_label' vector and do want to apply it after adding the vector,
        # we will explicitly call word_embedding here.
        word = self.mwpsource_hidden.embeddings.word_embeddings(input_ids.pad_fill(self.mwpsource_hidden.pad_id))

        # Compute entire embedding [B, P+D, H] or [B, 1, H]
        embeddings = self.mwpsource_hidden.embeddings(inputs_embeds=word, token_type_ids=token_type)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)

        # Wrap as Encoded instance
        mwp_emb = Encoded(word, input_ids.pad)
        full_emb = Encoded(embeddings, input_ids.pad)

        mwp_enc, key_value_cache = self.build_context(full_emb, selected_kw_enc, cached)

        if is_cached:
            # Cached: we need only the last token (encoded has already been cut)
            mwp_emb = mwp_emb[:, -1:]

        prefix_len = 0 if is_cached else context_len


        if kwargs.get('no_pred', False):
            return mwp_enc, key_value_cache, None
        else:
            predicted, head_cache = \
                self.mwp_pghead.forward(
                    text=selected_kw_enc, 
                    text_label=selected_kws,
                    decoded=mwp_enc[:, prefix_len:],
                    decoder_embedding=mwp_emb[:, prefix_len:],
                    prev_key=head_cache,
                    pad_value=self._pad_token,
                    no_copying=True
                )

            # Append cache
            if key_value_cache is not None:
                key_value_cache = key_value_cache + (head_cache,)

            return mwp_enc, key_value_cache, Prediction(predicted)
    

__all__ = [
    'MathWordProblemGenerator', 
    'MathWordProblemGenerator_GeneratorOnly',
    'MathWordProblemGenerator_NoPGN'
]
