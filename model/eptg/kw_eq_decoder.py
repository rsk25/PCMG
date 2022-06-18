from typing import Tuple, Optional, List
from functools import partial
from model.base.util import init_weights

import torch

from common.const.model import *
from common.const.operand import VAR_MAX
from common.const.pad import PAD_ID, UNEXPLAINED_NUMBER
from common.data import Encoded, Text, Label, label
from common.torch.util import stack_tensors
from model.base.chkpt import *


class KeywordEquationEncoder(CheckpointingModule):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, encoder: str = DEF_ENCODER, init_factor: float = 0.01, **kwargs):
        """
        Initiate Equation Builder instance

        :param dict config: Configuration of this model
        """
        super().__init__(encoder=encoder)

        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(encoder, add_cross_attention=True, is_decoder=True)
        # Copy encoder and embeddings
        self.kw_linear = torch.nn.Linear(model.config.embedding_size, model.config.vocab_size)
        self.embeddings = model.embeddings
        self.embed_dim = model.config.embedding_size

        tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.tokens_ignored = {PAD_ID, self.eos_id}
        self.empty_sequence = tokenizer.encode(UNEXPLAINED_NUMBER, add_special_tokens=False)

        self.tokenizer = tokenizer
        self.encoder = model.encoder

        # self.is_initialized = False
        self.apply(lambda module: init_weights(module ,init_factor))

        self.extended_attention_mask = model.get_extended_attention_mask
        self.invert_attention_mask = model.invert_attention_mask
        if hasattr(model, 'embeddings_project'):
            self.embeddings_project = model.embeddings_project
        
        # Register prefix for generating MWP (exclude [SEP] at the end)
        # Shape [P]
        self.register_buffer('_prefix_prompt', torch.LongTensor(tokenizer.encode('generate:')[:-1]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # def _init_kw_model(self, training: bool) -> None:
    #     if (not self.is_initialized) and training:
    #         torch.nn.init.xavier_uniform_(self.kw_linear.weight)
    #         torch.nn.init.zeros_(self.kw_linear.bias)
    #         self.kw_linear.weight.requires_grad=True
    #         self.kw_linear.bias.requires_grad=True
    #         self.embeddings.requires_grad=False
    #         self.is_initialized = True


    def _encode(self, string: str) -> torch.Tensor:
        return self.tokenizer.encode(string, add_special_tokens=False)


    def _select_keywords(self, kwb: List[str], train: bool, drop=0) -> Tuple[torch.Tensor, str]:
        kwb_split = kwb.split()
        # kw_emb_tmp: T x Emb
        kw_emb_tmp = torch.stack([self.embeddings.word_embeddings(torch.LongTensor(self._encode(kw_str)).to(self.device)).sum(0) for kw_str in kwb_split]) # sum the subword embedding to get a single embedding for a word with subwords
        # attn: T x T
        attn = torch.softmax(torch.mm(kw_emb_tmp, kw_emb_tmp.t()) / torch.sqrt(torch.FloatTensor([kw_emb_tmp.shape[1]]).to(self.device)) , dim=1) 
        # kw_summary: T x Emb
        kw_summary = (attn.unsqueeze(1).repeat(1, kw_emb_tmp.shape[1], 1) * kw_emb_tmp.unsqueeze(-1).repeat(1, 1, kw_emb_tmp.shape[0])).sum(-1) 
        # kw_logits: T x Vocab
        kw_logits = self.kw_linear(kw_summary)
        sample_soft = torch.sigmoid(kw_logits)[:,0]
        if train:
            sample_hard = torch.distributions.bernoulli.Bernoulli(probs=sample_soft).sample()
        else:
            sample_hard = (sample_soft > 0.5).float() * 1

            if drop > 0:
                selected = torch.where(sample_hard == 1)[0]
                selected_to_drop = selected[torch.bernoulli(torch.tensor([drop] * len(selected))).bool()]
                sample_hard[selected_to_drop] = 0

        if torch.count_nonzero(sample_hard) == 0:
            # random select a keyword if non are chosen
            sample_hard[torch.randint(low=0,high=len(sample_hard),size=[1])[0]] = 1

        # get a vector of length kw_tok (a keyword may have more than 1 token and thus the sample vector needs to be expanded)
        new_kw = kwb_split
        new_kw_tok = []
        kw_pos = 0
        kw_idx = 0
        for kw_str in new_kw:
            kw_tok_tmp = self._encode(kw_str)
            new_kw_tok += kw_tok_tmp
        new_sample_hard = torch.zeros(len(new_kw_tok)).to(self.device)
        new_sample_soft = torch.zeros(len(new_kw_tok)).to(self.device)
        for kw_str in new_kw:
            kw_tok_tmp = self._encode(kw_str)
            length = len(kw_tok_tmp)
            new_sample_hard[kw_pos:kw_pos+length] = sample_hard[kw_idx]
            new_sample_soft[kw_pos:kw_pos+length] = sample_soft[kw_idx]
            kw_idx += 1
            kw_pos += length 

        # straight through gradient
        sample = new_sample_hard - new_sample_soft.detach() + new_sample_soft if train else new_sample_hard
        select = map(bool, sample.tolist())
        selected_kws = [w for w, s in zip(new_kw_tok, select) if s]
        # # compute the masked keyword embeddings (wte==WordTokenEmbeddings)
        # kw_emb = self.embedding(torch.LongTensor(new_kw_tok).to(self.device)) # dim = T x D
        # if train:
        #     kw_emb_masked = kw_emb * sample.unsqueeze(1).expand_as(kw_emb)
        # else:
        #     kw_emb_masked = kw_emb[sample.bool()]

        return kw_logits, selected_kws


    def _create_input_ids(self, keywords: Label, equations: Label, target: Label, prefix: Label = None) -> Tuple[Label, int]:
        # Concatenate prefix and keyword & equation labels.  [P] + [B, T] + [B, Eq] -> [B, P+T+Eq]
        tmp = keywords.prepend(self._prefix_prompt)
        context_len = tmp.shape[-1]
        # Extend target with prefix. [B, D] -> [B, P+T+D]
        input_ids = Label.concat(prefix, tmp, target, dim=1)
        # context_input = Label.concat(tmp, equations, dim=1)
        # context_len = context_input.shape[-1]
        # # Extend target with prefix. [B, D] -> [B, P+T+D]
        # input_ids = Label.concat(prefix, context_input, target, dim=1)
        input_ids_copy = input_ids.copy()
        input_ids_for_debug = input_ids_copy.flatten().to_human_readable(converter=partial(self.tokenizer.decode, skip_special_tokens=True))['target']
        return input_ids, context_len, input_ids_for_debug


    def build_input(self, text: Text, train: bool):
        # if not self.is_initialized:
        #     self._init_kw_model(train)
        selected_kws_list = []
        kw_logits_list = []
        kw_batch = [tk.flatten() for tk in text.keywords]
        kw_batch = [self.tokenizer.decode(kb.indices, skip_special_tokens=True) for kb in kw_batch]

        # compute samples
        for kwb in kw_batch:
            kw_logits, selected_kws = self._select_keywords(kwb, train)
            ### Need to collect kw_logits for loss calculation
            selected_kws_list.append(selected_kws)
            kw_logits_list.append(kw_logits)
        selected_kws_batch = Label.from_list(selected_kws_list).to(self.device)
        kw_logits_batch = stack_tensors(kw_logits_list, pad_value=PAD_ID)
        assert kw_logits_batch.shape[0] == len(kw_batch)
        
        input_ids, context_len, _ = self._create_input_ids(selected_kws_batch, text.prompt_eq, text.tokens)
        assert input_ids.is_batched
        # Build token-type indices. [T] -> [1, T]
        token_type = torch.arange(input_ids.shape[-1]).ge(context_len).long().unsqueeze(0).to(self.device)

        # As we may add 'text_label' vector and do want to apply it after adding the vector,
        # we will explicitly call word_embedding here.
        word = self.embeddings.word_embeddings(input_ids.pad_fill(self.pad_id))

        # Compute entire embedding [B, P+D, H] or [B, 1, H]
        embeddings = self.embeddings(inputs_embeds=word, token_type_ids=token_type)
        if hasattr(self, 'embeddings_project'):
            embeddings = self.embeddings_project(embeddings)

        # Wrap as Encoded instance
        word = Encoded(word, input_ids.pad)
        embeddings = Encoded(embeddings, input_ids.pad)

        return word, embeddings, context_len, kw_logits_batch


    def build_context(self, embedding: Encoded, text: Encoded = None,
                              prev_key_value: tuple = None) -> Tuple[Encoded, tuple]:
        if (not self.training) and (prev_key_value is not None):
            # Cached: we need only the last token
            embedding = embedding[:, -1:]

        # Build attention masks
        # Note: we need full mask (raw_input_ids.attn_mask_float) even if we cached
        extended_attention_mask = self.extended_attention_mask(embedding.attn_mask_float,
                                                               embedding.shape, embedding.device)
        extended_text_mask = self.invert_attention_mask(text.attn_mask_float) if text is not None else None

        # Compute hidden states [B, P+D, H]
        outputs = self.encoder.forward(
            embedding.vector,
            attention_mask=extended_attention_mask,  # [B, H, P+D, P+D]
            head_mask=[None] * self.encoder.config.num_hidden_layers,
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
        # On evaluation, return cached output. otherwise None
        next_key_value = None if self.training else outputs.past_key_values

        return encoded, next_key_value


    def forward(self, text: Text, text_enc: Encoded, 
                **kwargs) -> Tuple[Label, Encoded, Encoded, Optional[tuple], int, torch.Tensor]:
        # text: [B,S]
        # target: [B,D]
        # out: [B,D]

        # Whether key-value pair is cached or not
        cached = kwargs.get('cached', None)
        is_cached = (not self.training) and (cached is not None)

        # Compute keyword embedding and logits
        word_emb, full_emb, prefix_len, kw_logits = self.build_input(text, self.training)

        # Compute hidden state vectors
        encoded, cached = self.build_context(full_emb, text_enc, cached)

        if is_cached:
            # Cached: we need only the last token (encoded has already been cut)
            word_emb = word_emb[:, -1:]

        return encoded, word_emb, cached, (0 if is_cached else prefix_len), kw_logits


__all__ = ['KeywordEquationEncoder']
