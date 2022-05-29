from typing import Tuple, Optional, List

import torch

from common.const.model import *
from common.const.operand import VAR_MAX
from common.const.pad import PAD_ID, UNEXPLAINED_NUMBER
from common.data import Encoded, Text, Label
from model.base.chkpt import *


class KeywordSelector(CheckpointingModule):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, encoder: str = DEF_ENCODER, **kwargs):
        """
        Initiate Equation Builder instance

        :param dict config: Configuration of this model
        """
        super().__init__(encoder=encoder)

        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(encoder, add_cross_attention=True, is_decoder=True)

        tokenizer = AutoTokenizer.from_pretrained(encoder)
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.cls_token_id
        self.eos_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id
        self.tokens_ignored = {PAD_ID, self.eos_id}
        self.empty_sequence = tokenizer.encode(UNEXPLAINED_NUMBER, add_special_tokens=False)

        self.tokenizer = tokenizer

        # Copy encoder and embeddings
        self.kw_linear = torch.nn.Linear(model.config.embedding_size, model.config.vocab_size)
        self.embedding = model.embeddings.word_embeddings ### Need to check if this exists
        
        self.is_initialized = False
        self.extended_attention_mask = model.get_extended_attention_mask
        self.invert_attention_mask = model.invert_attention_mask
        if hasattr(model, 'embeddings_project'):
            self.embeddings_project = model.embeddings_project

        # Register prefix for generating MWP (exclude [SEP] at the end)
        # Shape [P]
        self.register_buffer('_prefix_mwp_gen', torch.LongTensor(tokenizer.encode('generate:')[:-1]))

    @property
    def prefix_length(self) -> int:
        return self._prefix_number.shape[0]

    def _init_kw_model(self, training: bool) -> None:
        if (not self.is_initialized) and training:
            torch.nn.init.xavier_uniform_(self.kw_linear.weight)
            torch.nn.init.zeros_(self.kw_linear.bias)
            self.kw_linear.weight.requires_grad=True
            self.kw_linear.bias.requires_grad=True
            self.embedding.requires_grad=False
            self.is_initialized = True


    def _select_keywords(self, text: Text, train: bool, drop=0):
        if not self.is_initialized:
            self._init_kw_model(train)
        kw_batch = [tk.flatten() for tk in text.keywords]
        kw_batch = [self.tokenizer.decode(kb.indices) for kb in kw_batch]

        # compute samples
        kw_emb_tmp = torch.stack([self.embedding(torch.LongTensor(self.tokenizer.encode(kw_str)).cuda()).sum(0) for kw_str in kw_batch]) # sum the subword embedding to get a single embedding for a word with subwords
        attn = torch.softmax(torch.mm(kw_emb_tmp, kw_emb_tmp.t()) / torch.sqrt(torch.FloatTensor([kw_emb_tmp.shape[1]]).cuda()) , dim=1) # dim = T x T
        kw_summary = (attn.unsqueeze(1).repeat(1, kw_emb_tmp.shape[1], 1) * kw_emb_tmp.unsqueeze(-1).repeat(1, 1, kw_emb_tmp.shape[0])).sum(-1) # shape = T x D
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
        new_kw = kw_batch
        new_kw_tok = []
        kw_pos = 0
        kw_idx = 0
        for kw_str in new_kw:
            kw_tok_tmp = self.tokenizer.encode(kw_str)
            new_kw_tok += kw_tok_tmp
        new_sample_hard = torch.zeros(len(new_kw_tok)).cuda()
        new_sample_soft = torch.zeros(len(new_kw_tok)).cuda()
        for kw_str in new_kw:
            kw_tok_tmp = self.tokenizer.encode(kw_str)
            new_sample_hard[kw_pos:kw_pos+len(kw_tok_tmp)] = sample_hard[kw_idx]
            new_sample_soft[kw_pos:kw_pos+len(kw_tok_tmp)] = sample_soft[kw_idx]
            kw_idx += 1
            kw_pos += len(kw_tok_tmp) 

        # straight through gradient
        sample = new_sample_hard - new_sample_soft.detach() + new_sample_soft if train else new_sample_hard
        
        # compute the masked keyword embeddings (wte==WordTokenEmbeddings)
        kw_emb = self.embedding(torch.LongTensor(new_kw_tok).cuda()) # dim = T x D
        if train:
            kw_emb_masked = kw_emb * sample.unsqueeze(1).expand_as(kw_emb)
        else:
            kw_emb_masked = kw_emb[sample.bool()]

        return kw_emb_masked, kw_logits, sample


    def forward(self, text: Text, **kwargs) -> Tuple[torch.LongTensor, torch.LongTensor, Label]:
        # text: [B,S]

        # Compute keyword embedding and logits
        kw_emb_masked, kw_logits, selected_kw = self._select_keywords(text, self.training)

        return kw_emb_masked, kw_logits, Label(selected_kw)


__all__ = ['KeywordSelector']
