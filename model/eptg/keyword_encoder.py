from typing import Tuple, List
from model.base.util import init_weights

import torch

from common.const.model import *
from common.const.pad import PAD_ID, UNEXPLAINED_NUMBER
from common.data import Encoded, Text, Label
from common.torch.util import stack_tensors
from model.base.chkpt import *

class KeywordEncoder(CheckpointingModule):
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
        self.model = model
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

        self.apply(lambda module: init_weights(module ,init_factor))

        self.extended_attention_mask = model.get_extended_attention_mask
        self.invert_attention_mask = model.invert_attention_mask
        if hasattr(model, 'embeddings_project'):
            self.embeddings_project = model.embeddings_project
        
        # Register prefix for generating MWP (exclude [SEP] at the end)
        # Shape [P]
        self.register_buffer('_prefix_prompt', torch.LongTensor(tokenizer.encode('generate:')[:-1]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def tok_encode(self, string: str) -> torch.Tensor:
        return self.tokenizer.encode(string, add_special_tokens=False)
    

    def encode(self, label: Label) -> Encoded:
        # Replace PAD_ID (-1) with pad of tokenizer
        model_out = self.model.forward(input_ids=label.pad_fill(self.pad_id),
                                    attention_mask=label.attn_mask_float).last_hidden_state

        return Encoded(model_out, label.pad)


    def _select_keywords(self, kwb: List[str], train: bool, drop=0) -> Tuple[torch.Tensor, str]:
        kwb_split = kwb.split()
        # kw_emb_tmp: T x Emb
        kw_emb_tmp = torch.stack([self.embeddings.word_embeddings(torch.LongTensor(self.tok_encode(kw_str)).to(self.device)).sum(0) for kw_str in kwb_split]) # sum the subword embedding to get a single embedding for a word with subwords
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
            kw_tok_tmp = self.tok_encode(kw_str)
            new_kw_tok += kw_tok_tmp
        new_sample_hard = torch.zeros(len(new_kw_tok)).to(self.device)
        new_sample_soft = torch.zeros(len(new_kw_tok)).to(self.device)
        for kw_str in new_kw:
            kw_tok_tmp = self.tok_encode(kw_str)
            length = len(kw_tok_tmp)
            new_sample_hard[kw_pos:kw_pos+length] = sample_hard[kw_idx]
            new_sample_soft[kw_pos:kw_pos+length] = sample_soft[kw_idx]
            kw_idx += 1
            kw_pos += length 

        # straight through gradient
        sample = new_sample_hard - new_sample_soft.detach() + new_sample_soft if train else new_sample_hard
        select = map(bool, sample.tolist())
        selected_kws = [w for w, s in zip(new_kw_tok, select) if s]

        return kw_logits, selected_kws


    def keywords_and_logits(self, text_keywords, train: bool):
        selected_kws_list = []
        kw_logits_list = []
        kw_batch = [tk.flatten() for tk in text_keywords]
        kw_batch = [self.tokenizer.decode(kb.indices, skip_special_tokens=True) for kb in kw_batch]

        # compute samples
        for kwb in kw_batch:
            kw_logits, selected_kws = self._select_keywords(kwb, train)
            selected_kws_list.append(selected_kws)
            kw_logits_list.append(kw_logits)
        selected_kws_batch = Label.from_list(selected_kws_list).to(self.device)
        kw_logits_batch = stack_tensors(kw_logits_list, pad_value=PAD_ID)
        assert kw_logits_batch.shape[0] == len(kw_batch)

        return selected_kws_batch, kw_logits_batch
        
        
    def forward(self, text:Text, **kwargs) -> Tuple[Encoded, Label, torch.Tensor]:
        # text: [B,S]
        # target: [B,D]
        # out: [B,D]

        # Compute keyword embedding and logits
        selected_kws, kw_logits = self.keywords_and_logits(text.keywords, self.training)
        selected_kws_enc = self.encode(selected_kws)

        return selected_kws_enc, selected_kws, kw_logits


__all__ = ['KeywordEncoder']
