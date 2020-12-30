"""
Title   : Joint Embedding of Words and Labels for Text Classification
Author  : Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang,
          Dinghan Shen, Xinyuan Zhang, Ricardo Henao, Lawrence Carin
Papers  : https://arxiv.org/pdf/1805.04174.pdf
Source  : https://github.com/guoyinwang/LEAM
          https://github.com/yzye/leam
"""
import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import get_activation_func


class LEAM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int,
        ngram: int = 55,
        fc_hidden_sizes: list = [25],
        fc_activation_functions: list = ["relu"],
        fc_dropouts: list = [0.5],
        pretrained_vec=None,
        update_embedding: bool = True,
        padding_idx: int = 0,
        apply_sm: bool = True,
        device: str = "cuda",
    ):
        super(LEAM, self).__init__()

        self.num_classes = num_classes
        self.pretrained_vec = pretrained_vec
        self.apply_sm = apply_sm
        self.device = device

        self.hidden_sizes = fc_hidden_sizes
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )
        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

        self.embedding_class = nn.Embedding(num_classes, embedding_size)
        self.conv = torch.nn.Conv1d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=2 * ngram + 1, padding=ngram
        )

        self.hidden_sizes = [embedding_size] + self.hidden_sizes + [num_classes]
        modules = []
        for i in range(len(self.hidden_sizes)-1):
             modules.append(nn.Linear(in_features=self.hidden_sizes[i], out_features=self.hidden_sizes[i + 1]))
             if i < len(self.hidden_sizes)-2:
                 modules.append(get_activation_func(fc_activation_functions[i]))
                 if fc_dropouts[i] is not None:
                     if fc_dropouts[i] > 0.0:
                         modules.append(torch.nn.Dropout(p=fc_dropouts[i]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):

        # w_emb : Token Embedding
        # cls_emb : class/label Embedding
        w_emb = self.embedding(x)
        cls_emb = self.embedding_class(
            torch.tensor(
                [[i for i in range(self.num_classes)] for j in range(x.size(0))],
                device=self.device,
            )
        )

        # Joint Embeddings of Words and Labels via cosine similarity
        w_emb_norm = torch.norm(w_emb, p=2, dim=2).detach()
        w_emb_norm = w_emb.div(w_emb_norm.unsqueeze(2))
        w_emb_norm = w_emb_norm.permute(0, 2, 1)

        cls_emb_norm = torch.norm(cls_emb, p=2, dim=2).detach()
        cls_emb_norm = cls_emb.div(cls_emb_norm.unsqueeze(2))

        # beta : The compatibility / attention score for the entire text sequence
        g = torch.bmm(cls_emb_norm, w_emb_norm)
        g = F.relu(self.conv(g))
        beta = torch.max(g, 1)[0].unsqueeze(2)
        beta = F.softmax(beta, 1)

        # z : weighted averaging of word embeddings through the proposed label attentive score
        z = torch.mul(beta, w_emb)
        z = z.sum(1)

        # Optional MLP layers on top
        for m in self.module_list:
            z = m(z)

        if self.apply_sm:
            out = F.log_softmax(x, dim=1)
            return out
        else:
            return z
