import torch
import torch.nn as nn
from torch.nn import functional as F, init
from nlplay.models.pytorch.utils import get_activation_func


class MLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 300,
        embedding_mode: str = "avg",
        fc_hidden_sizes: list = [256, 128, 64],
        fc_activation_functions: list = ["relu", "relu", "relu"],
        fc_dropouts: list = [0.2, None, None],
        padding_idx: int = 0,
        pretrained_vec=None,
        update_embedding: bool = True,
        apply_sm: bool = True
    ):
        """
        Args:
            num_classes (int) : number of classes
            vocabulary_size (int): number of items in the vocabulary
            embedding_size (int): size of the embeddings
            embedding_mode (str): "avg","max" or "concat"
            fc_activation_functions (str)
            drop_out (float) : default 0.2; drop out rate applied to the embedding layer
            padding_idx (int): default 0; Embedding will not use this index
            pretrained_vec (nd.array): default None : numpy matrix containing pretrained word vectors
            update_embedding (boolean) : default True : train (True) or freeze(False) the embedding layer

        """
        super(MLP, self).__init__()

        self.embedding_mode = embedding_mode
        self.hidden_sizes = fc_hidden_sizes
        self.pretrained_vec = pretrained_vec
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )
        self.apply_sm = apply_sm
        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

        if self.embedding_mode == "concat":
            in_size = embedding_size * 2
        else:
            in_size = embedding_size

        # Dynamic setup of MLP given the input parameters
        self.hidden_sizes = [in_size] + self.hidden_sizes + [num_classes]
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
        x_embedding = self.embedding(x)
        # Pooling over the embedding
        if self.embedding_mode == "avg":
            # apply global average pooling only
            x_embedding = x_embedding.mean(dim=1)
        elif self.embedding_mode == "max":
            # apply global max pooling only
            x_embedding, _ = torch.max(x_embedding, dim=1)
        elif self.embedding_mode == "concat":
            # apply global average pooling
            x1 = x_embedding.mean(dim=1)
            # apply global max pooling
            x2, _ = torch.max(x_embedding, dim=1)
            # concat average & max pooling
            x_embedding = torch.cat((x1, x2), dim=1)

        # Apply each module of the MLP Layer setup
        x = x_embedding
        for m in self.module_list:
            x = m(x)

        if self.apply_sm:
            out = F.log_softmax(x, dim=1)
            return out
        else:
            return x





