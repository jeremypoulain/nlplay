"""
Title    : Deep Pyramid Convolutional Neural Networks for Text Categorization - 2017
Authors  : Rie Johnson, Tong Zhang
Papers   : https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
Source   : https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/dpcnn.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        embedding_size: int = 300,
        num_blocks: int = 2,
        pooling_stride: int = 2,
        num_kernels: int = 16,
        kernel_size: int = 3,
        dropout: float = 0.2,
        pretrained_vec=None,
        update_embedding: bool = True,
        pad_index: int = 0,
        apply_sm: bool = True,
    ):
        super(DPCNN, self).__init__()

        self.num_classes = num_classes
        self.apply_sm = apply_sm
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.pretrained_vec = pretrained_vec
        self.update_embedding = update_embedding

        self.embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, padding_idx=pad_index
        )
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        self.embedding.weight.requires_grad = self.update_embedding

        self.num_kernels = num_kernels
        self.pooling_stride = pooling_stride
        self.kernel_size = kernel_size

        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "DPCNN kernel size should be odd!"

        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                self.embedding_size,
                self.num_kernels,
                self.kernel_size,
                padding=self.radius,
            )
        )

        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(
                        self.num_kernels,
                        self.num_kernels,
                        self.kernel_size,
                        padding=self.radius,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(
                        self.num_kernels,
                        self.num_kernels,
                        self.kernel_size,
                        padding=self.radius,
                    ),
                )
                for i in range(num_blocks + 1)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = torch.nn.Linear(self.num_kernels, self.num_classes)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.permute(0, 2, 1)

        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features

        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride
            )
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features

        doc_embedding = F.max_pool1d(conv_features, conv_features.size(2)).squeeze()

        out = self.dropout(self.fc1(doc_embedding))
        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
