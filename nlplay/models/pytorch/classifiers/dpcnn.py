"""
Title    : Deep Pyramid Convolutional Neural Networks for Text Categorization - 2017
Authors  : Rie Johnson, Tong Zhang
Papers   : https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
Source   : https://github.com/Cheneng/
           https://github.com/Cheneng/DPCNN/blob/master/model/DPCNN.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nlplay.models.pytorch.utils import get_activation_func


class DPCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        num_classes,
        activation_function="relu",
        channel_size=250,
        embedding_size=300,
        drop_out=0.0,
        pretrained_vec=None,
        update_embedding=True,
        pad_index=0,
    ):
        super(DPCNN, self).__init__()

        self.embedding_size = embedding_size
        self.pretrained_vec = pretrained_vec
        self.channel_size = channel_size
        self.update_embedding = update_embedding
        self.input_dropout_p = drop_out
        self.num_classes = num_classes
        self.vocabulary_size = vocabulary_size
        self.pretrained_vec = pretrained_vec

        self.embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, padding_idx=pad_index
        )
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        self.embedding.weight.requires_grad = self.update_embedding

        self.input_dropout = nn.Dropout(p=self.input_dropout_p)

        self.conv_region_embedding = nn.Conv2d(
            1, self.channel_size, (3, self.embedding_size), stride=1
        )
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))

        self.act_fun = get_activation_func(activation_function.lower())

        self.linear_out = nn.Linear(self.channel_size, self.num_classes)

    def forward(self, input_var, lengths=None):
        embeded = self.embedding(input_var)
        embeded = self.input_dropout(embeded)
        batch, width, height = embeded.shape
        embeded = embeded.view((batch, 1, width, height))

        # Region embedding
        x = self.conv_region_embedding(embeded)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] >= 2:
            x = self._block(x)

        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)
        x = F.log_softmax(x, dim=1)
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x
