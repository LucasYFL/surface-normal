import torch
import torch.nn as nn
import torch.nn.functional as F

from models.submodules.encoder import Encoder
from models.submodules.decoder import Decoder


class NNET(nn.Module):
    def __init__(self, args):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        self.decoder = Decoder(args)

    def get_1x_lr_params(self): 
        return self.encoder.parameters()

    def get_10x_lr_params(self):  
        return self.decoder.parameters()

    def forward(self, img, **kwargs):
        return self.decoder(self.encoder(img), **kwargs)