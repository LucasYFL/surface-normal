import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
# newly implemented by us
class Encoder(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.model = torchvision.models.efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    def forward(self, x):
        # return self.model(x)
        features = [x]
        for k, v in self.model.features._modules.items():
            # print(k)
            if (k == 'blocks'):

                for ki, vi in v._modules.items():
                    # print(ki)
                    # if len(features) in (4,5,6,8,11):
                    #     print(ki)
                    #     print(vi)
                    features.append(vi(features[-1]))
            else:
                # if len(features) in (4,5,6,8,11):
                #     print(v)
                features.append(v(features[-1]))
        return features
    