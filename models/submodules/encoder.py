import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights

# class Encoder(nn.Module):
#     def __init__(
#         self
#     ):
#         super().__init__()
#         self.model = torchvision.models.efficientnet_v2_s(EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#     def forward(self, x):
#         # return self.model(x)
#         features = [x]
#         for k, v in self.model.features._modules.items():
#             # print(k)
#             if (k == 'blocks'):

#                 for ki, vi in v._modules.items():
#                     # print(ki)
#                     # if len(features) in (4,5,6,8,11):
#                     #     print(ki)
#                     #     print(vi)
#                     features.append(vi(features[-1]))
#             else:
#                 # if len(features) in (4,5,6,8,11):
#                 #     print(v)
#                 features.append(v(features[-1]))
#         return features
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name, pretrained=True)
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features
