import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision.models import resnet50,resnet152

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, feature_dim=128, pretrained=False):
        super(Model, self).__init__()

        self.f = resnet50(pretrained=pretrained)
        self.f.fc = Identity()
        # projection head
        self.g1 = nn.Sequential(nn.Linear(2048, 1024, bias=False),
                               nn.BatchNorm1d(1024),
                               nn.ReLU(inplace=True)
                               )
        self.g2 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True)
                                )
        self.g3=nn.Linear(512, feature_dim, bias=True)

    @amp.autocast()
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out1 = self.g1(feature)
        out2=self.g2(out1)
        out3 = self.g3(out2)
        return F.normalize(out1, dim=-1), F.normalize(out3, dim=-1)