import torch
import torch.nn as nn
import torchvision
import timm

class Model(nn.Module):
    def __init__(self,output=9):

        super(Model, self).__init__()

        self.m = timm.create_model('densenet169', pretrained=True,num_classes=output)
        self.m.features.conv0=nn.Conv2d(in_channels=15,out_channels=64,kernel_size=(7,7),stride=(2, 2), padding=(3, 3),bias=False)

    def forward(self,x):

        out=self.m(x)

        return out

if __name__ == "__main__":
    x = torch.randn((10, 15, 224, 224))
    model=Model()
    print(model)
    print(model(x).shape)
