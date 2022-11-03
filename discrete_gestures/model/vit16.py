import torch
import torch.nn as nn
import torchvision
import timm

class Model(nn.Module):
    def __init__(self,output=9):

        super(Model, self).__init__()

        self.m = timm.create_model('vit_base_patch16_224_in21k', pretrained=True,num_classes=output)
        #self.m.features[0]=nn.Conv2d(in_channels=15,out_channels=64,kernel_size=(3,3),stride=(1, 1), padding=(1, 1))
        self.m.patch_embed.proj=nn.Conv2d(in_channels=5,out_channels=768,kernel_size=(16,16),stride=(16, 16))

        #self.m=torchvision.models.vgg16_bn()
        #self.m.features[0]=nn.Conv2d(in_channels=15,out_channels=64,kernel_size=(3,3),stride=(1, 1), padding=(1, 1))
        #self.m.classifier[6]=nn.Linear(in_features=4096,out_features=output,bias=True)

    def forward(self,x):

        out=self.m(x)

        return out

if __name__ == "__main__":
    x = torch.randn((10, 15, 224, 224))
    model=Model()
    print(model)
    print(model(x).shape)
