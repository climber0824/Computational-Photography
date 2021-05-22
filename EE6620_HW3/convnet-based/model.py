import torch
import torch.nn as nn

'''
    Example model construction in pytorch
'''
class example_resblock(nn.Module):
    def __init__(self, bias=True, act=nn.ReLU(True)):
        super(example_resblock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(16, 16, 3, padding=1), bias=bias)
        modules.append(act)
        modules.append(nn.Conv2d(16, 16, 3, padding=1), bias=bias)
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class resblock(nn.Module):
    def __init__(self, nFeat, kernel_size=3, bias=True, act=nn.ReLU(True)):
        super(resblock, self).__init__()
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        modules.append(act)
        modules.append(nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias))
        self.body = nn.Sequential(*modules)
    
    def forward(self, x):
        out = self.body(x)
        out += x
        return out

class upsampler(nn.Module):
    def __init__(self, scale=2, nFeat, act=nn.ReLU(True)):
        super(upsampler, self).__init__()
        #===== write your model definition here =====#
        
        super(upsampler, self).__init__()
        # add the definition of layer here
        modules = []
        modules.append(nn.Conv2d(nFeat, nFeat * scale, 3, padding=1))
        modules.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*modules)
        
    def forward(self, x):
        #===== write your dataflow here =====#
        out = self.body(x)
        return out

class ZebraSRNet(nn.Module):
    def __init__(self, nFeat=64, kernel_size=3, nResBlock=8, imgChannel=3):
        super(ZebraSRNet, self).__init__()
        #===== write your model definition here using 'resblock' and 'upsampler' as the building blocks =====#
    
        self.module1 = nn.Conv2d(imgChannel, nFeat, kernel_size, padding=1)

        ResBlock = []
        for _ in range(nResBlock):
            ResBlock.append(ResBlock2(nFeat, kernel_size))
        self.module2 = nn.Sequential(*ResBlock)

        module3 = []
        module3.append(upsampler(SRscale, nFeat))
        module3.append(upsampler(SRscale, nFeat))
        module3.append(nn.Conv2d(nFeat, imgChannel, kernel_size, padding=1))
        self.module3 = nn.Sequential(*module3)

    def forward(self, x):
        #===== write your dataflow here =====#
        
        x = self.module1(x)
        out = self.module2(x)
        out += x
        out = self.module3(out)

        return out
