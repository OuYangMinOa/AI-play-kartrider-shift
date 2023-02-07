from   numpy              import *
import torchvision.models as models
import torch.nn           as nn



def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb

def conv_block(in_s, out_s, pool = False):
    layers = [nn.Conv2d(in_s, out_s, 3 ,padding=1),
              nn.BatchNorm2d(out_s),
              nn.ReLU()
             ]
    if (pool): layers.append( nn.MaxPool2d(2))
        
    return nn.Sequential(*layers)

    
class MyResNet(nn.Module):
    def __init__(self,in_channels=3, out_classes=6,fiugure_size=array([270,480])):
        super().__init__()
        self.fiugure_size = fiugure_size
        self.con1 = conv_block(in_channels,64)  # 64*fiugure_size
        self.con2 = conv_block(64,128, True)    # 128*16*16
        self.res1 = nn.Sequential( conv_block(128,128), conv_block(128,128) )  # 128*16*16
        
        
        self.con3 = conv_block(128,256 , True)   # 256*8*8 
        self.con4 = conv_block(256,256 , True)   # 256*4*4
        self.res2 = nn.Sequential( conv_block(256,256), conv_block(256,256) )  # 256*4*4 
        
        self.con5 = conv_block(256,128 , True)   # 512*8*8 
        self.con6 = conv_block(128,128 , True)   # 512*4*4
        self.res3 = nn.Sequential( conv_block(128,128), conv_block(128,128) )  # 32*4*4 
        
        self.con7 = conv_block(128,64 , True)   # 512*8*8 
        self.con8 = conv_block(64,64  , True)   # 512*4*4
        self.res4 = nn.Sequential( conv_block(64,64), conv_block(64,64) )  # 32*4*4 
        
        fiugure_size = fiugure_size//(2**7)
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*fiugure_size[0] * fiugure_size[1], 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, out_classes)
        )
        
    def forward(self,x):
        x = self.con1(x)
        x = self.con2(x)
        x = self.res1(x) + x
        
        x = self.con3(x)
        x = self.con4(x)
        x = self.res2(x) + x
        
        x = self.con5(x)
        x = self.con6(x)
        x = self.res3(x) + x

        x = self.con7(x)
        x = self.con8(x)
        x = self.res4(x) + x
        return self.classifier(x)



def model_50():
    model = nn.Sequential(
    nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),  # 32 x 480 x 270
    nn.ReLU(),
    nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), # 64 x 480 x 270
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 256 x 240 x 135
    
    nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), # 128 x 16 x 270
    nn.ReLU(),
    nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1), # 128 x 16 x 270
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 128 x 120 x 67
    
    nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 256 x 60 x 33
    
    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 256 x 30 x 16
    
    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 512 x 15 x 8
    
    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 512 x 7 x 4
    
    nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 1024 x 3 x 2
    
    nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.Conv2d(1024,1024,kernel_size=3,stride=1,padding=1), # 256 x 16 x 67
    nn.ReLU(),
    nn.MaxPool2d(2,2), # 1024 x 3 x 2
    
    nn.Flatten(),
    nn.Linear(1024, 1024), # 2048
    nn.ReLU(),
    nn.Linear(1024,512),
    nn.ReLU(),
    nn.Linear(512,6))
    return model


def mnasnet():
    model = nn.Sequential(
        models.mnasnet1_0(pretrained=True),
        nn.Flatten(),
        nn.Linear(1000, 512),
        nn.ReLUI(),
        nn.Linear(512, 6)
        )
    return model



def wide_resnet101_2():
    return nn.Sequential(
        models.wide_resnet101_2(pretrained=False),
        # nn.Flatten(),
        # nn.Linear(1000, 512),
        # nn.Linear(512, 6)
        )


if __name__ == '__main__':
    import torch
    model = wide_resnet101_2()
    test_torch = torch.randn(1,3,224,224)
    print(model(test_torch).size)
    print(
        model.eval())

    model_size(model)