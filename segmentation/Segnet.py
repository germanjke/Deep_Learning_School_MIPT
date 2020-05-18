import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.conv00 = nn.Conv2d(3,64, kernel_size=3, padding=1 )
        self.bn00 = nn.BatchNorm2d(64)
        self.conv01 = nn.Conv2d(64,64, kernel_size=3, padding=1 )
        self.bn01 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)    
        
        self.conv10 = nn.Conv2d(64,128, kernel_size=3, padding=1 )
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128,128, kernel_size=3, padding=1 )
        self.bn11 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) 
        
        self.conv20 = nn.Conv2d(128,256, kernel_size=3, padding=1 )
        self.bn20 = nn.BatchNorm2d(256)
        self.conv21 = nn.Conv2d(256,256, kernel_size=3, padding=1 )
        self.bn21 = nn.BatchNorm2d(256)
        self.conv22 = nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) 
        
        self.conv30 = nn.Conv2d(256,512, kernel_size=3, padding=1 )
        self.bn30 = nn.BatchNorm2d(512)
        self.conv31 = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn31 = nn.BatchNorm2d(512)
        self.conv32 = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn32 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv40 = nn.Conv2d(512,512,kernel_size=3,padding=1) 
        self.bn40 = nn.BatchNorm2d(512)
        self.conv41 = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn42 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # bottleneck
        #self.bottleneck_conv = 

        # decoder (upsampling)
        self.upsample4 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
        self.conv40d = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn40d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn41d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512,512, kernel_size=3, padding=1 )
        self.bn42d = nn.BatchNorm2d(512)
        
        self.upsample3 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
        self.conv30d = nn.Conv2d(512,256, kernel_size=3, padding=1 )
        self.bn30d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256,256, kernel_size=3, padding=1 )
        self.bn31d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256,256, kernel_size=3, padding=1 )
        self.bn32d = nn.BatchNorm2d(256)
       
        self.upsample2 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
        self.conv20d = nn.Conv2d(256,128, kernel_size=3, padding=1 )
        self.bn20d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128,128, kernel_size=3, padding=1 )
        self.bn21d = nn.BatchNorm2d(128)
        self.conv22d = nn.Conv2d(128,128, kernel_size=3, padding=1 )
        self.bn22d = nn.BatchNorm2d(128)

        self.upsample1 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
        self.conv10d = nn.Conv2d(128,64, kernel_size=3, padding=1 )
        self.bn10d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64,64, kernel_size=3, padding=1 )
        self.bn11d = nn.BatchNorm2d(64)

        self.upsample0 = nn.MaxUnpool2d(kernel_size=2, stride=2) 
        self.conv00d = nn.Conv2d(64,64, kernel_size=3, padding=1 )
        self.bn00d = nn.BatchNorm2d(64)
        self.conv01d = nn.Conv2d(64,1, kernel_size=3, padding=1 )
        self.bn01d = nn.BatchNorm2d(1)


    def forward(self, x):
        # encoder
        x = F.relu(self.bn00(self.conv00(x)))
        x = F.relu(self.bn01(self.conv01(x)))
        x, idx0 = self.pool0(x)
        
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x, idx1 = self.pool1(x)
        
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x, idx2 = self.pool2(x)

        x = F.relu(self.bn30(self.conv30(x)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x, idx3 = self.pool3(x)

        x = F.relu(self.bn40(self.conv40(x)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x, idx4 = self.pool4(x)
        
        # bottleneck
        #b = 

        # decoder
        x = self.upsample4(x,idx4)
        x = F.relu(self.bn40d(self.conv40d(x)))
        x = F.relu(self.bn41d(self.conv41d(x)))
        x = F.relu(self.bn42d(self.conv42d(x)))
        
        x = self.upsample3(x,idx3)
        x = F.relu(self.bn30d(self.conv30d(x)))
        x = F.relu(self.bn31d(self.conv31d(x)))
        x = F.relu(self.bn32d(self.conv32d(x)))

        x = self.upsample2(x,idx2)
        x = F.relu(self.bn20d(self.conv20d(x)))
        x = F.relu(self.bn21d(self.conv21d(x)))
        x = F.relu(self.bn22d(self.conv22d(x)))

        x = self.upsample1(x,idx1)
        x = F.relu(self.bn10d(self.conv10d(x)))
        x = F.relu(self.bn11d(self.conv11d(x)))

        x = self.upsample0(x,idx0)
        x = F.relu(self.bn00d(self.conv00d(x)))
        x = self.bn01d(self.conv01d(x)) # no activation

        return x
