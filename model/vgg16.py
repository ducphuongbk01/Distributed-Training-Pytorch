import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, num_layers=2):
        super(ConvBlock, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)))
        model.append(nn.ReLU(True))
        for _ in range(num_layers-1):
            model.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1)))
            model.append(nn.ReLU(True))
        model.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
        self.conv = nn.Sequential(*model)
    def forward(self,x):
        return self.conv(x)


class VGG16(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_weights=True):
        super(VGG16, self).__init__()
        # 2D feature extractor
        self.backbone = nn.Sequential(ConvBlock(3, 64),
                                      ConvBlock(64, 128),
                                      ConvBlock(128, 256, num_layers=3),
                                      ConvBlock(256, 512, num_layers=3),
                                      ConvBlock(512, 512, num_layers=3))
        # self.conv1 = ConvBlock(3, 64)
        # self.conv2 = ConvBlock(64, 128)
        # self.conv3 = ConvBlock(128, 256, num_layers=3)
        # self.conv4 = ConvBlock(256, 512, num_layers=3)
        # self.conv5 = ConvBlock(512, 512, num_layers=3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7,7))

        # 1D feature process
        self.linear1 = nn.Linear(512*7*7, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # Predict layer
        self.linear3 = nn.Linear(4096, out_channels)
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        # 2D feature extractor
        x = self.backbone(x)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.avgpool(x)
        
        # 1D feature process
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Get output
        x = self.linear3(x)
        return x
