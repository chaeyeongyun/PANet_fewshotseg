import torch
import torch.nn as nn

class MVGG16(nn.Module):
    '''modified vgg16 used in PANet'''
    def __init__(self, init_weights=True):
        super(MVGG16, self).__init__()
        self.in_channels = 3
        # there are out_channels and M(maxpool) in self.vgg_cfg 
        self.vgg_cfg = [64, 'M', 128, 'M', 256, 'M', 512, 'Ms1', 512]
        
        self.conv_layers = self._make_layers(self.vgg_cfg)
       
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for i, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            elif v == 'Ms1':
                layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
            elif i==(len(cfg)-1):
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, dilation=2, padding=2)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(v), 
                            nn.ReLU()]
                in_channels = v
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv_layers(x)
        return output
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.conv_layers(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            