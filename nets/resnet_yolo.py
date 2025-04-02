import torch
import torch.nn as nn
import torchvision.models as models
from .nn import YOLO, yolo_v11_n
import time

class ResNetYOLOAdapter(nn.Module):
    def __init__(self, backbone_channels=2048):
        super().__init__()
        
        # Adapter network to convert ResNet features to YOLO input format
        self.adapter = nn.Sequential(
            # Initial channel reduction
            nn.Conv2d(backbone_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Upsample to match YOLO input size (will be resized based on input)
            
            # Progressive channel reduction with spatial processing
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            # Final adaptation to match YOLO input (3 channels)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.SiLU()
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get input spatial dimensions
        _, _, H, W = x.shape
        
        # Normalize features
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Process through adapter network
        x = self.adapter[0:3](x)  # Initial reduction
        
        # Upsample to YOLO input size (640x640)
        x = nn.functional.interpolate(x, size=(640, 640), mode='bilinear', align_corners=True)
        
        # Further processing
        x = self.adapter[3:](x)
        
        # Ensure output is in expected range [0, 1]
        x = torch.sigmoid(x)
        return x

class ResNetYOLO(nn.Module):
    def __init__(self, pretrained_yolo_path=None):
        super().__init__()
        
        # Create ResNet backbone
        resnet = models.resnet50(pretrained=True)
        
        # Extract ResNet layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Create adapter
        self.adapter = ResNetYOLOAdapter(backbone_channels=2048)  # ResNet50 outputs 2048 channels
        
        # First create YOLO model with original 80 classes to load weights properly
        self.yolo = yolo_v11_n(num_classes=80)
        
        # Expose YOLO attributes needed by loss function
        self.head = self.yolo.head
        self.stride = self.yolo.stride
        
        # Load pretrained weights
        if pretrained_yolo_path:
            print(f"Loading pretrained YOLO weights from {pretrained_yolo_path}")
            state_dict = torch.load(pretrained_yolo_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
        
        # Freeze only the ResNet backbone
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = False

    def forward(self, x):
        with torch.no_grad(), torch.cuda.amp.autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            torch.cuda.empty_cache()
            
            x = self.layer2(x)
            torch.cuda.empty_cache()
            
            x = self.layer3(x)
            torch.cuda.empty_cache()
            
            x = self.layer4(x)
            torch.cuda.empty_cache()
            
            x = self.adapter(x)
            torch.cuda.empty_cache()
        
        x = x.detach().contiguous()
        
        return self.yolo(x)
        