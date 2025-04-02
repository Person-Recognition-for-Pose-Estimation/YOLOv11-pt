import torch
import torch.nn as nn
import torchvision.models as models
from .nn import YOLO, yolo_v11_n

class ResNetYOLOAdapter(nn.Module):
    def __init__(self, backbone_channels=2048):
        super().__init__()
        
        # Adapter network to convert ResNet features to YOLO input format
        self.adapter = nn.Sequential(
            # Initial channel reduction
            nn.Conv2d(backbone_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Upsample to match YOLO input size
            nn.Upsample(size=(640, 640), mode='bilinear', align_corners=True),
            
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
        # Normalize features
        x = x - x.mean(dim=(2, 3), keepdim=True)
        x = x / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
        
        # Convert features to RGB-like format
        x = self.adapter(x)
        
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
        
        # Load pretrained weights
        if pretrained_yolo_path:
            print(f"Loading pretrained YOLO weights from {pretrained_yolo_path}")
            state_dict = torch.load(pretrained_yolo_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Load all weights first
            self.yolo.load_state_dict(state_dict, strict=True)
            
            # Now modify the classification head for single class
            width = [3, 16, 32, 64, 128, 256]  # Default for yolo_v11_n
            filters = (width[3], width[4], width[5])
            
            # Create new classification heads with 1 class
            new_cls_heads = nn.ModuleList()
            for x in filters:
                # Keep the feature extraction part of cls head
                old_cls_head = self.yolo.head.cls[len(new_cls_heads)]
                new_cls_head = nn.Sequential(
                    # Copy first 4 layers that extract features
                    *list(old_cls_head[:-1]),
                    # New final layer for single class
                    nn.Conv2d(x, out_channels=1, kernel_size=1)
                )
                
                # Initialize the new classification layer
                new_cls_head[-1].weight.data.normal_(0, 0.01)
                new_cls_head[-1].bias.data.fill_(-4.595)  # -log((1 - 0.01) / 0.01)
                
                new_cls_heads.append(new_cls_head)
            
            # Replace classification heads
            self.yolo.head.cls = new_cls_heads
            self.yolo.head.nc = 1
            self.yolo.head.no = 1 + self.yolo.head.ch * 4  # Update output size
        
        # Freeze only the ResNet backbone
        for param in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = False
    
    def forward(self, x):
        # ResNet backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Convert ResNet features to YOLO input format
        x = self.adapter(x)
        
        # Pass through YOLO model
        return self.yolo(x)