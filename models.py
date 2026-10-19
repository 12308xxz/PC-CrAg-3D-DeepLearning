"""3D CT models for within-disease qualitative CrAg classification.

Architectures are based on torchvision.models.video.r3d_18.
By default, pretrained=True loads the Kinetics-400 pretrained initialization.
"""
import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SEBasicBlockWrapper(nn.Module):
    def __init__(self, block, reduction=16):
        super().__init__()
        self.block = block
        # torchvision VideoResNet BasicBlock.conv2 is Sequential(conv, bn),
        # not a bare Conv3d. Using block.conv2.out_channels raises AttributeError.
        out_channels = block.conv2[0].out_channels
        self.se = SEBlock3D(out_channels, reduction=reduction)

    def forward(self, x):
        return self.se(self.block(x))


class SmallStem3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.stem(x)


def convert_first_conv_to_1ch(conv3d_layer):
    old_conv = conv3d_layer
    new_conv = nn.Conv3d(
        1, old_conv.out_channels, old_conv.kernel_size,
        old_conv.stride, old_conv.padding, bias=False,
    )
    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    return new_conv


def add_se_to_layers(model, stages=("layer3", "layer4"), reduction=16):
    for layer_name in stages:
        old_layer = getattr(model, layer_name)
        blocks = [SEBasicBlockWrapper(block, reduction=reduction) for block in old_layer]
        setattr(model, layer_name, nn.Sequential(*blocks))
    return model


def _weights(pretrained):
    return R3D_18_Weights.KINETICS400_V1 if pretrained else None


def create_model_3d_baseline(num_classes=2, dropout=0.3, pretrained=True):
    model = r3d_18(weights=_weights(pretrained))
    model.stem[0] = convert_first_conv_to_1ch(model.stem[0])
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def create_model_3d_optimized(num_classes=2, dropout=0.5, pretrained=True):
    model = r3d_18(weights=_weights(pretrained))
    model.stem = SmallStem3D(in_channels=1, out_channels=64)
    # VideoResNet has no separate maxpool in its forward. Retain this Identity
    # attribute used for model interface compatibility.
    model.maxpool = nn.Identity()
    model = add_se_to_layers(model, stages=("layer3", "layer4"))
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = r3d_18(weights=_weights(pretrained))
        model.stem = SmallStem3D(in_channels=1, out_channels=64)
        model.maxpool = nn.Identity()
        model = add_se_to_layers(model, stages=("layer3", "layer4"))
        self.stem = model.stem
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class Model3DPlusClinical(nn.Module):
    def __init__(self, num_classes=2, clinical_dim=8, pretrained=True):
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained)
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Dropout(0.3),
        )
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 + 32, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, image, clinical):
        img_feat = self.image_encoder(image)
        clin_feat = self.clinical_branch(clinical)
        return self.fusion_head(torch.cat([img_feat, clin_feat], dim=1))


def create_model_3d_clinical(num_classes=2, clinical_dim=8, pretrained=True):
    return Model3DPlusClinical(num_classes=num_classes, clinical_dim=clinical_dim, pretrained=pretrained)
