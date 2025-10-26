import torch
import timm
from torchvision.models import (
    vit_b_16, resnet152, swin_s,
    ViT_B_16_Weights, ResNet152_Weights, Swin_S_Weights,
)

# Use pretrained ImageNet weights to improve convergence on small datasets
VIT_MODEL = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
SWIN_MODEL = swin_s(weights=Swin_S_Weights.DEFAULT)
RESNET_MODEL = resnet152(weights=ResNet152_Weights.DEFAULT)
DINO_MODEL = timm.create_model("vit_base_patch16_dinov3", pretrained=True)

# У модели нет .head, только num_features
in_features = DINO_MODEL.num_features

# Добавляем свою классификационную голову
DINO_MODEL.head = torch.nn.Sequential(
    torch.nn.Linear(in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(256, 2),
)


VIT_MODEL.heads = torch.nn.Sequential(
    torch.nn.Linear(in_features=VIT_MODEL.heads.head.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)
SWIN_MODEL.head = torch.nn.Sequential(
    torch.nn.Linear(in_features=SWIN_MODEL.head.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)
RESNET_MODEL.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=RESNET_MODEL.fc.in_features, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(in_features=256, out_features=2),
)

# list with pre-trained models
models = [DINO_MODEL]


def get_model_parameters(_model):
    return sum(p.numel() for p in _model.parameters() if p.requires_grad)


def get_model_size(_model):
    param_size = 0
    for param in _model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in _model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 3
    return 'model size: {:.3f}GB'.format(size_all_mb)


for model in models:
    model_name = model.__class__.__name__
    print(model_name, get_model_parameters(model))
    print(model_name, get_model_size(model))
