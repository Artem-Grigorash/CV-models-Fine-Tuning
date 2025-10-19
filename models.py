import torch
from torchvision.models import vit_b_16, resnet152, swin_s

VIT_MODEL = vit_b_16()
SWIN_MODEL = swin_s()
RESNET_MODEL = resnet152()

VIT_MODEL.heads.head = torch.nn.Linear(in_features=VIT_MODEL.heads.head.in_features, out_features=2)
SWIN_MODEL.head = torch.nn.Linear(in_features=SWIN_MODEL.head.in_features, out_features=2)
RESNET_MODEL.fc = torch.nn.Linear(in_features=RESNET_MODEL.fc.in_features, out_features=2)

# list with pre-trained models
models = [RESNET_MODEL, SWIN_MODEL, VIT_MODEL]


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
