import timm
import torch


def load_model_from_timm(model_name, device, device_ids, pretrained, num_classes):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    return model


def load_model_from_local_file():
    pass
