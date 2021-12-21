import sys, logging
sys.path.append('../')

import torch 
import torchvision.models as models
from tool.trainer_cls import MyModelTrainerCLS

# device is cuda:0 if possible
#trainer is cls
def generate_cls_model(
    model_name: str,
    num_classes: int = 10,
):
    if model_name == 'resnet18':
        print('NOT default setting for resnet18 !!!!!')
        from models.preact_resnet import PreActResNet18
        net = PreActResNet18(num_classes= num_classes)
    elif model_name == 'alexnet':
        net = models.alexnet(num_classes= num_classes)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes= num_classes)
    elif model_name == 'squeezenet1_0':
        net = models.squeezenet1_0(num_classes= num_classes)
    elif model_name == 'densenet161':
        net = models.densenet161(num_classes= num_classes)
    elif model_name == 'inception_v3':
        net = models.inception_v3(num_classes= num_classes)
    elif model_name == 'googlenet':
        net = models.googlenet(num_classes= num_classes)
    elif model_name == 'shufflenet_v2_x1_0':
        net = models.shufflenet_v2_x1_0(num_classes= num_classes)
    elif model_name == 'mobilenet_v2':
        net = models.mobilenet_v2(num_classes= num_classes)
    elif model_name == 'resnext50_32x4d':
        net = models.resnext50_32x4d(num_classes= num_classes)
    elif model_name == 'wide_resnet50_2':
        net = models.wide_resnet50_2(num_classes= num_classes)
    elif model_name == 'mnasnet1_0':
        net = models.mnasnet1_0(num_classes= num_classes)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    trainer = MyModelTrainerCLS(
        model=net,
    )

    return net, device, trainer