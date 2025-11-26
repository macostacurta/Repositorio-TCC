'''
# Adaptação do ResNet34 do DeepFakeBench para ataques adversariais
# Baseado no código original de: Zhiyuan Yan (zhiyuanyan@link.cuhk.edu.cn)
'''

import torch
import torch.nn as nn
import torchvision


class ResNet34(nn.Module):
    """ResNet34 do DeepFakeBench"""
    def __init__(self, num_classes=2, inc=3, mode='Original'):
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.mode = mode

        # Define layers of the backbone
        resnet = torchvision.models.resnet34(pretrained=False)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, inp):
        x = self.resnet(inp)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out


def load_deepfakebench_resnet34(weights_path, num_classes=2, inc=3, mode='Original'):
    """
    Carrega o modelo ResNet34 do DeepFakeBench com pesos pré-treinados.
    
    Args:
        weights_path: Caminho para o arquivo de pesos (.pth, .pt)
        num_classes: Número de classes (padrão: 2 para real/fake)
        inc: Número de canais de entrada (padrão: 3 para RGB)
        mode: Modo do modelo ('Original' ou 'adjust_channel')
    
    Returns:
        Modelo carregado em modo eval
    """
    print(f"🔧 Carregando ResNet34 do DeepFakeBench")
    print(f"📂 Pesos: {weights_path}")
    
    # Cria o modelo
    model = ResNet34(num_classes=num_classes, inc=inc, mode=mode)
    
    # Carrega pesos
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Trata diferentes formatos de checkpoint do DeepFakeBench
    if isinstance(checkpoint, dict):
        # Formato comum do DeepFakeBench: {'model': state_dict, ...}
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        # Outros formatos possíveis
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    
    new_state_dict = {}
    for k, v in state_dict.items():
        
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
        
        elif k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Carrega os pesos no modelo
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    print("ResNet34 do DeepFakeBench carregado")
    
    return model