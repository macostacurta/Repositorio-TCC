"""
Implementação dos modelos Meso4 e MesoInception4
Baseado no DeepFakeBench framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Meso4(nn.Module):
    """
    Meso4 model para detecção de deepfakes
    """
    def __init__(self, num_classes=2, inc=3):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(inc, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, self.num_classes)

    def features(self, input):
        x = self.conv1(input)  # (8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (16, 8, 8)
        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        
        return x
    
    def classifier(self, feature):
        out = self.dropout(feature)
        out = self.fc1(out)  # (Batch, 16)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out


class MesoInception4(nn.Module):
    """
    MesoInception4 model para detecção de deepfakes
    Usa blocos Inception customizados
    """
    def __init__(self, num_classes=2, inc=3):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        
        # InceptionLayer1
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)

        # InceptionLayer2
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)

        # Normal Layers
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16*8*8, 16)
        self.fc2 = nn.Linear(16, self.num_classes)

    def InceptionLayer1(self, input):
        x1 = self.Incption1_conv1(input)
        x2 = self.Incption1_conv2_1(input)
        x2 = self.Incption1_conv2_2(x2)
        x3 = self.Incption1_conv3_1(input)
        x3 = self.Incption1_conv3_2(x3)
        x4 = self.Incption1_conv4_1(input)
        x4 = self.Incption1_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        y = self.maxpooling1(y)
        return y

    def InceptionLayer2(self, input):
        x1 = self.Incption2_conv1(input)
        x2 = self.Incption2_conv2_1(input)
        x2 = self.Incption2_conv2_2(x2)
        x3 = self.Incption2_conv3_1(input)
        x3 = self.Incption2_conv3_2(x3)
        x4 = self.Incption2_conv4_1(input)
        x4 = self.Incption2_conv4_2(x4)
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        y = self.maxpooling1(y)
        return y

    def features(self, input):
        x = self.InceptionLayer1(input)  # (Batch, 11, 128, 128)
        x = self.InceptionLayer2(x)  # (Batch, 12, 64, 64)

        x = self.conv1(x)  # (Batch, 16, 64, 64)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (Batch, 16, 32, 32)

        x = self.conv2(x)  # (Batch, 16, 32, 32)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)  # (Batch, 16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        return x
    
    def classifier(self, feature):
        out = self.dropout(feature)
        out = self.fc1(out)  # (Batch, 16)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out


def load_meso_model(model_type, weights_path=None):
    """
    Carrega modelo Meso4 ou MesoInception4
    
    Args:
        model_type: 'meso4' ou 'meso4inception'
        weights_path: Caminho para os pesos (opcional)
    
    Returns:
        Modelo carregado
    """
    if model_type == 'meso4':
        model = Meso4(num_classes=2, inc=3)
    elif model_type == 'meso4inception':
        model = MesoInception4(num_classes=2, inc=3)
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    # Carrega pesos se fornecidos
    if weights_path:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Extrai state_dict do checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove prefixo "backbone." se existir
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
    
    model.eval()
    return model