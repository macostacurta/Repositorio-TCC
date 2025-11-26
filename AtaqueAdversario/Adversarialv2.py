import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Certifique-se que estes arquivos estão na mesma pasta ou no PATH
from deepfakebench_meso_models import load_meso_model
from deepfakebench_resnet34 import load_deepfakebench_resnet34


def load_detector(weights_path, model_type):
    """
    Carrega apenas os detectores solicitados: ResNet34 (DeepFakeBench) ou MesoInception4.
    """
    print(f" Carregando detector: {model_type}")
    print(f" Pesos: {weights_path}")
    
    # 1. MesoInception4
    if model_type == 'meso4inception':
        # Assume que a função load_meso_model já carrega os pesos internamente
        model = load_meso_model(model_type, weights_path)
        print("✅ Modelo MesoInception4 carregado com sucesso!")
        return model
    
    # 2. ResNet34 do DeepFakeBench
    elif model_type == 'resnet34_deepfakebench':
        # Assume que a função load_deepfakebench_resnet34 já carrega os pesos internamente
        model = load_deepfakebench_resnet34(weights_path)
        print("✅ Modelo ResNet34 (DeepFakeBench) carregado com sucesso!")
        return model
    
    else:
        raise ValueError(f"Tipo de modelo não suportado neste script: {model_type}")


def fgsm_attack(image, epsilon, data_grad):
    """ Aplica ataque FGSM na imagem. """
    sign_data_grad = data_grad.sign()
    perturbed_image = image - epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def generate_adversarial(image_path, model, epsilon=0.03):
    """ Gera imagem adversarial usando FGSM. """
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                           std=[0.5, 0.5, 0.5])
    ])
    
    
    transform_no_norm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor_no_norm = transform_no_norm(image).unsqueeze(0)
    
    image_tensor.requires_grad = True
    
    
    output = model(image_tensor)
    
    
    if isinstance(output, tuple):
        output = output[0]
    
    
    predicted_class = output.max(1)[1].item()
    target = torch.tensor([1 - predicted_class])  
    
    
    loss = nn.CrossEntropyLoss()(output, target)
    
   
    model.zero_grad()
    loss.backward()
    
    
    data_grad = image_tensor.grad.data
    
    
    perturbed_image = fgsm_attack(image_tensor_no_norm, epsilon, data_grad)
    
    # Converte para numpy
    perturbed_np = perturbed_image.squeeze().detach().cpu().numpy()
    perturbed_np = np.transpose(perturbed_np, (1, 2, 0))
    perturbed_np = (perturbed_np * 255).astype(np.uint8)
    
    return perturbed_np


def process_folder(input_dir, output_dir, prefix, model, epsilon=0.03):
    """ Processa pastas de vídeos. """
    if not input_dir: return 0, 0

    try:
        video_folders = [f for f in os.listdir(input_dir) 
                         if os.path.isdir(os.path.join(input_dir, f))]
    except FileNotFoundError:
        print(f"Diretório não encontrado: {input_dir}")
        return 0, 0

    video_folders.sort()
    
    print(f"\n Processando {len(video_folders)} vídeos de {input_dir}")
    
    processed = 0
    errors = 0
    
    for video_folder in video_folders:
        try:
            src_folder = os.path.join(input_dir, video_folder)
            new_folder_name = f"{prefix}{video_folder}_ADV"
            dst_folder = os.path.join(output_dir, new_folder_name)
            
            os.makedirs(dst_folder, exist_ok=True)
            
            frames = [f for f in os.listdir(src_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not frames:
                print(f"ERRO, {video_folder}: Nenhum frame encontrado")
                errors += 1
                continue
            
            for frame in frames:
                frame_path = os.path.join(src_folder, frame)
                adv_image = generate_adversarial(frame_path, model, epsilon)
                adv_path = os.path.join(dst_folder, frame)
                Image.fromarray(adv_image).save(adv_path)
            
            processed += 1
            print(f"✓ [{processed}/{len(video_folders)}] {video_folder} → {new_folder_name} ({len(frames)} frames)")
        
        except Exception as e:
            print(f"ERRO em {video_folder}: {str(e)}")
            errors += 1
    
    print(f"\n Resumo: {processed} processados, {errors} erros")
    return processed, errors


def main():
    print("=" * 70)
    print("GERADOR DE ATAQUES ADVERSARIAIS FGSM (Módulos Selecionados)")
    print("=" * 70)
    
    # Configuração do detector
    print("\nCONFIGURAÇÃO DO DETECTOR")
    weights_path = input("Caminho dos pesos do detector (.pth/.pt): ").strip()
    
    if not os.path.exists(weights_path):
        print(f"Erro: Arquivo de pesos não encontrado: {weights_path}")
        return
    
    print("\nSelecione o modelo:")
    print("1 - ResNet34 (DeepFakeBench)")
    print("2 - MesoInception4")
    
    model_choice = input("Escolha (1 ou 2): ").strip()
    
    
    model_types = {
        '1': 'resnet34_deepfakebench',
        '2': 'meso4inception'
    }
    
    model_type = model_types.get(model_choice)
    
    if not model_type:
        print("Opção inválida! Escolha 1 ou 2.")
        return
    
    # Entradas dos diretórios
    print("\nDIRETÓRIOS")
    face2face_dir = input("Diretório Face2Face (enter para pular): ").strip()
    faceswap_dir = input("Diretório FaceSwap (enter para pular): ").strip()
    output_dir = input("Diretório de saída: ").strip()
    
    if not output_dir:
        print("Erro: Diretório de saída é obrigatório.")
        return

    # Epsilon
    epsilon_input = input("\nEpsilon (0.01-0.1, padrão=0.03): ").strip()
    epsilon = float(epsilon_input) if epsilon_input else 0.03
    
    if not face2face_dir and not faceswap_dir:
        print("Erro: Nenhum diretório de entrada válido fornecido!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    print("\n" + "=" * 70)
    try:
        model = load_detector(weights_path, model_type)
        
        model.eval() 
    except Exception as e:
        print(f"Erro Crítico ao carregar modelo: {str(e)}")
        return
    
    print(f"Epsilon configurado: {epsilon}")
    print("=" * 70)
    
    total_processed = 0
    total_errors = 0
    
    # Processa Face2Face
    if face2face_dir:
        print("\n PROCESSANDO FACE2FACE")
        p, e = process_folder(face2face_dir, output_dir, "F2F_", model, epsilon)
        total_processed += p
        total_errors += e
    
    # Processa FaceSwap
    if faceswap_dir:
        print("\n PROCESSANDO FACESWAP")
        p, e = process_folder(faceswap_dir, output_dir, "FS_", model, epsilon)
        total_processed += p
        total_errors += e
    
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUÍDO!")
    print(f"Total: {total_processed} vídeos processados")
    print(f"Erros: {total_errors}")
    print(f"Saída: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()