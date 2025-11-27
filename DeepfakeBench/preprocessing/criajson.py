import json
import os
import glob

#Este codigo serve para gerar o json com o caminho das imagens para que o test.py consiga ler.
#Ele substitui o rearange.py para datasets não disponibilizados pelo deepfakebench

# Configurações
JSON_NAME = "FF_FGSMV1_CNNaug"
JSON_OUTPUT_DIR = "/content/drive/MyDrive/TCCMarco/DeepfakeBench/preprocessing/dataset_json/"

# Caminhos COMPLETOS para as pastas com suas imagens
REAL_IMAGE_DIR = "/content/drive/MyDrive/TCCMarco/DeepfakeBench/datasets/rgb/FF_FGSMV1_CNNaug/Reals"
PERTURBED_FAKE_DIR = "/content/drive/MyDrive/TCCMarco/DeepfakeBench/datasets/rgb/FF_FGSMV1_CNNaug/Fakes"

# Labels que você definiu no test_config.yaml
REAL_LABEL_JSON = "FF-real"
FAKE_LABEL_JSON = "FF_FGSMV1_CNNaug"

real_files = glob.glob(os.path.join(REAL_IMAGE_DIR, "**", "*.png"), recursive=True)
fake_files = glob.glob(os.path.join(PERTURBED_FAKE_DIR, "**", "*.png"), recursive=True)

print(f"Encontradas {len(real_files)} imagens reais.")
print(f"Encontradas {len(fake_files)} imagens fake perturbadas.")

if len(real_files) > 0:
    print(f"Exemplo de caminho real: {real_files[0]}")
if len(fake_files) > 0:
    print(f"Exemplo de caminho fake: {fake_files[0]}")


data_dict = {
    JSON_NAME: {
        "Reals": {"test": {}},
        "Fakes": {"test": {}}
    }
}


real_videos = {}
for img_path in real_files:
    
    parent_folder = os.path.basename(os.path.dirname(img_path))
    
    
    if os.path.dirname(img_path) == REAL_IMAGE_DIR:
        
        video_name = os.path.splitext(os.path.basename(img_path))[0]
        real_videos[video_name] = [img_path]
    else:
        
        if parent_folder not in real_videos:
            real_videos[parent_folder] = []
        real_videos[parent_folder].append(img_path)


fake_videos = {}
for img_path in fake_files:
    
    parent_folder = os.path.basename(os.path.dirname(img_path))
    
    if parent_folder not in fake_videos:
        fake_videos[parent_folder] = []
    fake_videos[parent_folder].append(img_path)


for video_name, frames in real_videos.items():
    frames.sort() 
    data_dict[JSON_NAME]["Reals"]["test"][video_name] = {
        "label": REAL_LABEL_JSON,
        "frames": frames
    }


for video_name, frames in fake_videos.items():
    frames.sort()  
    data_dict[JSON_NAME]["Fakes"]["test"][video_name] = {
        "label": FAKE_LABEL_JSON,
        "frames": frames
    }

print(f"\nTotal de vídeos reais: {len(data_dict[JSON_NAME]['Reals']['test'])}")
print(f"Total de vídeos fake: {len(data_dict[JSON_NAME]['Fakes']['test'])}")


if len(fake_videos) > 0:
    example_video = list(fake_videos.keys())[0]
    print(f"\nExemplo de vídeo fake: '{example_video}'")
    print(f"Número de frames: {len(fake_videos[example_video])}")


os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)


output_path = os.path.join(JSON_OUTPUT_DIR, f"{JSON_NAME}.json")
with open(output_path, 'w') as f:
    json.dump(data_dict, f, indent=4)

print(f"\nJSON salvo com sucesso em: {output_path}")