import os
import gdown  # pip install gdown

# Link da pasta compartilhada no Google Drive
FOLDER_URL = "https://drive.google.com/drive/folders/1eAYlrz_wPWP6vTjSlCB1eHYrDo7pQKfo?usp=sharing"
TARGET_DIR = "data"  # pasta local onde o dataset será baixado

# Criar a pasta data se não existir
os.makedirs(TARGET_DIR, exist_ok=True)

# Baixar a pasta inteira do Drive (preservando a estrutura)
if not os.path.exists(os.path.join(TARGET_DIR, "train")):
    print("Baixando dataset do Google Drive...")
    gdown.download_folder(FOLDER_URL, output=TARGET_DIR, quiet=False, use_cookies=False)
    print("Download concluído. Dataset pronto em ./data")
else:
    print("Dataset já existe em ./data — pulando download.")