import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carregar modelo
model = load_model("folha_cnn_model.h5")

# Carregar mapeamento das classes
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# Inverter o dicionário para obter label a partir do índice
indices_to_class = {v: k for k, v in class_indices.items()}

# Dicionário para nomes amigáveis das classes
nomes_amigaveis = {
    's': 'Folha Saudável',
    'd': 'Folha com Anomalia'
}

def analisar_folha(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]  # vetor de probabilidades
    predicted_index = np.argmax(prediction)
    predicted_class = indices_to_class[predicted_index]
    confidence = prediction[predicted_index]

    print(f"Resultado: {nomes_amigaveis.get(predicted_class, predicted_class)} (probabilidade: {confidence:.2f})")

# Substitua pelo caminho da sua imagem de teste
analisar_folha("data/train/s/0.png")
