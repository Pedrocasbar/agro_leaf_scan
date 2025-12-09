import tensorflow as tf
from keras.models import load_model
import numpy as np
import json

# carregar modelo
model = load_model("folha_cnn_model.h5", compile=False)

# carregar classes
with open("classes.json", "r") as f:
    class_indices = json.load(f)

indices_to_class = {v: k for k, v in class_indices.items()}
nomes_amigaveis = {"s": "Saudável", "d": "Doente"}

def analisar_folha(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = indices_to_class[predicted_index]
    confidence = prediction[predicted_index]

    print(f"Resultado: {nomes_amigaveis.get(predicted_class, predicted_class)} (probabilidade: {confidence:.2f})")

# teste
analisar_folha(r"C:\Users\PEDRO\Downloads\folha_cnn\data\train\d\img_doente_466.jpg")
