import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Carrega o modelo treinado
model = load_model("folha_cnn_model.h5")

# Interface do app
st.title("Análise de Saúde da Folha 🌿")

uploaded_file = st.file_uploader("Envie uma imagem da folha", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")  # ⬅️ Aqui converte para RGB
    st.image(img, caption='Imagem enviada', use_container_width=True)

    # Pré-processa a imagem
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faz a predição
    prediction = model.predict(img_array)
    class_names = ['Doente', 'Saudável']
    class_idx = np.argmax(prediction[0])
    prob = prediction[0][class_idx]

    st.subheader("Resultado:")
    st.write(f"A folha está **{class_names[class_idx]}** com probabilidade de **{prob:.2f}**.")