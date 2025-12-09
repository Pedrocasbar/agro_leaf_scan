import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import json
from fpdf import FPDF
import io
from contextlib import redirect_stdout

st.title("Análise de Saúde da Folha 🌿")

# --- Inicializar histórico no session_state ---
if "historico" not in st.session_state:
    st.session_state["historico"] = []  # cada item: {"img": PIL.Image, "classe": str, "prob": float}

# --- Carregar modelo e classes ---
model = load_model("folha_cnn_model.h5", compile=False)

with open("classes.json", "r") as f:
    class_indices = json.load(f)

indices_to_class = {v: k for k, v in class_indices.items()}
nomes_amigaveis = {"s": "Saudável", "d": "Doente"}
class_names = [nomes_amigaveis.get(indices_to_class[i], str(i)) for i in range(len(indices_to_class))]

# --- Upload da imagem ---
uploaded_file = st.file_uploader("Envie uma imagem da folha", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if "ultimo_upload" not in st.session_state or st.session_state["ultimo_upload"] != uploaded_file.name:
        st.session_state["ultimo_upload"] = uploaded_file.name

        img = Image.open(uploaded_file).convert("RGB")

        # Pré-processamento
        img_array = img.resize((128, 128))
        img_array = tf.keras.utils.img_to_array(img_array) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predição
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        prob = prediction[0][class_idx]
        classe_nome = class_names[class_idx]

        # Salvar no histórico
        st.session_state["historico"].append({
            "img": img.copy(),
            "classe": classe_nome,
            "prob": prob
        })

        st.subheader("Última imagem:")

        # Exibe a imagem normalmente
        st.image(img, use_container_width=True)

        # Exibe o caption maior abaixo da imagem    
        st.markdown(
            f'<p style="font-size:22px; font-weight:bold;">Resultado: (Classe Detectada: {classe_nome}) | (Probabilidade: {prob:.2f})</p>',
            unsafe_allow_html=True
)
       
# --- Exibir histórico de imagens com miniaturas ---
if st.session_state["historico"]:
    st.subheader("Histórico de imagens enviadas:")
    for i, item in enumerate(st.session_state["historico"]):
        col1, col2 = st.columns([1, 3])
        with col1:
            thumbnail = item["img"].copy()
            thumbnail.thumbnail((120, 120))
            st.image(thumbnail, caption=f"Imagem {i+1}")
        with col2:
            st.write(f"**Classe:** {item['classe']}")
            st.write(f"**Probabilidade:** {item['prob']:.2f}")

# --- Função para gerar relatório PDF ---
def gerar_relatorio_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório do Modelo - Análise de Folha", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Número de classes: {len(class_names)}", ln=True)
    pdf.cell(0, 10, "Classes mapeadas:", ln=True)

    for idx, nome in enumerate(class_names):
        pdf.cell(0, 8, f"  - {idx}: {nome}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Arquitetura do modelo:", ln=True)

    # Capturar summary do modelo como string
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.summary()
    summary_str = buffer.getvalue()

    for line in summary_str.split('\n'):
        pdf.multi_cell(0, 6, line)

    # Adicionar histórico de imagens com resultados
    if st.session_state["historico"]:
        pdf.ln(5)
        pdf.cell(0, 10, "Histórico de predições:", ln=True)
        for i, item in enumerate(st.session_state["historico"]):
            pdf.cell(0, 8, f"{i+1}. Classe: {item['classe']} | Probabilidade: {item['prob']:.2f}", ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = io.BytesIO(pdf_bytes)
    return pdf_buffer

# --- Botão de visualização do relatório ---
if st.button("Visualizar Relatório do Modelo"):
    st.subheader("Relatório do Modelo:")
    st.write(f"Número de Classes: {len(class_names)}")
    st.write("Classes Mapeadas:")
    for idx, nome in enumerate(class_names):
        st.write(f"  - {idx}: {nome}")

    if st.session_state["historico"]:
        st.write("Histórico de predições:")
        for i, item in enumerate(st.session_state["historico"]):
            st.write(f"{i+1}.Imagem - Classe: {item['classe']} | Probabilidade: {item['prob']:.2f}")
    else:
        st.write("Nenhuma imagem enviada!")

# --- Botão de download do PDF ---
if st.button("Baixar Relatório em PDF"):
    pdf_buffer = gerar_relatorio_pdf()
    st.download_button(
        label="Clique aqui para baixar",
        data=pdf_buffer,
        file_name="relatorio_modelo_folha.pdf",
        mime="application/pdf"
    )
