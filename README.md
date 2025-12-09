🌿 Folha CNN – Detecção de Anomalias em Folhas de Soja
Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar folhas de soja entre Saudáveis e Doentes.

Inclui scripts para:

📥 Download automático do dataset
📁 Organização das pastas
🧠 Treinamento da CNN
🔍 Predição individual
🌐 Aplicativo Streamlit com histórico, miniaturas e exportação de PDF

Estrutura do Projeto
folha_cnn/
│── baixar_data.py        # Script para baixar e organizar o dataset do Google Drive
│── train_model.py        # Treinamento da CNN
│── predict_leaf.py       # Predição usando o modelo treinado
│── app.py                # Aplicação Streamlit
│── model.h5              # Modelo treinado (gerado após treino)
│── requirements.txt      # Bibliotecas do projeto
│── data/                 # Dataset    
├── train/                # Treino do Modelo
│   ├── d/                # 'd' para Doente
│   └── s/                # 's' para Saudavel
│
└── val/                  # Validação do Modelo
    ├── saudavel/
    └── doente/

1. Instalação - Criar e ativar o ambiente virtual:

   python -m venv venv
   .\venv\Scripts\activate
      
3. Instalar dependências:

   pip install -r requirements.txt

5. Baixar o dataset (Google Drive):

   python baixar_data.py

    Este script realiza automaticamente:
I. Baixa o arquivo data.zip do Google Drive
II. Salva em data/

    Você deve:
I.Descompacta em data/data
II.Move apenas o conteúdo interno para:

        data/train/
        data/val/
   
Apaga a pasta vazia data/data/
Após rodar, o caminho final fica assim:

        data/train/
        data/val/

5. Treinar o modelo:

   python train_model.py
   
    Isso irá:
I. carregar as imagens de data/train e data/val
II. treinar uma CNN
III. gerar o arquivo model.h5

7. Fazer predições via script:

   python predict_leaf.py caminho/da/imagem.jpg
Essa etapa garante que o medelo está apontando certo.

8. Executar o aplicativo Streamlit:

   streamlit run app.py
   
    O app permite:
✔ enviar imagens
✔ visualizar miniaturas no histórico
✔ ver porcentagem/confiança da predição
✔ gerar relatório PDF
✔ visualizar o relatório dentro do app

9. Gerar Relatório PDF:

    Dentro do app Streamlit:
I. Após enviar uma imagem
II. Clique em "Gerar Relatório"

    O arquivo é salvo automaticamente com:
✔ 20251209_084655_relatorio.pdf

10. Objetivo do Projeto:

   Criar um sistema completo capaz de:

✔ treinar uma CNN
✔ classificar imagens de folhas
✔ gerar relatório PDF
✔ manter histórico das análises
✔ rodar localmente em Streamlit

--------------------------------------------------------------------------------


📄Licença
Este projeto é de uso livre para fins acadêmicos e educacionais.

👨‍💻 Autor
Pedro Castro Barros – Projeto de TCC:
"Detecção de Anomalias em Folhas de Plantações de Soja Utilizando Redes Neurais Convolucionais".
Orientador(a): Professora Dra.  Kadidja Valéria.
