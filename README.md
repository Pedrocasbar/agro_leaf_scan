TCC: Análise de Saúde de Folhas com Redes Neurais Convolucionais (CNN)

----- Visão Geral do Projeto
Este projeto de Trabalho de Conclusão de Curso (TCC) implementa um sistema automatizado para diagnosticar a saúde de folhas de plantas através de imagens. O objetivo principal é classificar uma folha como Saudável ou Com Anomalia (Doente).

O sistema utiliza uma Rede Neural Convolucional (CNN), desenvolvida com as bibliotecas TensorFlow e Keras, treinada em um dataset de imagens de folhas. Para facilitar a interação e demonstração, uma aplicação web foi criada usando o framework Streamlit, permitindo que o usuário faça o upload de uma imagem e receba o diagnóstico em tempo real.

----- Tecnologias Utilizadas

Componente	               | Tecnologia	                 | Função
Modelo de Deep Learning	   |Python, TensorFlow, Keras	 |Treinamento e predição do modelo CNN.
Interface Web	           |Python, Streamlit	         |Interface amigável para upload de imagens e exibição do diagnóstico.
Processamento de Imagem	   |NumPy, PIL (Pillow)	         |Manipulação e pré-processamento de imagens.

----- Estrutura do Repositório

Arquivo/Pasta	      |  Descrição
app.py	              |Código da aplicação web Streamlit para o diagnóstico em tempo real.
train_model.py	      |Script para o treinamento da CNN, utilizando os dados de treino e validação.
predict_leaf.py	      |Script de exemplo para realizar predições em uma única imagem via terminal.
folha_cnn_model.h5 /  |
folha_cnn_model.keras | O modelo CNN treinado e salvo.
classes.json	      |Mapeamento das classes do modelo (ex: {"d": 0, "s": 1}).
data/ (Esperado)	  |Pasta de dados contendo as subpastas train e val com as imagens de treino e validação.

----- Instalação e Configuração

1. Pré-requisitos
Certifique-se de ter o Python (3.x) instalado em seu sistema.

2. Instalação de Bibliotecas
É altamente recomendável criar um ambiente virtual antes da instalação.

# Crie o ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Linux/Mac
.\venv\Scripts\activate   # No Windows

# Instale as dependências
pip install tensorflow keras numpy streamlit pillow
Como Usar:
A. Preparação dos Dados (Para Treinamento)
Para treinar o modelo, você precisa de um dataset de imagens estruturado da seguinte forma no diretório raiz do projeto:

data/
├── train/
│   ├── d/       # Imagens de folhas com anomalias (doentes)
│   └── s/       # Imagens de folhas saudáveis
└── val/
    ├── d/       # Imagens de validação com anomalias
    └── s/       # Imagens de validação saudáveis
(As letras 'd' e 's' são inferidas do arquivo classes.json e dos scripts.)

B. Treinamento do Modelo
Execute o script de treinamento. O modelo será treinado e salvo como folha_cnn_model.h5 (ou .keras), e o mapeamento das classes será salvo em classes.json.

&Bash;

python train_model.py
C. Uso da Aplicação Web (Recomendado)
A maneira mais prática de usar o sistema é através da interface Streamlit.

Certifique-se de que o modelo (folha_cnn_model.h5 ou .keras) e o arquivo classes.json estejam no diretório correto.

Execute a aplicação:

&Bash;

streamlit run app.py
O aplicativo será aberto automaticamente no seu navegador. Basta fazer o upload da imagem da folha e o diagnóstico será exibido.

D. Predição via Script
Se você precisar fazer uma predição rápida via terminal:

Ajuste a linha no final do arquivo predict_leaf.py com o caminho da sua imagem:

Python

# Exemplo no predict_leaf.py:
analisar_folha("caminho/para/sua/imagem.jpg")
Execute o script:

&Bash;

python predict_leaf.py


----- Detalhes Técnicos

O modelo é uma Rede Neural Convolucional Simples com a seguinte arquitetura (vista em train_model.py):

Camada Conv2D (32 filtros)

Camada MaxPooling2D

Camada Conv2D (64 filtros)

Camada MaxPooling2D

Camada Flatten

Camada Dense (128 neurônios, ativação relu)

Camada Dense (Saída com ativação softmax)

O modelo é compilado com o otimizador Adam e a função de perda categorical_crossentropy.

-----

Autor`: Pedro Castro Barros.
Centro Universitario do Distrito Federal.
Orientadora`: Kadidja Valeria Reginaldo de Oliveira.