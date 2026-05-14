import streamlit as st
import joblib
import numpy as np
from PIL import Image

#CÓDIGO GERADO VIA INTELIGÊNCIA ARTIFICIAL


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Classificador de Espécies Iris",
    page_icon="🌸",
    layout="centered"
)

# ==========================================
# CARREGAMENTO DO MODELO
# ==========================================

modelo = joblib.load(r"models/iris_knn_model.pkl")

# ==========================================
# FUNÇÃO PARA PADRONIZAR IMAGENS
# ==========================================

def carregar_imagem(caminho):
    imagem = Image.open(caminho)
    imagem = imagem.resize((800, 800))
    return imagem

# ==========================================
# TÍTULO E DESCRIÇÃO
# ==========================================

st.title("🌸 Classificador de Espécies Iris")

st.markdown("""
Este dashboard utiliza um modelo de Machine Learning treinado com o dataset Iris
para prever a espécie da flor com base nas medidas fornecidas.
""")

st.divider()

# ==========================================
# INPUTS DO USUÁRIO
# ==========================================

st.subheader("📏 Insira as medidas da flor")

sepal_length = st.number_input(
    "Comprimento da Sépala (cm)",
    min_value=0.0,
    step=0.1
)

sepal_width = st.number_input(
    "Largura da Sépala (cm)",
    min_value=0.0,
    step=0.1
)

petal_length = st.number_input(
    "Comprimento da Pétala (cm)",
    min_value=0.0,
    step=0.1
)

petal_width = st.number_input(
    "Largura da Pétala (cm)",
    min_value=0.0,
    step=0.1
)

# ==========================================
# BOTÃO DE PREDIÇÃO
# ==========================================

if st.button("🔍 Gerar Predição"):

    # Dados para previsão
    dados = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Predição
    predicao = modelo.predict(dados)[0]

    # ==========================================
    # MAPEAMENTO DAS ESPÉCIES
    # ==========================================

    especies = {
        0: "Iris Setosa",
        1: "Iris Versicolor",
        2: "Iris Virginica"
    }

    especie_predita = especies[predicao]

    # ==========================================
    # RESULTADO
    # ==========================================

    st.success(f"✅ Espécie predita: {especie_predita}")

    st.divider()

    # ==========================================
    # CONTEÚDO PARA CADA ESPÉCIE
    # ==========================================

    # ------------------------------
    # IRIS Setosa
    # ------------------------------
    if especie_predita == "Iris Setosa":

        st.header("🌱 Iris Setosa")

        st.markdown("""
        ### Características da espécie

        A Iris Setosa é uma das espécies mais conhecidas do conjunto de dados Iris e se destaca por suas pétalas menores e mais largas quando comparadas às outras espécies. Suas flores normalmente apresentam coloração violeta intensa e possuem uma estrutura compacta.

        Essa espécie é encontrada principalmente em regiões frias e úmidas do hemisfério norte, adaptando-se bem a ambientes próximos de áreas pantanosas e margens de rios.

        Uma das principais diferenças da Iris Setosa está no tamanho reduzido de suas pétalas, característica que facilita bastante sua separação em modelos de Machine Learning. Por esse motivo, ela costuma ser a espécie mais fácil de classificar no dataset Iris.

        Curiosamente, a Iris Setosa é frequentemente utilizada em estudos introdutórios de Ciência de Dados e Machine Learning por apresentar padrões bem definidos em suas medidas botânicas.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(carregar_imagem("img/Setosa1.jpg"))

        with col2:
            st.image(carregar_imagem("img/Setosa2.jpg"))

        with col3:
            st.image(carregar_imagem("img/Setosa3.jpg"))

    # ------------------------------
    # IRIS Versicolor
    # ------------------------------
    elif especie_predita == "Iris Versicolor":

        st.header("🌸 Iris Versicolor")

        st.markdown("""
        ### Características da espécie

        A Iris Versicolor é uma espécie de porte intermediário, apresentando características que ficam entre a Iris Setosa e a Iris Virginica. Suas pétalas possuem tamanho moderado e geralmente exibem tons de violeta, azul ou lilás.

        Ela é encontrada principalmente em áreas úmidas da América do Norte, especialmente em regiões próximas a lagos, pântanos e margens de rios.

        A Iris Versicolor costuma ser mais difícil de classificar em modelos de Machine Learning porque suas medidas podem se sobrepor às da Iris Virginica. Essa proximidade entre características torna a tarefa de predição mais desafiadora para alguns algoritmos.

        Além de sua relevância em estudos de Machine Learning, a Iris Versicolor também possui importância ornamental devido à beleza e variedade de cores de suas flores.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(carregar_imagem("img/Versicolor.jpg"))

        with col2:
            st.image(carregar_imagem("img/Versicolor.jpg"))

        with col3:
            st.image(carregar_imagem("img/Versicolor.jpg"))

    # ------------------------------
    # IRIS Virginica
    # ------------------------------
    elif especie_predita == "Iris Virginica":

        st.header("💐 Iris Virginica")

        st.markdown("""
        ### Características da espécie

        A Iris Virginica é conhecida por possuir flores maiores e pétalas mais longas entre as espécies do dataset Iris. Sua coloração varia entre tons de violeta e roxo intenso, apresentando uma aparência elegante e marcante.

        Essa espécie é nativa de regiões úmidas da América do Norte, sendo frequentemente encontrada em brejos, áreas alagadas e terrenos com alta umidade.

        Uma característica importante da Iris Virginica é o tamanho avantajado de suas pétalas e sépalas, fator que ajuda modelos de Machine Learning a diferenciá-la das demais espécies. Ainda assim, ela pode apresentar algumas semelhanças com a Iris Versicolor em determinados casos.

        A Iris Virginica é bastante valorizada na jardinagem ornamental devido ao tamanho de suas flores e à resistência da planta em diferentes condições climáticas.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(carregar_imagem("img/Virginica.jpg"))

        with col2:
            st.image(carregar_imagem("img/Virginica.jpg"))

        with col3:
            st.image(carregar_imagem("img/Virginica.jpg"))

# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption("Projeto de Machine Learning com Streamlit")