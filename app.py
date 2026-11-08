
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==========================================
# CONFIGURAÇÕES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "lioniks_rfc.pkl"
DATASET_PATH = BASE_DIR / "data" / "dataset" / "dataset.csv"

st.set_page_config(
    page_title="LIONIKS | Classificador Químico",
    page_icon="🧪",
    layout="wide"
)

# ==========================================
# CARREGAMENTO DOS ARQUIVOS
# ==========================================

@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def carregar_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    modelo = carregar_modelo()
    dataset = carregar_dataset()

except FileNotFoundError as erro:
    st.error(f"Arquivo não encontrado: {erro}")
    st.stop()


# ==========================================
# IDENTIFICAÇÃO DAS CARACTERÍSTICAS
# ==========================================

colunas_removidas = [
    "family",
    "name",
    "symbol",
    "electron_configuration"
]

X = dataset.drop(
    columns=colunas_removidas,
    errors="ignore"
)

caracteristicas_categoricas = [
    "block",
    "state_25C",
    "type"
]

caracteristicas_numericas = X.select_dtypes(
    include=["number"]
).columns.tolist()


# ==========================================
# TRADUÇÕES
# ==========================================

traducoes = {
    "block": "Bloco eletrônico",
    "state_25C": "Estado físico a 25 °C",
    "type": "Classificação química",
    "atomic_number": "Número atômico",
    "atomic_mass": "Massa atômica",
    "density": "Densidade",
    "melting_point": "Ponto de fusão",
    "boiling_point": "Ponto de ebulição",
    "electronegativity": "Eletronegatividade",
    "atomic_radius": "Raio atômico",
    "ionization_energy": "Energia de ionização",
    "electron_affinity": "Afinidade eletrônica",
    "period": "Período",
    "group": "Grupo"
}

familias = {
    "Actinide": "Actinídeos",
    "Alkali Metal": "Metais alcalinos",
    "Alkaline Earth Metal": "Metais alcalino-terrosos",
    "Boron Group": "Família do boro",
    "Carbon Group": "Família do carbono",
    "Chalcogen": "Calcogênios",
    "Halogen": "Halogênios",
    "Lanthanide": "Lantanídeos",
    "Noble Gas": "Gases nobres",
    "Pnictogen": "Família do nitrogênio",
    "Transition Metal": "Metais de transição",
    "Hydrogen": "Hidrogênio"
}

valores_traduzidos = {
    "s": "Bloco s",
    "p": "Bloco p",
    "d": "Bloco d",
    "f": "Bloco f",
    "Solid": "Sólido",
    "Liquid": "Líquido",
    "Gas": "Gasoso",
    "Unknown": "Desconhecido"
}


def traduzir_coluna(coluna):
    return traducoes.get(
        coluna,
        coluna.replace("_", " ").capitalize()
    )


def traduzir_valor(valor):
    return valores_traduzidos.get(valor, valor)


# ==========================================
# CABEÇALHO
# ==========================================

st.title("🧪 LIONIKS")

st.subheader("Classificador de Famílias Químicas")

st.write(
    """
    O LIONIKS utiliza Machine Learning para identificar
    a família química de um elemento com base em suas
    propriedades físico-químicas.
    """
)

st.divider()

# ==========================================
# FORMULÁRIO
# ==========================================

st.markdown("### Propriedades do elemento")

st.info(
    "Preencha as características do elemento químico "
    "que deseja classificar."
)

with st.form("formulario_elemento"):

    dados = {}

    coluna1, coluna2 = st.columns(2)

    # Características categóricas
    with coluna1:

        st.markdown("#### Características categóricas")

        for caracteristica in caracteristicas_categoricas:

            if caracteristica not in X.columns:
                continue

            valores = sorted(
                X[caracteristica].dropna().unique().tolist(),
                key=str
            )

            if not valores:
                continue

            valor_selecionado = st.selectbox(
                traduzir_coluna(caracteristica),
                options=valores,
                format_func=traduzir_valor
            )

            dados[caracteristica] = valor_selecionado

    # Características numéricas
    with coluna2:

        st.markdown("#### Características numéricas")

        for caracteristica in caracteristicas_numericas:

            valores = pd.to_numeric(
                X[caracteristica],
                errors="coerce"
            ).dropna()

            if valores.empty:
                continue

            minimo = float(valores.min())
            maximo = float(valores.max())
            padrao = float(valores.median())

            if pd.api.types.is_integer_dtype(X[caracteristica]):

                dados[caracteristica] = st.number_input(
                    traduzir_coluna(caracteristica),
                    min_value=int(minimo),
                    max_value=int(maximo),
                    value=int(padrao),
                    step=1
                )

            else:

                dados[caracteristica] = st.number_input(
                    traduzir_coluna(caracteristica),
                    min_value=minimo,
                    max_value=maximo,
                    value=padrao,
                    format="%.6f"
                )

    st.divider()

    classificar = st.form_submit_button(
        "🔬 Classificar elemento",
        use_container_width=True
    )


# ==========================================
# CLASSIFICAÇÃO
# ==========================================

if classificar:

    entrada = pd.DataFrame([dados])

    try:

        previsao = modelo.predict(entrada)[0]

        probabilidades = modelo.predict_proba(entrada)[0]

        classes = modelo.classes_

        resultados = pd.DataFrame({
            "Família": [
                familias.get(classe, classe)
                for classe in classes
            ],
            "Probabilidade": probabilidades
        })

        resultados = resultados.sort_values(
            by="Probabilidade",
            ascending=False
        )

        # Resultado principal
        st.divider()

        st.success("Classificação concluída!")

        st.markdown("### Família química identificada")

        st.title(
            f"⚛️ {familias.get(previsao, previsao)}"
        )

        probabilidade_principal = probabilidades[
            list(classes).index(previsao)
        ]

        st.metric(
            "Probabilidade estimada",
            f"{probabilidade_principal:.2%}"
        )

        # Gráfico
        st.markdown("### Distribuição das probabilidades")

        grafico = resultados.set_index("Família")

        st.bar_chart(
            grafico,
            horizontal=True
        )

        # Tabela
        st.markdown("### Resultados detalhados")

        st.dataframe(
            resultados.style.format({
                "Probabilidade": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "As probabilidades representam estimativas do "
            "Random Forest, não garantias de certeza."
        )

    except Exception as erro:

        st.error(
            f"Erro durante a classificação: {erro}"
        )


# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption(
    "LIONIKS | Inteligência Artificial aplicada à Química"
)
