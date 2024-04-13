#Importa bibliotecas
import streamlit as st
import pandas as pd
import numpy as np

# Define a configuração da página do aplicativo Streamlit
st.set_page_config(page_title='SINASC NYC',
                   page_icon='https://images.vexels.com/media/users/3/145773/isolated/preview/a4f1f6d6f8ba68ed67650e3feafda0ab-logotipo-da-cidade-de-nova-york.png',
                   layout='wide')

#Define titulo principal
st.title("Uber pickups in NYC")

# Define o nome da coluna de data/hora
DATE_COLUMN = "date/time"
# URL dos dados
DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/"
    "streamlit-demo-data/uber-raw-data-sep14.csv.gz"
)


# Função para carregar os dados com caching
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)

    def lowercase(x):
        return str(x).lower()

# Converte para formatos data e hora
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Cria um elemento de texto para indicar que os dados estão sendo carregados
data_load_state = st.text("Loading data...")
# Carrega dados
data = load_data(10000)
# Notifica que os dados foram carregados com sucesso
data_load_state.text("Done! (using st.cache_data) =]")

# Checkbox para mostrar os dados brutos
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data)

# Subtítulo para o número de pickups por hora
st.subheader("Number of pickups by hour")
# Calcula o histograma das horas das pickups
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
# Plota um gráfico de barras com os valores do histograma
st.bar_chart(hist_values)

# Controle deslizante para filtrar pickups por hora
hour_to_filter = st.slider("hour", 0, 23, 17)
# Filtra os dados para incluir apenas as pickups na hora selecionada
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# Subtítulo indicando o mapa das pickups na hora selecionada
st.subheader(f"Map of all pickups at {hour_to_filter}:00")
# Plota um mapa com as pickups na hora selecionada
st.map(filtered_data)
