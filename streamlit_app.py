import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# ----------------------------
# Banco de dados maior
# ----------------------------
np.random.seed(42)
num_medicos = 100
nomes = [f'Dr. {nome}' for nome in np.random.choice(
    ['Ana', 'Bruno', 'Carlos', 'Daniela', 'Eduardo', 'Fernanda', 'Gustavo', 'Helena', 'Igor', 'Juliana'], num_medicos
)]

df = pd.DataFrame({
    'Medico': nomes,
    'Horas_trabalhadas': np.random.randint(100, 220, size=num_medicos),
    'Cirurgias': np.random.randint(0, 25, size=num_medicos),
})

df['Risco_Horas_Extras'] = (df['Horas_trabalhadas'] > 180).astype(int)

# ----------------------------
# Explicação breve
# ----------------------------
st.title("Dashboard de Médicos")
st.markdown("""
Bem-vindo ao dashboard! Aqui você pode analisar a carga de trabalho, cirurgias e risco de horas extras dos médicos.
- **Clusters**: grupos de médicos com padrões similares de horas e cirurgias.
- **Risco de horas extras**: classificado pelo modelo XGBoost.
- **Importância das variáveis**: mostra o impacto das horas e cirurgias na classificação do risco.
""")

# ----------------------------
# XGBoost Classificação
# ----------------------------
X = df[['Horas_trabalhadas','Cirurgias']]
y = df['Risco_Horas_Extras']

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X, y)

# ----------------------------
# KMeans Agrupamento
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = st.sidebar.slider("Número de Clusters", 2, 6, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------
# Filtros e busca interativos
# ----------------------------
st.sidebar.subheader("Filtros")
cluster_filter = st.sidebar.multiselect("Selecione Cluster(s)", df['Cluster'].unique(), default=df['Cluster'].unique())
risco_filter = st.sidebar.multiselect("Risco de Horas Extras", [0,1], default=[0,1])
nome_filter = st.sidebar.multiselect("Selecione Médicos", df['Medico'].unique(), default=df['Medico'].unique())

df_filtered = df[
    (df['Cluster'].isin(cluster_filter)) & 
    (df['Risco_Horas_Extras'].isin(risco_filter)) &
    (df['Medico'].isin(nome_filter))
]

st.subheader(f"Médicos selecionados: {len(df_filtered)}")
st.dataframe(df_filtered[['Medico','Horas_trabalhadas','Cirurgias','Risco_Horas_Extras','Cluster']])

# ----------------------------
# Visualização interativa
# ----------------------------
fig = px.scatter(
    df_filtered, x='Horas_trabalhadas', y='Cirurgias', color='Cluster',
    symbol='Risco_Horas_Extras', hover_data=['Medico']
)
st.plotly_chart(fig)

# ----------------------------
# Importância das variáveis
# ----------------------------
st.subheader("Importância das Variáveis (XGBoost)")
importances = xgb_model.feature_importances_
imp_df = pd.DataFrame({
    'Variavel': ['Horas_trabalhadas','Cirurgias'],
    'Importancia': importances
})
st.bar_chart(imp_df.set_index('Variavel'))
st.markdown("""
> Observação: quanto maior a importância, maior o impacto daquela variável na previsão do risco de horas extras.
""")

# ----------------------------
# Previsão de risco para novo médico
# ----------------------------
st.subheader("Previsão de Risco para Novo Médico")
novo_horas = st.number_input("Horas Trabalhadas", min_value=0, max_value=300, value=160)
novo_cirurgias = st.number_input("Cirurgias", min_value=0, max_value=50, value=10)

if st.button("Prever Risco"):
    novo_df = pd.DataFrame({'Horas_trabalhadas':[novo_horas],'Cirurgias':[novo_cirurgias]})
    risco_previsto = xgb_model.predict(novo_df)[0]
    st.write("Risco de horas extras:", "Sim" if risco_previsto==1 else "Não")

# ----------------------------
# Download CSV filtrado
# ----------------------------
st.subheader("Baixar dados filtrados")
st.download_button("Download CSV", df_filtered.to_csv(index=False), "medicos_filtrados.csv")
