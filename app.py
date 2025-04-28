import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


st.set_page_config(page_title="Classificador de Gêneros de Filmes", layout="wide")


st.title("NLP Genre Classification")


@st.cache_resource
def load_models():
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/genre_classifier.pkl')
    mlb = joblib.load('models/label_encoder.pkl')
    return tfidf, model, mlb

tfidf, model, mlb = load_models()


GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", 
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western"
]


with st.form("genre_classifier_form"):
    st.subheader("Insira a sinopse do filme")
    overview = st.text_area("Sinopse:", "", height=150)
    
    threshold = st.slider(
        "Limiar de confiança:", 
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Ajuste o limiar para considerar um gênero como presente"
    )
    
    submitted = st.form_submit_button("Prever Gêneros")
    
    if submitted and overview:
        # Pré-processamento do texto
        text_processed = overview.lower().strip()
        
        # Vetorização com TF-IDF
        X = tfidf.transform([text_processed])
        
        # Previsão
        y_pred_proba = model.predict_proba(X)
        
        # Aplicar limiar
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Decodificar gêneros
        predicted_genres = mlb.inverse_transform(y_pred)
        
        # Exibir resultados
        st.subheader("Resultados")
        
        if predicted_genres and predicted_genres[0]:
            st.success("Gêneros previstos:")
            for genre in predicted_genres[0]:
                st.markdown(f"- {genre}")
            
            # Exibir probabilidades
            st.subheader("Probabilidades por Gênero")
            prob_df = pd.DataFrame({
                "Gênero": GENRES,
                "Probabilidade": np.round(y_pred_proba[0] * 100, 1)
            }).sort_values("Probabilidade", ascending=False)
            
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        else:
            st.warning("Nenhum gênero identificado com o limiar atual.")

# Seção de informações
with st.expander("ℹ️ Sobre este classificador"):
    st.markdown("""
    Este classificador utiliza:
    - **TF-IDF** para vetorização do texto
    - Um modelo **OneVsRestClassifier** com **LogisticRegression**
    - Foi treinado em um dataset de filmes com seus gêneros
    
    Ajuste o limiar para controlar quão conservadora é a previsão:
    - Limiares mais baixos = mais gêneros identificados (mas possivelmente menos precisos)
    - Limiares mais altos = menos gêneros (mas mais certeiros)
    """)