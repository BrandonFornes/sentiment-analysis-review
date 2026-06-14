import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

st.set_page_config(page_title="Detector de Sentimiento Avanzado", layout="centered")

st.title("Análisis de Sentimiento Multiclase ")
st.write("Escribe un comentario en inglés para evaluar si es Negativo, Neutral o Positivo.")

@st.cache_resource
def cargar_artefactos():
    with open('modelo_svm_3clases.pkl', 'rb') as f:
        modelo = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizador = pickle.load(f)
    return modelo, vectorizador

modelo, tfidf = cargar_artefactos()

@st.cache_resource
def configurar_stopwords():
    nltk.download('stopwords')
    all_stopwords = stopwords.words('english')
    negaciones = {'not', 'no', 'nor', 'neither', 'never', 'but'}
    return set(all_stopwords) - negaciones

all_stopwords_set = configurar_stopwords()
ps = PorterStemmer()

# 4. Diseñar la interfaz de usuario
user_input = st.text_area("Comentario a analizar:", placeholder="Type something like: The service was not bad, but the food could be better...")

if st.button("Evaluar Sentimiento", type="primary"):
    if user_input.strip() != "":
        # --- PROCESAMIENTO NLP EN TIEMPO REAL ---
        comment = re.sub('[^a-zA-Z]', ' ', user_input)
        comment = comment.lower().split()
        comment = [ps.stem(word) for word in comment if not word in all_stopwords_set]
        comment = " ".join(comment)
        
        # --- VECTORIZACIÓN Y PREDICCIÓN ---
        # .transform() NO vuelve a entrenar, usa el diccionario de las 2,500 características fijas
        vectorizado = tfidf.transform([comment]).toarray()
        prediccion = modelo.predict(vectorizado)[0]
        
        # --- DESPLIEGUE VISUAL DEL RESULTADO MULTICLASE ---
        # Mapea estos valores de acuerdo al orden real de tus etiquetas (0, 1, 2)
        if prediccion == 0:
            st.error("🔴 **Resultado: Sentimiento Negativo**")
        elif prediccion == 1:
            st.warning("🟡 **Resultado: Sentimiento Neutral**")
        elif prediccion == 2:
            st.success("🟢 **Resultado: Sentimiento Positivo**")
    else:
        st.warning("Por favor, escribe un texto válido.")