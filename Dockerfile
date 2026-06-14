# 1. Imagen base oficial de Python optimizada
FROM python:3.11.3-slim

# 2. Definir el espacio aislado de ejecución
WORKDIR /app

# 3. Copiar e instalar las dependencias de Python primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Descargar los diccionarios de NLTK directamente en la construcción de la imagen
RUN python -m nltk.downloader stopwords

# 5. Copiar los scripts de código y los archivos binarios (.pkl)
COPY . .

# 6. Abrir las comunicaciones del puerto nativo de Streamlit
EXPOSE 8501

# 7. Ejecutar Streamlit asignando una IP global abierta
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]