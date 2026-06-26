import joblib
import os

def save_model_and_vectorizer(modelo, vectorizer, caminho_pasta="models_saved"):
    """
    Salva o modelo treinado e o vetorizador TF-IDF no disco para reprodutibilidade.
    
    :param modelo: Modelo de Regressão Logística treinado.
    :param vectorizer: TfidfVectorizer ajustado aos dados de treino.
    :param caminho_pasta: Diretório onde os arquivos serão salvos.
    """
    # Garante que a pasta existe
    os.makedirs(caminho_pasta, exist_ok=True)
    
    caminho_modelo = os.path.join(caminho_pasta, "modelo_lr.joblib")
    caminho_vectorizer = os.path.join(caminho_pasta, "vectorizer_tfidf.joblib")
    
    # Exporta os arquivos
    joblib.dump(modelo, caminho_modelo)
    joblib.dump(vectorizer, caminho_vectorizer)