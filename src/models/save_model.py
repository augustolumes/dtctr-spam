import joblib
import os

def save_model_and_vectorizer(modelo, embedder, caminho_pasta="models_saved"):
    """
    Salva o modelo treinado. Não precisamos salvar o embedder localmente,
    pois o SentenceTransformers carrega automaticamente via HuggingFace.
    
    :param modelo: Modelo de Regressão Logística treinado.
    :param embedder: O embedder carregado (não será serializado).
    :param caminho_pasta: Diretório onde os arquivos serão salvos.
    """
    # Garante que a pasta existe
    os.makedirs(caminho_pasta, exist_ok=True)
    
    caminho_modelo = os.path.join(caminho_pasta, "modelo_lr.joblib")
    
    # Exporta os arquivos
    joblib.dump(modelo, caminho_modelo)
    
    # O arquivo antigo TF-IDF pode ser deletado caso exista
    caminho_vectorizer = os.path.join(caminho_pasta, "vectorizer_tfidf.joblib")
    if os.path.exists(caminho_vectorizer):
        os.remove(caminho_vectorizer)