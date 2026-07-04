# ARQUIVO ATUALIZADO
#
# Mudança em relação à versão original:
#   - Salva um metadata.json com nome/revisão do modelo de embedding e
#     dimensionalidade, evitando drift silencioso entre o embedder usado
#     em treino e o usado em inferência (ver seção 1 da revisão).

import json
import os

import joblib
import sklearn


def save_model_and_vectorizer(modelo, embedder, feature_names=None,
                               caminho_pasta="models_saved"):
    """
    Salva o modelo treinado. Não precisamos salvar o embedder localmente,
    pois o SentenceTransformers carrega automaticamente via HuggingFace.

    :param modelo: Modelo de Regressão Logística treinado.
    :param embedder: O embedder carregado (não será serializado).
    :param feature_names: Lista de nomes das features lexicais usadas
        na hibridização (para rastreabilidade).
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

    # ATUALIZAÇÃO: metadados de versionamento do embedder + features
    metadata = {
        "embedding_model": getattr(embedder, "model_card_data", None)
        and str(embedder.model_card_data)
        or "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": embedder.get_sentence_embedding_dimension(),
        "lexical_features": feature_names or [],
        "n_features_total": embedder.get_sentence_embedding_dimension()
        + len(feature_names or []),
        "sklearn_version": sklearn.__version__,
    }
    caminho_metadata = os.path.join(caminho_pasta, "metadata.json")
    with open(caminho_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)