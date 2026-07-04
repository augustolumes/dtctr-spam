# ARQUIVO ATUALIZADO
#
# Mudanças em relação à versão original:
#   1. Normalização de leetspeak antes do embedding (reduz OOV).
#   2. embedder.encode(..., normalize_embeddings=True) — L2-normalização,
#      recomendada para classificadores lineares como a Regressão Logística.
#   3. Extração de features lexicais (URL, maiúsculas, símbolos, dígitos)
#      concatenadas ao embedding semântico (hibridização).
#   4. Retorno passa a incluir os nomes das features, usados depois para
#      montar o dicionário de "sinais" enviado ao Gemini (seção 3 da revisão).

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# ATUALIZAÇÃO: import do novo módulo de features lexicais
from src.features.lexical_features import (
    build_feature_names,
    extract_lexical_features,
    normalizar_leetspeak,
)

# ATUALIZAÇÃO: nome do modelo e revisão centralizados em constantes,
# para permitir versionamento explícito (ver save_model.py)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = None  # defina um hash de revisão do HF Hub em produção


def preprocess_and_vectorize(df: pd.DataFrame):
    """
    Codifica a variável alvo, divide os dados e aplica Embeddings + features
    lexicais no texto.

    :param df: DataFrame bruto contendo 'label' e 'mensagem'.
    :return: Matrizes de treino e teste (embeddings + features lexicais),
             labels, o embedder carregado e a lista de nomes das features
             lexicais (para uso posterior na explicação via LLM).
    """
    # Codificação Numérica
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['mensagem']
    y = df['label_num']

    # Separação de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ATUALIZAÇÃO: normalização de leetspeak antes do embedding
    # (mitiga OOV para erros ortográficos propositais — ver seção 5 da revisão)
    X_train_norm = [normalizar_leetspeak(m) for m in X_train.tolist()]
    X_test_norm = [normalizar_leetspeak(m) for m in X_test.tolist()]

    # Engenharia de Features (Embeddings densos via Transformer)
    print("      Carregando modelo all-MiniLM-L6-v2 e gerando embeddings "
          "(pode demorar alguns segundos na 1ª vez)...")
    embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        revision=EMBEDDING_MODEL_REVISION,
    )

    # ATUALIZAÇÃO: normalize_embeddings=True (L2) — melhora a escala das
    # features para a Regressão Logística; batch_size explícito
    X_train_embeddings = embedder.encode(
        X_train_norm, normalize_embeddings=True, batch_size=32
    )
    X_test_embeddings = embedder.encode(
        X_test_norm, normalize_embeddings=True, batch_size=32
    )

    # ATUALIZAÇÃO: extração e concatenação de features lexicais
    # (texto ORIGINAL, não normalizado — queremos capturar o próprio
    # sinal de ofuscação, ex.: presença de dígitos/símbolos, como feature)
    lex_train = extract_lexical_features(X_train.tolist())
    lex_test = extract_lexical_features(X_test.tolist())

    X_train_final = np.hstack([X_train_embeddings, lex_train])
    X_test_final = np.hstack([X_test_embeddings, lex_test])

    feature_names = build_feature_names()

    return X_train_final, X_test_final, y_train, y_test, embedder, feature_names