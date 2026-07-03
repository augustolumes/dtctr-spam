import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

def preprocess_and_vectorize(df: pd.DataFrame):
    """
    Codifica a variável alvo, divide os dados e aplica Embeddings no texto.
    
    :param df: DataFrame bruto contendo 'label' e 'mensagem'.
    :return: Matrizes de treino e teste e o embedder carregado.
    """
    # Codificação Numérica
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['mensagem']
    y = df['label_num']

    # Separação de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Engenharia de Features (Embeddings densos via Transformer)
    print("      Carregando modelo all-MiniLM-L6-v2 e gerando embeddings (pode demorar alguns segundos na 1ª vez)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    X_train_embeddings = embedder.encode(X_train.tolist())
    X_test_embeddings = embedder.encode(X_test.tolist())

    return X_train_embeddings, X_test_embeddings, y_train, y_test, embedder