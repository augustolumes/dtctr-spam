import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_vectorize(df: pd.DataFrame):
    """
    Codifica a variável alvo, divide os dados e aplica TF-IDF no texto.
    
    :param df: DataFrame bruto contendo 'label' e 'mensagem'.
    :return: Matrizes de treino e teste e o vetorizador treinado.
    """
    # Codificação Numérica
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    X = df['mensagem']
    y = df['label_num']

    # Separação de Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Engenharia de Features (Vetorização)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer