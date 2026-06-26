from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Treina o modelo de Regressão Logística para classificação de spam.
    
    :param X_train: Matriz de features de treino (TF-IDF).
    :param y_train: Labels de treino.
    :return: Modelo treinado.
    """
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
    modelo_lr.fit(X_train, y_train)
    return modelo_lr