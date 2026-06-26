from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(modelo, X_test, y_test) -> dict:
    """
    Calcula as métricas principais do modelo de classificação.
    
    :param modelo: Modelo já treinado.
    :param X_test: Matriz de features de teste.
    :param y_test: Labels reais do conjunto de teste.
    :return: Dicionário contendo os resultados das métricas.
    """
    y_pred = modelo.predict(X_test)
    
    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    return metricas