import pandas as pd
import pytest
from src.preprocessing.process_data import preprocess_and_vectorize

def test_preprocess_and_vectorize_success():
    """
    Testa se a função de pré-processamento lida corretamente com as entradas,
    realiza o split adequadamente e instancia o vetorizador.
    """
    # 1. Preparação (Arrange): Criação de dados simulados
    dados_mock = {
        'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
        'mensagem': [
            'Oi, tudo bem?', 
            'URGENTE: Você ganhou um prêmio', 
            'A reunião está confirmada para amanhã', 
            'Clique aqui para resgatar seu dinheiro', 
            'Não esqueça de comprar pão',
            'Sua conta foi bloqueada, acesse o link'
        ]
    }
    df_mock = pd.DataFrame(dados_mock)
    
    # 2. Ação (Act): Executa a função do nosso pipeline
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess_and_vectorize(df_mock)
    
    # 3. Verificação (Assert): Checa se os resultados são os esperados
    
    # Verifica se os dados não voltaram vazios
    assert X_train_tfidf is not None
    assert X_test_tfidf is not None
    
    # Como definimos test_size=0.2 (20%), e temos 6 amostras, o teste no mock deve ficar com 2 amostras e o treino com 4
    # Devido ao arredondamento e estratificação do train_test_split em datasets muito pequenos, 
    # validamos se as partições foram geradas e se a soma é igual ao total
    assert X_train_tfidf.shape[0] + X_test_tfidf.shape[0] == 6
    
    # Verifica se o TfidfVectorizer foi instanciado e tem vocabulário treinado
    assert hasattr(vectorizer, 'vocabulary_'), "O vetorizador não foi ajustado (fit) corretamente aos dados."