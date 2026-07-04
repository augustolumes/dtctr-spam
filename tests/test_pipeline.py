# ARQUIVO ATUALIZADO
#
# Mudança em relação à versão original (marcada com "# ATUALIZAÇÃO:"):
#   - O teste antigo verificava `hasattr(vectorizer, 'vocabulary_')`, um
#     atributo do TfidfVectorizer (sklearn) que NÃO existe mais desde a
#     migração para SentenceTransformer — esse teste estava obsoleto e
#     provavelmente falhando/dando falso positivo. Agora validamos a
#     dimensionalidade real do embedding + features lexicais concatenadas.

import pandas as pd
import pytest

from src.preprocessing.process_data import preprocess_and_vectorize


def test_preprocess_and_vectorize_success():
    """
    Testa se a função de pré-processamento lida corretamente com as entradas,
    realiza o split adequadamente e gera embeddings + features lexicais
    com a dimensionalidade esperada.
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
    # ATUALIZAÇÃO: a função agora retorna também feature_names
    X_train, X_test, y_train, y_test, embedder, feature_names = preprocess_and_vectorize(df_mock)

    # 3. Verificação (Assert): Checa se os resultados são os esperados

    # Verifica se os dados não voltaram vazios
    assert X_train is not None
    assert X_test is not None

    # Como definimos test_size=0.2 (20%), e temos 6 amostras, o teste no mock deve ficar com 2 amostras e o treino com 4
    # Devido ao arredondamento e estratificação do train_test_split em datasets muito pequenos,
    # validamos se as partições foram geradas e se a soma é igual ao total
    assert X_train.shape[0] + X_test.shape[0] == 6

    # ATUALIZAÇÃO: valida a dimensionalidade combinada (embedding + features
    # lexicais), em vez do atributo `vocabulary_` do antigo TfidfVectorizer
    dim_embedding = embedder.get_sentence_embedding_dimension()
    dim_esperada = dim_embedding + len(feature_names)
    assert X_train.shape[1] == dim_esperada, (
        f"Dimensão do vetor final ({X_train.shape[1]}) não bate com "
        f"embedding ({dim_embedding}) + features lexicais ({len(feature_names)})."
    )
    assert X_test.shape[1] == dim_esperada

    # ATUALIZAÇÃO: garante que os nomes das features lexicais foram retornados
    assert len(feature_names) > 0