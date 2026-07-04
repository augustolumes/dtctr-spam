# ARQUIVO NOVO
#
# Motivação (ver revisão, seções 1, 2 e 5):
#   Embeddings semânticos (Sentence-Transformers) diluem sinais estilísticos
#   e ortográficos que são altamente preditivos de spam (URLs, maiúsculas
#   excessivas, símbolos monetários, dígitos, leetspeak). Este módulo extrai
#   esses sinais explicitamente para hibridizá-los com os embeddings, e
#   normaliza leetspeak/erros propositais para reduzir OOV antes do embedding.

import re

import numpy as np


def extract_lexical_features(mensagens: list[str]) -> np.ndarray:
    """
    Extrai features estilísticas/estatísticas complementares aos embeddings.

    Estas features capturam sinais que a compressão semântica de um
    Sentence-Transformer tende a perder: presença de URL, uso de símbolos
    monetários, proporção de maiúsculas, pontuação de urgência (!) e
    densidade de dígitos.

    :param mensagens: Lista de mensagens de texto brutas.
    :return: Array numpy (n_mensagens, 6) com as features extraídas.
    """
    feats = []
    for m in mensagens:
        tamanho = max(len(m), 1)
        feats.append([
            len(m),
            sum(c.isupper() for c in m) / tamanho,                     # proporção de maiúsculas
            m.count('!'),                                              # pontuação de urgência
            int(bool(re.search(r'[$£€]', m))),                         # símbolo monetário
            int(bool(re.search(r'http[s]?://|www\.', m))),             # presença de URL
            sum(c.isdigit() for c in m) / tamanho,                     # densidade de dígitos
        ])
    return np.array(feats, dtype=np.float32)


def normalizar_leetspeak(mensagem: str) -> str:
    """
    Normaliza substituições comuns de leetspeak (ex.: 'fr33' -> 'free')
    antes da geração do embedding, reduzindo a taxa de OOV para erros
    ortográficos propositais e neologismos comuns em spam.

    :param mensagem: Mensagem de texto bruta.
    :return: Mensagem normalizada.
    """
    substituicoes = {
        '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '@': 'a', '$': 's',
    }
    resultado = mensagem.lower()
    for char, letra in substituicoes.items():
        resultado = resultado.replace(char, letra)
    return resultado


def build_feature_names() -> list[str]:
    """Nomes das features lexicais, na mesma ordem de extract_lexical_features."""
    return [
        "tamanho_mensagem",
        "proporcao_maiusculas",
        "qtd_exclamacao",
        "contem_simbolo_monetario",
        "contem_url",
        "densidade_digitos",
    ]