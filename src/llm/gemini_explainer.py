# ARQUIVO ATUALIZADO
#
# Mudança em relação à versão original (marcada com "# ATUALIZAÇÃO:"):
#   - explain_spam_reason agora recebe a probabilidade prevista pela
#     Regressão Logística e um dicionário de "sinais" (features lexicais)
#     extraídos no pipeline, e injeta esses dados no prompt. Isso torna a
#     explicação gerativa fundamentada no que o modelo realmente detectou,
#     em vez de uma justificativa especulativa gerada só a partir do texto
#     (ver seção 3 da revisão).

import os

from dotenv import load_dotenv
from google import genai

# Carrega a variável GEMINI_API_KEY do arquivo .env
load_dotenv()

# O cliente instanciado vazio já busca automaticamente a variável de ambiente GEMINI_API_KEY
client = genai.Client()


def explain_spam_reason(mensagem: str, probabilidade: float = None,
                         sinais: dict = None) -> str:
    """
    Envia uma mensagem de spam para o Gemini explicar o motivo da classificação.

    :param mensagem: Texto do SMS classificado como spam.
    :param probabilidade: Probabilidade de spam prevista pela Regressão
        Logística (predict_proba), usada para dar contexto de confiança.
    :param sinais: Dicionário com sinais lexicais detectados (ex.:
        {'contem_url': True, 'proporcao_maiusculas': 0.42, ...}), usado
        para fundamentar a explicação em evidência real do pipeline.
    """
    try:
        # ATUALIZAÇÃO: construção de um bloco de contexto com evidências
        # concretas do modelo, em vez de mandar só a mensagem crua
        contexto_extra = ""
        if probabilidade is not None:
            contexto_extra += f"\nConfiança do modelo: {probabilidade:.1%} de probabilidade de spam."
        if sinais:
            sinais_formatados = ", ".join(f"{k}: {v}" for k, v in sinais.items())
            contexto_extra += f"\nSinais detectados automaticamente: {sinais_formatados}."

        prompt = (
            "Você é um especialista em cibersegurança. Nosso filtro de Machine Learning "
            "acabou de classificar o SMS abaixo como SPAM. Explique de forma clara e em "
            "no máximo 2 frases por que essa mensagem é suspeita, usando os sinais "
            "detectados automaticamente como evidência quando disponíveis.\n"
            f"{contexto_extra}\n\n"
            f"Mensagem SMS: '{mensagem}'"
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return f"Erro ao contatar o LLM: {str(e)}"