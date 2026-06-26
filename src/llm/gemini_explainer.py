from google import genai
import os
from dotenv import load_dotenv

# Carrega a variável GEMINI_API_KEY do arquivo .env
load_dotenv()

# O cliente instanciado vazio já busca automaticamente a variável de ambiente GEMINI_API_KEY
client = genai.Client()

def explain_spam_reason(mensagem: str) -> str:
    """
    Envia uma mensagem de spam para o Gemini explicar o motivo da classificação.
    """
    try:
        prompt = (
            "Você é um especialista em cibersegurança. Nosso filtro de Machine Learning "
            "acabou de classificar o SMS abaixo como SPAM. Explique de forma clara e em "
            "no máximo 2 frases por que essa mensagem é suspeita.\n\n"
            f"Mensagem SMS: '{mensagem}'"
        )
        
        # Chamada usando o novo padrão do SDK
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
        
    except Exception as e:
        return f"Erro ao contatar o LLM: {str(e)}"