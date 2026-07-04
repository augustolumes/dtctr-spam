# ARQUIVO ATUALIZADO
#
# Mudanças em relação à versão original (marcadas com "# ATUALIZAÇÃO:"):
#   - Recebe feature_names de preprocess_and_vectorize.
#   - Passa feature_names para save_model_and_vectorizer (metadados).
#   - No passo de explicação, monta o vetor de features do SMS suspeito
#     (embedding normalizado + features lexicais), usa predict_proba,
#     e envia probabilidade + sinais lexicais ao Gemini.

import os

import numpy as np

from src.evaluation.evaluate import evaluate_model
from src.features.lexical_features import extract_lexical_features, normalizar_leetspeak
from src.ingestion.load_data import load_raw_data
from src.llm.gemini_explainer import explain_spam_reason
from src.models.save_model import save_model_and_vectorizer
from src.models.train_model import train_logistic_regression
from src.preprocessing.process_data import preprocess_and_vectorize


def main():
    print("🚀 Iniciando pipeline de Machine Learning (Filtro de Spam)...\n")

    # 1. Ingestão
    print("[1/4] Carregando dados...")
    filepath = os.path.join("data", "raw", "SMSSpamCollection")
    df = load_raw_data(filepath)

    # 2. Pré-processamento
    print("[2/4] Pré-processando texto e extraindo features (Embeddings + lexicais)...")
    # ATUALIZAÇÃO: agora recebemos também feature_names
    X_train, X_test, y_train, y_test, embedder, feature_names = preprocess_and_vectorize(df)

    # 3. Treinamento
    print("[3/4] Treinando modelo de Regressão Logística...")
    modelo = train_logistic_regression(X_train, y_train)

    # 4. Avaliação
    print("[4/4] Avaliando performance do modelo...\n")
    metricas = evaluate_model(modelo, X_test, y_test)

    print("=== Resultados Finais ===")
    for metrica, valor in metricas.items():
        print(f"> {metrica.capitalize()}: {valor:.4f}")

    # --- Persistência ---
    print("\n[Extra] Salvando modelo em disco...")
    # ATUALIZAÇÃO: passa feature_names para registrar nos metadados
    save_model_and_vectorizer(modelo, embedder, feature_names=feature_names)
    print("💾 Arquivo 'modelo_lr.joblib' e 'metadata.json' exportados com sucesso "
          "para a pasta 'models_saved/'!")
    # --------------------------------

    print("\n✅ Pipeline executado com sucesso!")

    # 5. Explicação com LLM
    print("\n[5/5] criando laudo com o Google Gemini...")

    # Simulando um SMS que o modelo tradicional barrou
    sms_suspeito = "URGENT! Your bank account has been locked. Click here to verify your details: http://scam-link.com"

    print(f"Mensagem retida: '{sms_suspeito}'")

    # ATUALIZAÇÃO: monta o vetor híbrido (embedding normalizado + features lexicais)
    # exatamente como foi feito no treino, para manter consistência
    sms_normalizado = normalizar_leetspeak(sms_suspeito)
    embedding_sms = embedder.encode([sms_normalizado], normalize_embeddings=True)
    lex_sms = extract_lexical_features([sms_suspeito])
    sms_vetorizado = np.hstack([embedding_sms, lex_sms])

    predicao = modelo.predict(sms_vetorizado)[0]
    # ATUALIZAÇÃO: obtém a probabilidade, não só a classe
    probabilidade_spam = modelo.predict_proba(sms_vetorizado)[0][1]
    resultado_modelo = "SPAM (1)" if predicao == 1 else "HAM (0)"
    print(f"Resultado do nosso modelo Local: {resultado_modelo} "
          f"(confiança: {probabilidade_spam:.1%})\n")

    print("Gerando explicação...")

    # ATUALIZAÇÃO: monta o dicionário de sinais lexicais em formato legível
    # e envia junto com a probabilidade para o LLM
    valores_lexicais = lex_sms[0]
    sinais = dict(zip(feature_names, valores_lexicais.tolist()))

    explicacao = explain_spam_reason(sms_suspeito, probabilidade=probabilidade_spam, sinais=sinais)
    print(f"🤖 Diagnóstico da IA: {explicacao}")

    print("\n✅ Pipeline completo e integrado com sucesso!")


if __name__ == "__main__":
    main()