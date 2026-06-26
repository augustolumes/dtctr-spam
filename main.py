import os
from src.ingestion.load_data import load_raw_data
from src.preprocessing.process_data import preprocess_and_vectorize
from src.models.train_model import train_logistic_regression
from src.models.save_model import save_model_and_vectorizer
from src.evaluation.evaluate import evaluate_model
from src.llm.gemini_explainer import explain_spam_reason

def main():
    print("🚀 Iniciando pipeline de Machine Learning (Filtro de Spam)...\n")
    
    # 1. Ingestão
    print("[1/4] Carregando dados...")
    filepath = os.path.join("data", "raw", "SMSSpamCollection")
    df = load_raw_data(filepath)
    
    # 2. Pré-processamento
    print("[2/4] Pré-processando texto e extraindo features (TF-IDF)...")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_vectorize(df)
    
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
    print("\n[Extra] Salvando modelo e vetorizador em disco...")
    save_model_and_vectorizer(modelo, vectorizer)
    print("💾 Arquivos '.joblib' exportados com sucesso para a pasta 'models_saved/'!")
    # --------------------------------
        
    print("\n✅ Pipeline executado com sucesso!")

    # 5. O Diferencial: Explicação com LLM (GenAI)
    print("\n[5/5] Testando o diferencial com Google Gemini...")
    
    # Simulando um SMS que o modelo tradicional barrou
    sms_suspeito = "URGENT! Your bank account has been locked. Click here to verify your details: http://scam-link.com"
    
    print(f"Mensagem retida: '{sms_suspeito}'")
    print("Gerando explicação...")
    
    explicacao = explain_spam_reason(sms_suspeito)
    print(f"🤖 Diagnóstico da IA: {explicacao}")
    
    print("\n✅ Pipeline completo e integrado com sucesso!")

if __name__ == "__main__":
    main()