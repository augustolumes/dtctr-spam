# Classificador Automático de Spam com Regressão Logística e GenAI
 
## 📌 Explicação do Problema
 
O volume de mensagens indesejadas (Spam) em caixas de entrada e SMS é um problema crônico de segurança e produtividade. Este projeto tem como objetivo construir um classificador automático capaz de distinguir mensagens legítimas (Ham) de mensagens indesejadas (Spam). Além da triagem inicial, o sistema conta com integração a Large Language Models (LLMs) para explicar o motivo da classificação de ameaças.
 
---

## 🎲 Dataset utilizado
Nome do dataset: SMS Spam Collection.<br>
Autores: Tiago Almeida, Jos Hidalgo.<br>
Disponível em: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection.<br> Acessado em: 26 jun. 2026.<br>

---
 
## ⚙️ Instruções de Execução
 
### Pré-requisitos
 
- Python 3.10 ou superior
- Conta no [Google AI Studio](https://aistudio.google.com/) (para obtenção da chave de API do Gemini)
### Passo a Passo
 
1. **Clonar o Repositório:**
```bash
git clone <url-do-repositorio>
cd projeto_ml
```
 
2. **Configurar e Ativar o Ambiente Virtual:**
```bash
# Criação do ambiente
python -m venv .venv
 
# Ativação (Windows)
.venv\Scripts\activate
 
# Ativação (Linux/Mac)
source .venv/bin/activate
```
 
3. **Instalar as Dependências:**
```bash
pip install -r requirements.txt
```
 
4. **Configurar as Variáveis de Ambiente:**
   Crie um arquivo chamado `.env` na raiz do projeto e insira a sua chave de acesso à API do Gemini:
```env
GEMINI_API_KEY=AIzaSy...seu_token_aqui
```
 
5. **Executar o Pipeline Completo:**
```bash
python main.py
```
 
---
 
## 📚 Bibliotecas Utilizadas e Justificativa
 
| Biblioteca | Justificativa |
|---|---|
| `pandas` & `numpy` | Essenciais para a manipulação de matrizes, ingestão estruturada do arquivo bruto e tratamento dos dados. |
| `scikit-learn` | Utilizado para a divisão estratificada dos dados (`train_test_split`), treinamento do classificador linear e cálculo estatístico de validação. |
| `sentence-transformers` | Utilizado para a extração de embeddings semânticos densos das mensagens de texto (usando o modelo `all-MiniLM-L6-v2`), modernizando a vetorização. |
| `google-genai` | SDK oficial e atualizado da Google para a realização de chamadas nativas de inferência ao modelo `gemini-2.5-flash`. |
| `python-dotenv` | Utilizado para carregar de forma transparente e segura as variáveis de ambiente a partir do arquivo local `.env`. |
| `pytest` | Framework recomendado para a construção e execução dos testes unitários, garantindo a validação das entradas. |
 
---
 
## 📊 Análise de Resultados e Métricas
 
O modelo adotado para o pipeline de produção combina **Embeddings Semânticos** (via Transformers) com **Regressão Logística**. Os textos são convertidos em vetores densos e profundos antes da classificação, garantindo um alto desempenho na identificação semântica de Spams. Os resultados consolidados obtidos no conjunto de teste foram:
 
| Métrica | Valor | Interpretação |
|---|---|---|
| Acurácia | `0.9767` | Taxa geral de acerto de 97,67% em todas as classificações. |
| Precisão | `0.9556` | Altíssima precisão, apontando raríssimos falsos positivos. |
| Recall | `0.8658` | O modelo interceptou 86,58% de todas as mensagens reais de Spam no conjunto de validação, um salto em relação à abordagem estatística pura. |
| F1-Score | `0.9085` | Média harmônica equilibrada que demonstra o excelente balanço entre Precision e Recall. |
 
### 🌟 Diferencial Tecnológico — Explicabilidade com IA Generativa
 
Quando uma mensagem suspeita ultrapassa a fronteira de decisão da Regressão Logística e é classificada como Spam (`label = 1`), o pipeline aciona a camada do LLM (`gemini-2.5-flash`). O modelo generativo analisa a estrutura linguística do SMS barrado e gera um **laudo explicativo** direto para o usuário final, detalhando o gatilho de engenharia social ou a fraude identificada.
 
---
 
## 🧪 Testes Unitários
 
Para garantir a confiabilidade e resiliência da aplicação, o projeto conta com testes automatizados utilizando o framework `pytest`. O escopo de testes cobre o módulo central de pré-processamento, validando:
 
- A correta divisão estratificada entre dados de treino e teste.
- A instanciação e a correta geração de Embeddings, assegurando as dimensões dos vetores (features) passados ao modelo.
- O tratamento de entradas, garantindo que matrizes nulas não quebrem o pipeline.
Para executar a suíte de testes, na raiz do projeto com o ambiente virtual ativado, rode o comando:
 
```bash
python -m pytest
```
 
---