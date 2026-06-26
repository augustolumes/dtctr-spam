# Classificador Automático de Spam com Regressão Logística e GenAI
 
## 📌 Explicação do Problema
 
O volume de mensagens indesejadas (Spam) em caixas de entrada e SMS é um problema crônico de segurança e produtividade. Este projeto tem como objetivo construir um classificador automático capaz de distinguir mensagens legítimas (Ham) de mensagens indesejadas (Spam). Além da triagem inicial, o sistema conta com integração a Large Language Models (LLMs) para explicar o motivo da classificação de ameaças.
 
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
| `scikit-learn` | Utilizado para a divisão estratificada dos dados (`train_test_split`), extração de atributos textuais via `TfidfVectorizer`, treinamento do classificador linear e cálculo estatístico de validação. |
| `google-genai` | SDK oficial e atualizado da Google para a realização de chamadas nativas de inferência ao modelo `gemini-2.5-flash`. |
| `python-dotenv` | Utilizado para carregar de forma transparente e segura as variáveis de ambiente a partir do arquivo local `.env`. |
| `pytest` | Framework recomendado para a construção e execução dos testes unitários, garantindo a validação das entradas. |
 
---
 
## 📊 Análise de Resultados e Métricas
 
O modelo adotado para o pipeline de produção foi a **Regressão Logística**, escolhida por sua alta eficiência computacional no tratamento de matrizes textuais esparsas de alta dimensionalidade. Os resultados consolidados obtidos no conjunto de teste foram:
 
| Métrica | Valor | Interpretação |
|---|---|---|
| Acurácia | `0.9677` | Taxa geral de acerto de 96,77% em todas as classificações. |
| Precisão | `1.0000` | Zero falsos positivos — nenhuma mensagem legítima (Ham) foi classificada erroneamente como Spam. |
| Recall | `0.7584` | O modelo interceptou 75,84% de todas as mensagens reais de Spam no conjunto de validação. |
| F1-Score | `0.8626` | Média harmônica equilibrada que valida a estabilidade e robustez do classificador binário. |
 
### 🌟 Diferencial Tecnológico — Explicabilidade com IA Generativa
 
Quando uma mensagem suspeita ultrapassa a fronteira de decisão da Regressão Logística e é classificada como Spam (`label = 1`), o pipeline aciona a camada do LLM (`gemini-2.5-flash`). O modelo generativo analisa a estrutura linguística do SMS barrado e gera um **laudo explicativo** direto para o usuário final, detalhando o gatilho de engenharia social ou a fraude identificada.
 
---
 
## 🧪 Testes Unitários
 
Para garantir a confiabilidade e resiliência da aplicação, o projeto conta com testes automatizados utilizando o framework `pytest`. O escopo de testes cobre o módulo central de pré-processamento, validando:
 
- A correta divisão estratificada entre dados de treino e teste.
- A instanciação e o ajuste (`fit`) do vetorizador TF-IDF, evitando vazamento de dados.
- O tratamento de entradas, garantindo que matrizes nulas não quebrem o pipeline.
Para executar a suíte de testes, na raiz do projeto com o ambiente virtual ativado, rode o comando:
 
```bash
python -m pytest
```
 
---