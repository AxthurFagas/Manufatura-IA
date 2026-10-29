import streamlit as st
import json
from pathlib import Path
import ollama
import numpy as np

# Configurações
MODELO = "qwen3:4b"
MODELO_EMBEDDING = "nomic-embed-text"
ARQUIVO_LICOES = Path(__file__).parent / "licoes.json"

# --- Funções de Lógica (RAG) ---

def carregar_licoes():
    with open(ARQUIVO_LICOES, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)

def gerar_embedding(texto):
    resposta = ollama.embeddings(model=MODELO_EMBEDDING, prompt=texto)
    return np.array(resposta["embedding"])

def calcular_similaridade(vetor1, vetor2):
    return np.dot(vetor1, vetor2) / (np.linalg.norm(vetor1) * np.linalg.norm(vetor2))

@st.cache_resource
def inicializar_indice():
    """Carrega as lições e cria o índice semântico apenas uma vez."""
    licoes = carregar_licoes()
    indice = []
    for licao in licoes:
        texto_combinado = f"{licao.get('equipamento', '')} {licao.get('problema', '')} {licao.get('causa', '')} {licao.get('solucao', '')}"
        vetor = gerar_embedding(texto_combinado)
        indice.append({"vetor": vetor, "licao": licao})
    return indice

def buscar_licoes_semanticas(pergunta, indice):
    vetor_pergunta = gerar_embedding(pergunta)
    resultados = []
    for item in indice:
        similaridade = calcular_similaridade(vetor_pergunta, item["vetor"])
        resultados.append((similaridade, item["licao"]))
    resultados.sort(key=lambda x: x[0], reverse=True)
    return resultados[:3]

def responder(pergunta, licoes_encontradas):
    if not licoes_encontradas:
        contexto = "Nenhuma lição aprendida correspondente foi encontrada."
    else:
        licoes_texto = [item[1] for item in licoes_encontradas]
        contexto = json.dumps(licoes_texto, ensure_ascii=False, indent=2)

    resposta = ollama.chat(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a manufacturing engineering assistant. "
                    "Answer in Brazilian Portuguese, with clarity and "
                    "technical language. Use the provided lessons as "
                    "evidence. Do not invent occurrences, causes, or solutions. "
                    "The records are fictitious test examples. "
                    "If no relevant lesson is found, report "
                    "that there is insufficient evidence in the base. "
                    "Do not present the examples as actual company facts."
                )
            },
            {
                "role": "user",
                "content": f"Pergunta: {pergunta}\n\nLições recuperadas da base:\n{contexto}"
            }
        ]
    )
    return resposta["message"]["content"]

# --- Interface Streamlit ---

st.set_page_config(page_title="Assistente de Manufatura", page_icon="⚙️")

st.title("⚙️ Assistente de Manufatura")
st.markdown("Consulte lições aprendidas e resolva problemas técnicos de forma rápida.")

# Barra lateral
with st.sidebar:
    st.header("Configurações")
    st.info(f"**Modelo LLM:** {MODELO}\n\n**Modelo Embedding:** {MODELO_EMBEDDING}")
    if st.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

# Inicialização do índice e histórico
if "indice" not in st.session_state:
    with st.spinner("Carregando base de conhecimento..."):
        st.session_state.indice = inicializar_indice()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibição do histórico de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("📚 Ver evidências recuperadas"):
                st.json(message["evidence"])

# Entrada do usuário
if prompt := st.chat_input("Ex: A prensa PH-01 está tremendo muito..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento..."):
            encontradas = buscar_licoes_semanticas(prompt, st.session_state.indice)
            
            if not encontradas or encontradas[0][0] < 0.3: 
                resposta = "Não encontrei lições correspondentes a essa pergunta na base de conhecimento. Não há evidências suficientes para fornecer uma orientação."
                evidencias = []
            else:
                resposta = responder(prompt, encontradas)
                evidencias = [item[1] for item in encontradas]

            st.markdown(resposta)
            if evidencias:
                with st.expander("📚 Ver evidências recuperadas"):
                    st.json(evidencias)
            
            msg_bot = {"role": "assistant", "content": resposta}
            if evidencias:
                msg_bot["evidence"] = evidencias
            st.session_state.messages.append(msg_bot)