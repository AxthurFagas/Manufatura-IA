import json
from pathlib import Path
import ollama
import numpy as np

MODELO = "qwen3:4b"
MODELO_EMBEDDING = "nomic-embed-text"
ARQUIVO_LICOES = Path(__file__).parent / "licoes.json"

def carregar_licoes():
    with open(ARQUIVO_LICOES, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)

def gerar_embedding(texto):
    resposta = ollama.embeddings(model=MODELO_EMBEDDING, prompt=texto)
    return np.array(resposta["embedding"])

def calcular_similaridade(vetor1, vetor2):
    return np.dot(vetor1, vetor2) / (np.linalg.norm(vetor1) * np.linalg.norm(vetor2))

def criar_indice_semantico(licoes):
    print("Criando índice semântico das lições... Aguarde.")
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
    return [licao for _, licao in resultados[:3]]

def responder(pergunta, licoes_encontradas):
    if not licoes_encontradas:
        contexto = "Nenhuma lição aprendida correspondente foi encontrada."
    else:
        contexto = json.dumps(licoes_encontradas, ensure_ascii=False, indent=2)

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
                "content": (
                    f"Pergunta: {pergunta}\n\n"
                    f"Lições recuperadas da base:\n{contexto}"
                )
            }
        ]
    )
    return resposta["message"]["content"]

def main():
    print("=== Assistente de Manufatura (Busca Semântica) ===")
    print("Digite 'sair' para encerrar.\n")
    try:
        licoes = carregar_licoes()
        indice_semantico = criar_indice_semantico(licoes)
    except Exception as erro:
        print(f"Erro: {erro}")
        return

    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() == "sair": break
        if not pergunta: continue
        encontradas = buscar_licoes_semanticas(pergunta, indice_semantico)
        print(f"\nLições recuperadas por similaridade: {len(encontradas)}")
        if not encontradas:
            print("\nAssistente: Não encontrei lições correspondentes.\n")
            continue
        resposta = responder(pergunta, encontradas)
        print(f"\nAssistente: {resposta}\n")

if __name__ == "__main__":
    main()