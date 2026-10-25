
import json
from pathlib import Path
import ollama

MODELO = "qwen3:4b"
ARQUIVO_LICOES = Path(__file__).parent / "licoes.json"


def carregar_licoes():
    with open(ARQUIVO_LICOES, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def buscar_licoes(pergunta, licoes):
    termos = set(pergunta.lower().split())
    resultados = []

    for licao in licoes:
        texto = " ".join(str(valor) for valor in licao.values()).lower()
        pontuacao = sum(1 for termo in termos if termo in texto)

        if pontuacao > 0:
            resultados.append((pontuacao, licao))

    resultados.sort(key=lambda item: item[0], reverse=True)
    return [licao for _, licao in resultados[:3]]


def responder(pergunta, licoes_encontradas):
    if not licoes_encontradas:
        contexto = "Nenhuma lição aprendida correspondente foi encontrada."
    else:
        contexto = json.dumps(
            licoes_encontradas,
            ensure_ascii=False,
            indent=2
        )

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
    print("=== Assistente de Manufatura ===")
    print("Digite 'sair' para encerrar.\n")

    try:
        licoes = carregar_licoes()
    except (OSError, json.JSONDecodeError) as erro:
        print(f"Erro ao carregar licoes.json: {erro}")
        return

    while True:
        pergunta = input("Você: ").strip()

        if pergunta.lower() == "sair":
            break

        if not pergunta:
            continue

        encontradas = buscar_licoes(pergunta, licoes)

        print(f"\nLições recuperadas: {len(encontradas)}")

        if not encontradas:
            print(
                "\nAssistente: Não encontrei lições aprendidas "
                "correspondentes a essa pergunta na base de conhecimento. "
                "Não há evidências suficientes para fornecer uma orientação. "
                "Verifique se o equipamento está cadastrado ou consulte "
                "o responsável técnico.\n"
            )
            continue

        resposta = responder(pergunta, encontradas)
        print(f"\nAssistente: {resposta}\n")


if __name__ == "__main__":
    main()
