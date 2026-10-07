from solano_ai.rag import RAGSystem


def main():
    rag = RAGSystem()

    question = "¿Quién es el protagonista de Cenizas del Vacío?"

    print(f"\nPregunta: {question}\n")

    results = rag.search(
        question,
        top_k=3
    )

    if not results:
        print("No se encontraron resultados.")
        return

    for number, result in enumerate(results, start=1):
        print(f"RESULTADO {number}")
        print(f"Fuente: {result['source']}")
        print(f"Similitud: {result['score']:.4f}")
        print(result["text"])
        print("-" * 60)


if __name__ == "__main__":
    main()