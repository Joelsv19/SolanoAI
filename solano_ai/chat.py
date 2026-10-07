import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from solano_ai.rag import RAGSystem


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_HISTORY_MESSAGES = 12

# Solo utilizaremos resultados RAG suficientemente relacionados.
RAG_MIN_SCORE = 0.82
RAG_TOP_K = 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "local_data"
MEMORY_FILE = MEMORY_DIR / "user_memory.json"


SSYSTEM_PROMPT = """
Tu nombre es SolanoAI.

DATOS VERDADEROS:
- Fuiste creado por Joel Solano.
- Eres un proyecto personal de ingenieria de software e inteligencia artificial.
- No perteneces a ninguna empresa.
- No fuiste creado por una universidad.
- Utilizas Qwen2.5-0.5B-Instruct como modelo de lenguaje base.

Distingue siempre entre:
- "yo": SolanoAI.
- "tu" o "usuario": la persona que conversa contigo.

REGLAS ESTRICTAS PARA DOCUMENTOS:
- Cuando recibas informacion recuperada desde documentos, debes responder
  unicamente utilizando hechos presentes explicitamente en esos documentos.
- Esta prohibido anadir historias, antecedentes, relaciones o detalles que
  no aparezcan en el contexto recuperado.
- No completes informacion basandote en suposiciones.
- Si el contexto solo dice una cosa sobre una persona o elemento, responde
  unicamente con esa informacion.
- Si el documento no contiene la respuesta, responde:
  "No encuentro esa informacion en mis documentos."
- Se breve cuando respondas utilizando documentos.

Utiliza correctamente la memoria proporcionada.
Si desconoces algo, di que no lo sabes.
"""


def load_model():
    print("Cargando SolanoAI...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


def load_user_memory():
    if not MEMORY_FILE.exists():
        return {}

    try:
        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, OSError):
        print(
            "Advertencia: no se pudo leer la memoria."
        )

    return {}


def save_user_memory(user_memory):
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            user_memory,
            file,
            ensure_ascii=False,
            indent=4
        )


def clear_saved_memory():
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


def check_system_response(user_message):
    message = user_message.lower()

    identity_keywords = [
        "quién eres",
        "quien eres",
        "quién te creó",
        "quien te creo",
        "quién es tu creador",
        "quien es tu creador",
        "quién te hizo",
        "quien te hizo"
    ]

    if any(
        keyword in message
        for keyword in identity_keywords
    ):
        return (
            "Soy SolanoAI, un proyecto de inteligencia "
            "artificial creado por Joel Solano. "
            "Actualmente utilizo Qwen2.5-0.5B-Instruct "
            "como modelo de lenguaje base."
        )

    return None


def update_user_memory(
    user_message,
    user_memory
):
    changed = False

    name_match = re.search(
        r"(?:me llamo|mi nombre es)\s+"
        r"([A-Za-zÁÉÍÓÚáéíóúÑñ]+)",
        user_message,
        re.IGNORECASE
    )

    if name_match:
        name = name_match.group(1)

        if user_memory.get("nombre") != name:
            user_memory["nombre"] = name
            changed = True

    career_match = re.search(
        r"(?:estudio|estoy estudiando)\s+"
        r"(.+?)(?:\.|$)",
        user_message,
        re.IGNORECASE
    )

    if career_match:
        career = career_match.group(1).strip()

        if user_memory.get("carrera") != career:
            user_memory["carrera"] = career
            changed = True

    return changed


def check_memory_response(
    user_message,
    user_memory
):
    message = user_message.lower()

    if (
        "cómo me llamo" in message
        or "como me llamo" in message
        or "cuál es mi nombre" in message
        or "cual es mi nombre" in message
    ):
        if "nombre" in user_memory:
            return (
                f"Te llamas "
                f"{user_memory['nombre']}."
            )

        return (
            "Todavía no me has dicho tu nombre."
        )

    if (
        "qué carrera estudio" in message
        or "que carrera estudio" in message
        or "cuál es mi carrera" in message
        or "cual es mi carrera" in message
    ):
        if "carrera" in user_memory:
            return (
                f"Estudias "
                f"{user_memory['carrera']}."
            )

        return (
            "Todavía no me has dicho "
            "qué carrera estudias."
        )

    if (
        "qué sabes de mí" in message
        or "que sabes de mi" in message
        or "qué recuerdas de mí" in message
        or "que recuerdas de mi" in message
    ):
        if not user_memory:
            return (
                "Todavía no tengo información "
                "guardada sobre ti."
            )

        facts = []

        if "nombre" in user_memory:
            facts.append(
                f"te llamas "
                f"{user_memory['nombre']}"
            )

        if "carrera" in user_memory:
            facts.append(
                f"estudias "
                f"{user_memory['carrera']}"
            )

        return (
            "En mi memoria tengo registrado que "
            + " y ".join(facts)
            + "."
        )

    return None


def get_rag_context(
    rag_system,
    user_message
):
    results = rag_system.search(
        user_message,
        top_k=RAG_TOP_K
    )

    relevant_results = [
        result
        for result in results
        if result["score"] >= RAG_MIN_SCORE
    ]

    if not relevant_results:
        return "", []

    context_parts = []

    sources = []

    for result in relevant_results:
        context_parts.append(
            f"Fuente: {result['source']}\n"
            f"Contenido: {result['text']}"
        )

        if result["source"] not in sources:
            sources.append(
                result["source"]
            )

    context = "\n\n".join(
        context_parts
    )

    return context, sources


def generate_response(
    tokenizer,
    model,
    user_message,
    history,
    user_memory,
    rag_system
):
    system_response = check_system_response(
        user_message
    )

    if system_response is not None:
        return system_response, []

    memory_response = check_memory_response(
        user_message,
        user_memory
    )

    if memory_response is not None:
        return memory_response, []

    rag_context, sources = get_rag_context(
        rag_system,
        user_message
    )

        # Modo RAG seguro:
    # evitamos que el modelo pequeño invente información
    # cuando existe una fuente documental.
    if sources and rag_context:
        top_results = rag_system.search(
            user_message,
            top_k=1
        )

        if top_results:
            best_result = top_results[0]
            document_text = best_result["text"]

            message_lower = user_message.lower()
            document_lower = document_text.lower()

            # Preguntas sobre edad requieren información
            # explícita relacionada con edad.
            if (
                "cuántos años" in message_lower
                or "cuantos años" in message_lower
                or "qué edad" in message_lower
                or "que edad" in message_lower
            ):
                if (
                    "años" not in document_lower
                    and "edad" not in document_lower
                ):
                    return (
                        "No encuentro esa información "
                        "en mis documentos.",
                        sources
                    )

            return document_text, sources

def main():
    tokenizer, model = load_model()

    rag_system = RAGSystem()

    history = []

    user_memory = load_user_memory()

    print("\nSolanoAI V0.4")
    print("Memoria persistente + RAG activados.")

    if user_memory:
        print(
            f"Se recuperaron "
            f"{len(user_memory)} datos de memoria."
        )

    print("\nComandos:")
    print("  salir     -> cerrar")
    print("  /limpiar  -> borrar memoria")
    print("  /memoria  -> mostrar memoria")
    print()

    while True:
        message = input("Tú: ").strip()

        if message.lower() in [
            "salir",
            "exit",
            "quit"
        ]:
            print(
                "SolanoAI: Hasta luego."
            )
            break

        if message.lower() == "/limpiar":
            history.clear()
            user_memory.clear()
            clear_saved_memory()

            print(
                "\nSolanoAI: Conversación y "
                "memoria eliminadas.\n"
            )
            continue

        if message.lower() == "/memoria":
            print(
                "\nMemoria persistente "
                "de SolanoAI:"
            )

            if not user_memory:
                print("Sin datos.")

            for key, value in user_memory.items():
                print(
                    f"- {key}: {value}"
                )

            print()
            continue

        changed = update_user_memory(
            message,
            user_memory
        )

        if changed:
            save_user_memory(
                user_memory
            )

        response, sources = generate_response(
            tokenizer,
            model,
            message,
            history,
            user_memory,
            rag_system
        )

        history.append(
            {
                "role": "user",
                "content": message
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[
                -MAX_HISTORY_MESSAGES:
            ]

        print(
            f"\nSolanoAI: {response}"
        )

        if sources:
            print("\nFuentes:")

            for source in sources:
                print(
                    f"- {source}"
                )

        print()


if __name__ == "__main__":
    main()