"""SolanoAI V0.6.1: conversaciones mas completas, memoria y RAG documental."""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from solano_ai.rag import RAGSystem


# Modelo de lenguaje. No necesita cuantizacion en una GPU con 6 GB libres.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_HISTORY_MESSAGES = 12
MAX_NEW_TOKENS = 512
RAG_TOP_K = 3
RAG_MIN_SCORE = 0.82  # Valor inicial experimental, no probabilidad.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_FILE = PROJECT_ROOT / "local_data" / "user_memory.json"

SYSTEM_PROMPT = """Eres SolanoAI, un asistente local creado por Joel Solano
como proyecto personal de ingeniería de software e inteligencia artificial.
Usas Qwen2.5-1.5B-Instruct como modelo de lenguaje base.
No perteneces a ninguna empresa ni universidad.

Responde en español, de forma útil y clara.
Al explicar programación, proporciona ejemplos autocontenidos, correctos
y completos; cierra los bloques de código Markdown que abras.
Distingue tus datos de los del usuario; no atribuyas al usuario cosas que
él no haya dicho. Para los ejemplos usa personajes ficticios sin atribuir
al usuario paises de nacimiento, edades ni otras caracteristicas.
Si no conoces un dato, admítelo. No inventes fuentes.
"""


def load_model():
    print(f"Cargando SolanoAI con {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map="auto",
    )
    model.eval()

    print(f"Modelo cargado en: {model.device}")
    if model.device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return tokenizer, model


def load_user_memory():
    if not MEMORY_FILE.is_file():
        return {}

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        print("Aviso: no pude leer el archivo de memoria local.")
        return {}


def save_user_memory(user_memory):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(
        json.dumps(user_memory, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def check_system_response(question):
    text = question.casefold()
    keywords = (
        "quién eres", "quien eres", "quién te creó", "quien te creo",
        "quién te hizo", "quien te hizo", "quién es tu creador",
        "quien es tu creador", "cuál es tu creador",
    )
    if any(keyword in text for keyword in keywords):
        return (
            "Soy SolanoAI, un asistente local creado por Joel Solano. "
            "Mi modelo base es Qwen2.5-1.5B-Instruct."
        )
    return None


def update_user_memory(message, memory):
    changed = False

    name_match = re.search(
        r"(?:me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)",
        message,
        re.IGNORECASE,
    )
    if name_match:
        name = name_match.group(1)
        if memory.get("nombre") != name:
            memory["nombre"] = name
            changed = True

    career_match = re.search(
        r"(?:estudio|estoy estudiando)\s+(.+?)(?:[.!?]|$)",
        message,
        re.IGNORECASE,
    )
    if career_match:
        career = career_match.group(1).strip()
        if memory.get("carrera") != career:
            memory["carrera"] = career
            changed = True

    return changed


def check_memory_response(question, memory):
    text = question.casefold()

    if any(q in text for q in (
        "cómo me llamo", "como me llamo", "cuál es mi nombre",
        "cual es mi nombre",
    )):
        return (
            f"Te llamas {memory['nombre']}."
            if "nombre" in memory else "Todavía no sé tu nombre."
        )

    if any(q in text for q in (
        "qué carrera estudio", "que carrera estudio", "cuál es mi carrera",
        "cual es mi carrera",
    )):
        return (
            f"Estudias {memory['carrera']}."
            if "carrera" in memory else "Todavía no sé qué carrera estudias."
        )

    if any(q in text for q in (
        "qué sabes de mí", "que sabes de mi", "qué recuerdas de mí",
        "que recuerdas de mi",
    )):
        if not memory:
            return "No tengo información personal guardada sobre ti."
        facts = []
        if "nombre" in memory:
            facts.append(f"te llamas {memory['nombre']}")
        if "carrera" in memory:
            facts.append(f"estudias {memory['carrera']}")
        return "Recuerdo que " + " y ".join(facts) + "." if facts else (
            "No tengo información personal compatible con esta pregunta."
        )

    return None


def extract_rag_answer(question, result):
    """Devuelve extractos de la fuente, sin pedir al LLM que improvise."""
    text = result["text"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    query = question.casefold()

    if any(term in query for term in (
        "tres posibles eventos", "3 posibles eventos", "tres eventos",
        "3 eventos", "nombra los eventos",
    )):
        events = [
            line for line in lines
            if any(f"evento {letter}:" in line.casefold() for letter in "abc")
        ]
        if events:
            return "\n".join(events)

    for letter in "abc":
        if f"evento {letter}" in query:
            for line in lines:
                if f"evento {letter}:" in line.casefold():
                    return line

    if "probabilidad conjunta" in query or "b y c" in query:
        for line in lines:
            if "b y c" in line.casefold() and "0,05" in line:
                return line

    # No sabemos inferir una respuesta a cualquier pregunta de forma segura:
    # mostramos un extracto textual para no atribuir detalles inventados.
    return text


def answer_with_rag(question, rag_system):
    results = rag_system.search(question, top_k=RAG_TOP_K)
    relevant = [item for item in results if item["score"] >= RAG_MIN_SCORE]
    if not relevant:
        return None

    best = relevant[0]
    source = best["source"]
    if best.get("page") is not None:
        source += f" — página {best['page']}"

    # Una pregunta sobre la edad necesita un dato explicito sobre esa edad.
    age_question = any(k in question.casefold() for k in (
        "cuántos años", "cuantos años", "qué edad", "que edad",
    ))
    if age_question:
        age_statement = re.search(
            r"\b\d{1,3}\s+años\b", best["text"], re.IGNORECASE
        )
        if not age_statement:
            return "No encuentro esa información en mis documentos.", [source]

    return extract_rag_answer(question, best), [source]


def answer_with_model(tokenizer, model, question, history, memory):
    facts = "\n".join(f"- {key}: {value}" for key, value in memory.items())
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT + "\nDatos confirmados del usuario:\n" +
        (facts or "No hay datos guardados."),
    }]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if new_tokens.shape[0] >= MAX_NEW_TOKENS:
        response += (
            "\n\n[Aviso: se alcanzo el limite de la respuesta. "
            "Puedes pedir: /chat Continua la explicacion anterior.]"
        )
    return response or "No pude generar una respuesta.", []


def generate_response(tokenizer, model, question, history, memory, rag_system,
                      skip_rag=False):
    system_answer = check_system_response(question)
    if system_answer is not None:
        return system_answer, []

    memory_answer = check_memory_response(question, memory)
    if memory_answer is not None:
        return memory_answer, []

    if not skip_rag:
        rag_answer = answer_with_rag(question, rag_system)
        if rag_answer is not None:
            return rag_answer

    return answer_with_model(tokenizer, model, question, history, memory)


def show_documents(rag_system):
    documents = rag_system.get_documents_info()
    print("\nDocumentos cargados:")
    if not documents:
        print("No hay documentos cargados.")
    for doc in documents:
        page_info = f" | {doc['pages']} páginas" if doc["pages"] else ""
        print(f"- {doc['source']}{page_info} | {doc['chunks']} fragmentos")
    print()


def main():
    tokenizer, model = load_model()
    rag_system = RAGSystem()
    history = []
    memory = load_user_memory()

    print("\nSolanoAI V0.6.1")
    print("Memoria persistente + RAG + PDF/TXT activados.")
    print(f"Se recuperaron {len(memory)} datos de memoria.")
    print("\nComandos:")
    print("  salir        -> cerrar")
    print("  /limpiar     -> borrar conversación y memoria guardada")
    print("  /memoria     -> mostrar datos recordados")
    print("  /documentos  -> mostrar documentos cargados")
    print("  /recargar    -> recargar los documentos")
    print("  /modelo      -> mostrar el modelo y el dispositivo")
    print("  /chat TEXTO  -> preguntar directamente al LLM, sin RAG\n")

    while True:
        message = input("Tú: ").strip()
        if not message:
            continue
        command = message.casefold()

        if command in ("salir", "exit", "quit"):
            print("SolanoAI: Hasta luego.")
            break

        if command == "/limpiar":
            history.clear()
            memory.clear()
            if MEMORY_FILE.exists():
                MEMORY_FILE.unlink()
            print("\nSolanoAI: Conversación y memoria borradas.\n")
            continue

        if command == "/memoria":
            print("\nMemoria persistente de SolanoAI:")
            if memory:
                for key, value in memory.items():
                    print(f"- {key}: {value}")
            else:
                print("Sin datos guardados.")
            print()
            continue

        if command == "/documentos":
            show_documents(rag_system)
            continue

        if command == "/recargar":
            print("\nRecargando documentos...")
            documents = rag_system.reload_documents()
            print(f"SolanoAI: documentos disponibles: {len(documents)}\n")
            continue

        if command == "/modelo":
            print(f"\nModelo: {MODEL_NAME}")
            print(f"Dispositivo: {model.device}")
            print(f"Limite de salida: {MAX_NEW_TOKENS} tokens\n")
            continue

        skip_rag = command.startswith("/chat ")
        question = message[6:].strip() if skip_rag else message
        if not question:
            continue

        if update_user_memory(question, memory):
            save_user_memory(memory)

        response, sources = generate_response(
            tokenizer, model, question, history, memory, rag_system,
            skip_rag=skip_rag,
        )

        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ])
        history = history[-MAX_HISTORY_MESSAGES:]

        print(f"\nSolanoAI: {response}")
        if sources:
            print("\nFuentes:")
            for source in sources:
                print(f"- {source}")
        print()


if __name__ == "__main__":
    main()
