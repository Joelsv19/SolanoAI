import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_HISTORY_MESSAGES = 12

# Ruta donde SolanoAI guardará su memoria privada
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "local_data"
MEMORY_FILE = MEMORY_DIR / "user_memory.json"


SYSTEM_PROMPT = """
Tu nombre es SolanoAI.

DATOS VERDADEROS:
- Fuiste creado por Joel Solano.
- Eres un proyecto personal de ingeniería de software e inteligencia artificial.
- No perteneces a ninguna empresa.
- No fuiste creado por una universidad.
- Utilizas Qwen2.5-0.5B-Instruct como modelo de lenguaje base.

Distingue siempre entre:
- "yo": SolanoAI.
- "tú" o "usuario": la persona que conversa contigo.

Utiliza correctamente la memoria proporcionada.
Si desconoces algo, di que no lo sabes.
"""


def load_model():
    print("Cargando SolanoAI...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, OSError):
        print(
            "Advertencia: no se pudo leer la memoria. "
            "Se iniciará una nueva."
        )

    return {}


def save_user_memory(user_memory):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
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

    if any(keyword in message for keyword in identity_keywords):
        return (
            "Soy SolanoAI, un proyecto de inteligencia artificial "
            "creado por Joel Solano. Actualmente utilizo "
            "Qwen2.5-0.5B-Instruct como modelo de lenguaje base."
        )

    return None


def update_user_memory(user_message, user_memory):
    changed = False

    # Detectar nombre
    name_match = re.search(
        r"(?:me llamo|mi nombre es)\s+([A-Za-zÁÉÍÓÚáéíóúÑñ]+)",
        user_message,
        re.IGNORECASE
    )

    if name_match:
        name = name_match.group(1)

        if user_memory.get("nombre") != name:
            user_memory["nombre"] = name
            changed = True

    # Detectar carrera
    career_match = re.search(
        r"(?:estudio|estoy estudiando)\s+(.+?)(?:\.|$)",
        user_message,
        re.IGNORECASE
    )

    if career_match:
        career = career_match.group(1).strip()

        if user_memory.get("carrera") != career:
            user_memory["carrera"] = career
            changed = True

    return changed


def check_memory_response(user_message, user_memory):
    message = user_message.lower()

    if (
        "cómo me llamo" in message
        or "como me llamo" in message
        or "cuál es mi nombre" in message
        or "cual es mi nombre" in message
    ):
        if "nombre" in user_memory:
            return f"Te llamas {user_memory['nombre']}."

        return "Todavía no me has dicho tu nombre."

    if (
        "qué carrera estudio" in message
        or "que carrera estudio" in message
        or "cuál es mi carrera" in message
        or "cual es mi carrera" in message
    ):
        if "carrera" in user_memory:
            return f"Estudias {user_memory['carrera']}."

        return "Todavía no me has dicho qué carrera estudias."

    if (
        "qué sabes de mí" in message
        or "que sabes de mi" in message
        or "qué recuerdas de mí" in message
        or "que recuerdas de mi" in message
    ):
        if not user_memory:
            return "Todavía no tengo información guardada sobre ti."

        facts = []

        if "nombre" in user_memory:
            facts.append(
                f"te llamas {user_memory['nombre']}"
            )

        if "carrera" in user_memory:
            facts.append(
                f"estudias {user_memory['carrera']}"
            )

        return (
            "En mi memoria tengo registrado que "
            + " y ".join(facts)
            + "."
        )

    return None


def generate_response(
    tokenizer,
    model,
    user_message,
    history,
    user_memory
):
    system_response = check_system_response(user_message)

    if system_response is not None:
        return system_response

    memory_response = check_memory_response(
        user_message,
        user_memory
    )

    if memory_response is not None:
        return memory_response

    memory_text = "\n".join(
        f"- {key}: {value}"
        for key, value in user_memory.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + "\n\nMemoria conocida del usuario:\n"
                + (
                    memory_text
                    if memory_text
                    else "Sin datos todavía."
                )
            )
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


def main():
    tokenizer, model = load_model()

    history = []

    # Recuperar memoria guardada anteriormente
    user_memory = load_user_memory()

    print("\nSolanoAI V0.3")
    print("Memoria persistente activada.")

    if user_memory:
        print(
            f"Se recuperaron "
            f"{len(user_memory)} datos de memoria."
        )

    print("\nComandos:")
    print("  salir     -> cerrar")
    print("  /limpiar  -> borrar conversación y memoria")
    print("  /memoria  -> mostrar memoria\n")

    while True:
        message = input("Tú: ").strip()

        if message.lower() in ["salir", "exit", "quit"]:
            print("SolanoAI: Hasta luego.")
            break

        if message.lower() == "/limpiar":
            history.clear()
            user_memory.clear()
            clear_saved_memory()

            print(
                "\nSolanoAI: Conversación y memoria "
                "persistente eliminadas.\n"
            )
            continue

        if message.lower() == "/memoria":
            print("\nMemoria persistente de SolanoAI:")

            if not user_memory:
                print("Sin datos.")

            for key, value in user_memory.items():
                print(f"- {key}: {value}")

            print()
            continue

        changed = update_user_memory(
            message,
            user_memory
        )

        if changed:
            save_user_memory(user_memory)

        response = generate_response(
            tokenizer,
            model,
            message,
            history,
            user_memory
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
            history = history[-MAX_HISTORY_MESSAGES:]

        print(f"\nSolanoAI: {response}\n")


if __name__ == "__main__":
    main()