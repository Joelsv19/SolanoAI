import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


SYSTEM_PROMPT = """
Tu nombre es SolanoAI.

DATOS VERDADEROS E INMUTABLES:
- Fuiste creado por Joel Solano.
- Eres un proyecto personal de ingeniería de software e inteligencia artificial.
- No perteneces a ninguna empresa.
- No fuiste creado por una universidad.
- No existe un equipo de SolanoAI.
- Actualmente eres un proyecto experimental ejecutado localmente.
- Utilizas Qwen2.5-0.5B-Instruct como modelo de lenguaje base.

Si alguien pregunta quién eres o quién te creó,
debes responder que eres SolanoAI y que fuiste creado por Joel Solano.

No inventes nombres de empresas, universidades,
equipos de desarrollo, organizaciones o creadores.

Si desconoces una información, di que no la sabes.
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
        "quien te hizo",
        "cuál es tu creador",
        "cual es tu creador"
    ]

    if any(keyword in message for keyword in identity_keywords):
        return (
            "Soy SolanoAI, un proyecto de inteligencia artificial "
            "creado por Joel Solano como proyecto personal de "
            "ingeniería de software e inteligencia artificial. "
            "Actualmente utilizo Qwen2.5-0.5B-Instruct como "
            "modelo de lenguaje base."
        )

    return None


def generate_response(tokenizer, model, user_message):
    system_response = check_system_response(user_message)

    if system_response is not None:
        return system_response

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

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

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response


def main():
    tokenizer, model = load_model()

    print("\nSolanoAI V0.1")
    print("Escribe 'salir' para terminar.\n")

    while True:
        message = input("Tú: ")

        if message.lower() in ["salir", "exit", "quit"]:
            print("SolanoAI: Hasta luego.")
            break

        response = generate_response(
            tokenizer,
            model,
            message
        )

        print(f"\nSolanoAI: {response}\n")


if __name__ == "__main__":
    main()