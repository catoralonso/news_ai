import os
import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Modelos por agente
MODELS = {
    "fast":    "claude-haiku-4-5-20251001",   # José, Camila, Asti
    "quality": "claude-sonnet-4-6",            # Manuel
}

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_llm(prompt: str | list, system: str, model: str = "fast", max_tokens: int = 4096, temperature: float = 0.5) -> str:
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "assistant" if m["role"] == "model" else m["role"], "content": m["content"]}
            for m in prompt
        ]
        
    kwargs = dict(
        model=MODELS[model],
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    return _client.messages.create(**kwargs).content[0].text