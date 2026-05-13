from typing import Dict, List
import os
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI-compatible chat completions with context."""
    system_prompt = (
        "You are NASA Mission Intelligence, a careful assistant for astronauts, "
        "researchers, historians, and mission operators. Answer only from the "
        "provided NASA mission context when context is available. Cite the source "
        "headers from the context in your answer. If the context does not contain "
        "enough evidence, say what is missing instead of inventing details."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Retrieved NASA mission context follows. Use it as the source "
                    "of truth for the next user question.\n\n"
                    f"{context}"
                ),
            }
        )

    for message in conversation_history[-8:]:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
    client = OpenAI(api_key=openai_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )

    return response.choices[0].message.content or ""
