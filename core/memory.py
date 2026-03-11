"""
core/memory.py
──────────────
Historial de conversación con ventana deslizante.
Diseñado para Google ADK / google-genai types.

TODO (producción): persistencia en SQLite o Firestore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Turn:
    role: str           # "user" | "model"
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Memory:
    """
    Almacena el historial de chat como lista de Turn.
    Mantiene una ventana de los últimos `max_turns` turnos para no
    superar el context window del modelo.

    Uso:
        mem = Memory(max_turns=20)
        mem.add("user", "¿Qué noticias hay hoy?")
        mem.add("model", "Aquí tienes las tendencias...")
        history = mem.as_messages()   # lista lista para la API
    """

    def __init__(self, max_turns: int = 20):
        self._turns: list[Turn] = []
        self.max_turns = max_turns

    # ── Escritura ─────────────────────────────────────────────────────────────

    def add(self, role: str, text: str) -> None:
        self._turns.append(Turn(role=role, text=text))

    def clear(self) -> None:
        self._turns = []

    # ── Lectura ───────────────────────────────────────────────────────────────

    def as_messages(self) -> list[dict]:
        """
        Devuelve el historial como lista de dicts {role, content}
        compatible con la mayoría de SDKs (OpenAI, Google GenAI, LangChain).
        Aplica ventana deslizante.
        """
        window = self._turns[-self.max_turns :]
        return [{"role": t.role, "content": t.text} for t in window]

    def last_n(self, n: int = 3) -> list[Turn]:
        return self._turns[-n:]

    def __len__(self) -> int:
        return len(self._turns)
