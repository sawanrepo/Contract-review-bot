from typing import List, Dict

class ChatMemory:
    def __init__(self):
        self._messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self._messages)

    def format_history(self, last_n: int = 5) -> str:
        if not self._messages:
            return ""
        recent = self._messages[-last_n:]
        return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)

    def clear(self):
        self._messages.clear()