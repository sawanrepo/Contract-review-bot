from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

class ChatMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)

    def add_message(self, role: str, content: str):
        self.memory.chat_memory.add_message({"role": role, "content": content})

    def get_context(self) -> BaseMemory:
        return self.memory
    def get_messages(self) -> list:
        return self.memory.chat_memory.messages

    def clear(self):
        self.memory.clear()