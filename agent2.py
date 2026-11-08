
from langchain_openai import ChatOpenAI


class Agent:
    def __init__(self, system_message=""):
        self.system_message = system_message
        self.messages = []
        if self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result=self.execute()
        self.messages.append({"role": "system", "content": result.content})
        return result

    def execute(self):
        llm = ChatOpenAI(
            model="gpt-oss:120b-cloud",  # Or any model you pulled (e.g., mistral)
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Required but ignored
            temperature=0
        )
        return llm.invoke(self.messages)
