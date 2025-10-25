from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class VirtueLLM():

    def __init__(self):
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=1.0
        )

    def invoke(self, prompt: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        response = self.llm.invoke(messages)
        return response.content