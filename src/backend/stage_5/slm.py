from transformers import pipeline
from openai import OpenAI
import os


class SLM:

    def __init__(self):
        self._distill_bert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self._phi = OpenAI(
            base_url=os.getenv("AZURE_AI_NON_OPENAI_MODEL_ENDPOINT"),
            api_key=os.getenv("AZURE_AI_NON_OPENAI_MODEL_API_KEY")
        )
        


    def distill_bert_analyze_sentiment(self, text: str):
        result =  self._distill_bert(text)
        return result
    

    def phi_4_mini_instruct_analyze_sentiment(self, text: str):

        try:
            response = self._phi.chat.completions.create(
                model="Phi-4-mini-instruct",
                messages=[
                    {"role": "system", "content": """You are a sentiment analysis assistant.
                     return results as 'positive' or 'negative' based on the analysis of the user's texxt and also the confidence score 0 to 1.0
                     """},
                    {"role": "user", "content": f"Analyze the sentiment of the following text: {text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)
    


