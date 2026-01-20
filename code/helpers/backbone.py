from requests import post
from langchain_openai import ChatOpenAI


class Backbone:
    def __init__(self, model, temperature, n):
        self.model_name = model
        self.temperature = temperature
        self.n = n

    def __call__(self, prompt):
        content = prompt.text
        messages = [{"role": "system", "content": content}]
        if self.model_name == "qwen2.5-7b-instruct":
            utterance = post("http://localhost:7070/get_utterance", json=dict(messages=messages, temperature=self.temperature, n=self.n)).json()
        elif self.model_name == "llama3-8b-instruct":
            utterance = post("http://localhost:7072/get_utterance", json=dict(messages=messages, temperature=self.temperature, n=self.n)).json()
        elif self.model_name == "camel":
            utterance = post("http://localhost:7073/get_utterance", json=dict(messages=messages, temperature=self.temperature, n=self.n)).json()
        elif self.model_name in ["soulchat", "cpsycounx", "mechat", "psychat"]:
            utterance = post("http://localhost:7074/get_utterance", json=dict(messages=messages, temperature=self.temperature, n=self.n)).json()
        elif self.model_name == "ours":
            utterance = post("http://localhost:8080/get_utterance", json=dict(messages=messages, temperature=self.temperature, n=self.n)).json()
        elif "gpt" in self.model_name:
            model = ChatOpenAI(model=self.model_name, temperature=self.temperature, n=self.n)
            if self.n == 1:
                utterance = model.invoke(messages).content
            else:
                utterance = model.generate([messages]).generations[0]
                utterance = [utt.text for utt in utterance]
        else:
            raise NotImplementedError
        return utterance  
