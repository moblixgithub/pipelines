from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_DEPLOYMENT_NAME: str
        AZURE_OPENAI_API_VERSION: str
        SEARCH_ENDPOINT: str
        SEARCH_KEY: str

    def __init__(self):
        self.name = "ChatMPF"
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                "SEARCH_ENDPOINT": os.getenv("SEARCH_ENDPOINT", "your-search-endpoint-here"),
                "SEARCH_KEY": os.getenv("SEARCH_KEY", "your-search-key-here"),
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        url = f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.valves.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"

        allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
                          'enhancements', 'data_sources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                          'frequency_penalty', 'logit_bias', 'user', 'function_call', 'functions', 'tools',
                          'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}

        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}
        
        # Adiciona o data_source ao corpo da requisição
        filtered_body["data_sources"] = [{
            "type": "azure_search",
            "parameters": {
                "filter": None,
                "endpoint": self.valves.SEARCH_ENDPOINT,
                "index_name": "index-docs-serpro",
                "semantic_configuration": "azureml-default",
                "authentication": {
                    "type": "api_key",
                    "key": self.valves.SEARCH_KEY
                },
                "embedding_dependency": {
                    "type": "endpoint",
                    "endpoint": f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-07-01-preview",
                    "authentication": {
                        "type": "api_key",
                        "key": "a1304bd8f76d4a88bac275d7e565bd5a"
                    }
                },
                "query_type": "vector_simple_hybrid",
                "in_scope": False,
                "role_information": "ChatMPF é um assistente virtual desenvolvido para o Ministério Público Federal (MPF)...",
                "strictness": 2,
                "top_n_documents": 8
            }
        }]
        
        if len(body) != len(filtered_body):
            print(f"Dropped params: {', '.join(set(body.keys()) - set(filtered_body.keys()))}")

        r = None
        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=True if body.get("stream", False) else False,
            )

            r.raise_for_status()
            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            if r:
                text = r.text
                return f"Error: {e} ({text})"
            else:
                return f"Error: {e}"
