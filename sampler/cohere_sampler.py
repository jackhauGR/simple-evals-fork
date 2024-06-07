import base64
import json
import os
import time
from typing import Any

import cohere

from ..types import MessageList, SamplerBase

class CohereSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "command-r",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "COHERE_API_KEY"
        self.client = cohere.Client(api_key=os.environ.get(self.api_key_name))
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any):
        if role.lower() == "assistant":
            cohere_role = "CHATBOT"
        elif role.lower() == "user":
            cohere_role = "USER"
        elif role.lower() == "system":
            cohere_role = "SYSTEM"
        return {"role": cohere_role, "content": content}

    def _pack_message_var(self, role: str, content: Any):
        if role.lower() == "assistant":
            cohere_role = "CHATBOT"
        elif role.lower() == "user":
            cohere_role = "USER"
        elif role.lower() == "system":
            cohere_role = "SYSTEM"
        return {"role": cohere_role, "message": content}
    
    def __call__(self, message_list: MessageList) -> tuple[str, int, int]:
        message = message_list[-1].get("content")


        
        chat_history = []
        if self.system_message:
            chat_history.append(self._pack_message_var("system", self.system_message))
        for msg in message_list[:-1]:
            chat_history.append(self._pack_message_var(msg["role"], msg["content"]))
        trial = 0

        print(chat_history)

        while True:
            try:
                response = self.client.chat(
                    model=self.model,
                    message=message,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    chat_history=chat_history
                )
                return response.text, response.meta.tokens.input_tokens, response.meta.tokens.output_tokens
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except cohere.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
