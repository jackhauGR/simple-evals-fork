import os
import time
import google
import google.auth
import google.auth.exceptions
import google.generativeai as genai

from ..types import MessageList, SamplerBase

class GoogleSampler(SamplerBase):
    """
    Sampler from Google's Gemini chat completion API
    """

    def __init__(
            self, 
            model: str = "gemini-1.0-pro-latest",
            system_message: str | None = None,
            temperature: float = 1.0,
            max_tokes: int = 1024
    ):
        self.api_key_name = "GOOGLE_API_KEY"

        genai.configure(api_key=os.environ.get(self.api_key_name))
        self.model = model
        self.client = genai.GenerativeModel(self.model)
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokes
    
    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}
    
    
    def __call__(self, message_list: MessageList) -> str:

        trial = 0
        while True:
            try:
                config = genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                prompt = message_list[0].get("content")
                message = self.client.generate_content(prompt, generation_config=config)

                return message.text, message.usage_metadata.prompt_token_count, message.usage_metadata.candidates_token_count
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

            # unknown error shall throw exception