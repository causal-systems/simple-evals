import os
import time
import requests
import json
from ..types import MessageList, SamplerBase

class GraderAblationSampler(SamplerBase):
    """
    Sample from GR API
    """

    def __init__(
        self,
        temperature: float = 0.0,  # default in Anthropic example
        max_tokens: int = 8192,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "base64"
        self.num_tokens = 0
        self.num_calls = 0

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        raise NotImplementedError("Image is not supported for Claude")

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        trial = 0
        while True:
            try:
                # Prepare the request payload
                payload = {
                    "messages": message_list,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }

                # Make POST request to local endpoint
                port = os.getenv("PORT")
                response = requests.post(
                    f"http://localhost:{port}/answer",
                    headers={
                        "accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )
                
                # Raise exception for bad status codes
                response.raise_for_status()
                
                # Parse and return the response
                result = response.json()
                self.num_tokens += result["num_tokens"]
                self.num_calls += 1
                return result["answer"]  # Adjust this based on actual response structure
                
            except requests.exceptions.RequestException as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Request failed, retrying {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
                
            # Let other exceptions propagate up