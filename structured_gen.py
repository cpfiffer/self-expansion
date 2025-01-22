import modal
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict

CLIENT = OpenAI(
    base_url="https://txt-ai--cameron-vllm-serve.modal.run/v1/",
    api_key="super-secret-token",
)

MODELS = CLIENT.models.list()
DEFAULT_MODEL = MODELS.data[0].id

print("Using model:", DEFAULT_MODEL)

MAX_TOKENS = 12000

def messages(user: str, system: str = "You are a helpful assistant."):
    ms = [{"role": "user", "content": user}]
    if system:
        ms.insert(0, {"role": "system", "content": system})
    return ms

def generate(
    messages: List[Dict[str, str]],
    response_format: BaseModel,
) -> BaseModel:
    response = CLIENT.beta.chat.completions.parse(
        model=DEFAULT_MODEL,
        messages=messages,
    
        response_format=response_format,
        extra_body={
            # 'guided_decoding_backend': 'outlines',
            "max_tokens": MAX_TOKENS,
        }
    )
    return response

def generate_by_schema(
    messages: List[Dict[str, str]],
    schema: str,
) -> BaseModel:
    response = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={
            # 'guided_decoding_backend': 'outlines',
            "max_tokens": MAX_TOKENS,
            "guided_json": schema,
        }
    )
    return response

def choose(
    messages: List[Dict[str, str]],
    choices: List[str],
) -> BaseModel:
    completion = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={"guided_choice": choices, "max_tokens": MAX_TOKENS},
    )
    return completion.choices[0].message.content

def regex(
    messages: List[Dict[str, str]],
    regex: str,
) -> BaseModel:
    completion = CLIENT.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        extra_body={"guided_regex": regex, "max_tokens": MAX_TOKENS},
    )
    return completion.choices[0].message.content

def embed(content: str) -> List[float]:
    # response = CLIENT_EMBEDDING.embeddings.create(
    #     model=DEFAULT_EMBEDDING_MODEL,
    #     input=content,
    # )
    
    # return response.data[0].embedding
    f = modal.Function.lookup("cameron-embeddings", "embed")
    return f.remote(content)
