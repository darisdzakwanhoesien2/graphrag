# modules/llm_interface.py

import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ------------------------------------------------
# LMSTUDIO
# ------------------------------------------------

def call_lmstudio(prompt, model, temperature=0.2):

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an ESG analyst. Answer grounded strictly in context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(LMSTUDIO_URL, json=payload)

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON from LMStudio: {response.text}")

    if response.status_code != 200:
        raise RuntimeError(f"LMStudio Error: {data}")

    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unexpected LMStudio response format: {data}")


# ------------------------------------------------
# OPENROUTER
# ------------------------------------------------

def call_openrouter(prompt, model, temperature=0.2):

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an ESG analyst. Answer grounded strictly in context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON from OpenRouter: {response.text}")

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter Error: {data}")

    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unexpected OpenRouter response format: {data}")


# ------------------------------------------------
# UNIFIED ENTRY POINT
# ------------------------------------------------

def generate_response(prompt, provider, model, temperature=0.2):

    if provider == "lmstudio":
        return call_lmstudio(prompt, model, temperature)

    elif provider == "openrouter":
        return call_openrouter(prompt, model, temperature)

    else:
        raise ValueError("Unsupported LLM provider")


# import requests
# import os

# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# # -----------------------------------
# # OpenRouter
# # -----------------------------------

# def call_openrouter(prompt, model="mistralai/mistral-7b-instruct"):
#     url = "https://openrouter.ai/api/v1/chat/completions"

#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ]
#     }

#     response = requests.post(url, headers=headers, json=payload)
#     return response.json()["choices"][0]["message"]["content"]


# # -----------------------------------
# # LMStudio (Local)
# # -----------------------------------

# # modules/llm_interface.py

# import requests


# LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
# MODEL_NAME = "your-model-name"  # change this


# def call_lmstudio(prompt, temperature=0.2):
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [
#             {"role": "system", "content": "You are an ESG analyst. Answer grounded in context."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": temperature
#     }

#     response = requests.post(LMSTUDIO_URL, json=payload)

#     try:
#         data = response.json()
#     except Exception:
#         raise RuntimeError(f"Invalid JSON response: {response.text}")

#     if response.status_code != 200:
#         raise RuntimeError(f"LMStudio error: {data}")

#     # OpenAI-compatible format
#     if "choices" in data:
#         return data["choices"][0]["message"]["content"]

#     # Fallback formats
#     if "content" in data:
#         return data["content"]

#     if "response" in data:
#         return data["response"]

#     raise ValueError(f"Unexpected LM response format: {data}")



# # def call_lmstudio(prompt, model="local-model"):
# #     url = "http://localhost:1234/v1/chat/completions"

# #     payload = {
# #         "model": model,
# #         "messages": [
# #             {"role": "user", "content": prompt}
# #         ]
# #     }

# #     response = requests.post(url, json=payload)
# #     return response.json()["choices"][0]["message"]["content"]


# # -----------------------------------
# # Unified Interface
# # -----------------------------------

# def generate_response(prompt, provider="lmstudio"):
#     if provider == "lmstudio":
#         return call_lmstudio(prompt)
#     else:
#         raise ValueError("Unsupported provider")


# def generate_response(prompt, provider="openrouter"):

#     if provider == "openrouter":
#         return call_openrouter(prompt)

#     elif provider == "lmstudio":
#         return call_lmstudio(prompt)

#     else:
#         raise ValueError("Unsupported LLM provider")
