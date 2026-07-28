"""Centralized model initialization.

The notebooks all import `model` from here, so swapping providers
(OpenAI / Anthropic / Azure / Bedrock) only requires editing this file.

The default is OpenAI, direct. Module 3 §1.4 walks through introducing the
LangSmith LLM Gateway as the production-ready alternative — to make the
switch, comment out the default below and uncomment the gateway block.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4.1-mini"

# LangSmith prices a span from ls_provider / ls_model_name; setting them here
# keeps total_cost populated even when calls are routed through a gateway that
# reports a rewritten model name.
LS_METADATA = {"ls_provider": MODEL_PROVIDER, "ls_model_name": MODEL_NAME}

# --- Default: OpenAI, direct ---
model = init_chat_model(
    f"{MODEL_PROVIDER}:{MODEL_NAME}",
    metadata=LS_METADATA,
)

# --- OpenAI via the LangSmith LLM Gateway (Module 3 §1.4) ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# model = init_chat_model(
#     model="gpt-4.1-mini",
#     model_provider="openai",
#     base_url="https://gateway.smith.langchain.com/openai",
#     api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
#     metadata=LS_METADATA,
# )

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )
