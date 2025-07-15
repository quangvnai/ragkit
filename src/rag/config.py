from typing import Literal

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── LLM provider ──────────────────────────────────────────────────────────
    # "lmstudio" → local Gemma via LM Studio (default)
    # "openai"   → remote OpenAI API
    llm_provider: Literal["lmstudio", "openai"] = "lmstudio"

    # ── LM Studio (local) ─────────────────────────────────────────────────────
    # LM Studio exposes an OpenAI-compatible endpoint at localhost:1234
    lm_studio_base_url: str = "http://localhost:1234/v1"
    # Must match the model name shown in LM Studio's model loader
    lm_studio_model: str = "gemma-3-27b"
    # LM Studio doesn't require a real key, but the client needs a non-empty string
    lm_studio_api_key: str = "lm-studio"

    # ── OpenAI (remote, optional) ─────────────────────────────────────────────
    openai_api_key: str = ""
    default_openai_model: str = "gpt-3.5-turbo"

    # ── LangSmith tracing (optional) ──────────────────────────────────────────
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ── Cohere (optional, Part 15) ────────────────────────────────────────────
    cohere_api_key: str = ""

    # ── Indexing defaults ─────────────────────────────────────────────────────
    chunk_size: int = 300
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Factory helpers ───────────────────────────────────────────────────────

    def get_llm(self, temperature: float = 0, **kwargs) -> ChatOpenAI:
        """Return a ChatOpenAI instance wired to LM Studio or OpenAI."""
        if self.llm_provider == "lmstudio":
            return ChatOpenAI(
                base_url=self.lm_studio_base_url,
                api_key=self.lm_studio_api_key,
                model=self.lm_studio_model,
                temperature=temperature,
                **kwargs,
            )
        return ChatOpenAI(
            api_key=self.openai_api_key,
            model=self.default_openai_model,
            temperature=temperature,
            **kwargs,
        )

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Return an OpenAIEmbeddings instance wired to LM Studio or OpenAI.

        LM Studio also exposes an /embeddings endpoint on the same base URL,
        so we can reuse OpenAIEmbeddings by overriding the base_url.
        """
        if self.llm_provider == "lmstudio":
            return OpenAIEmbeddings(
                base_url=self.lm_studio_base_url,
                api_key=self.lm_studio_api_key,
                # Use whichever embedding model is loaded in LM Studio
                model=self.lm_studio_model,
                check_embedding_ctx_length=False,
            )
        return OpenAIEmbeddings(api_key=self.openai_api_key)


settings = Settings()
