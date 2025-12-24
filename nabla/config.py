"""Configuration management with secure API key storage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def get_config_dir() -> Path:
    """Get the configuration directory (~/.nabla/)."""
    config_dir = Path.home() / ".nabla"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    """Get the config file path."""
    return get_config_dir() / "config.json"


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai"  # openai, anthropic, xai, ollama, azure, gemini
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For custom endpoints like Ollama

    @property
    def display_name(self) -> str:
        """Get display name for the config."""
        if self.api_key:
            masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
            return f"{self.model} (key: {masked_key})"
        return f"{self.model} (no key set)"


@dataclass
class HFConfig:
    """HuggingFace configuration."""

    token: Optional[str] = None
    include_all_json: bool = False
    max_files: int = 50

    @property
    def display_name(self) -> str:
        """Get display name for the config."""
        if self.token:
            masked = self.token[:8] + "..." if len(self.token) > 8 else "***"
            return f"Token: {masked}"
        return "No token set (public repos only)"


@dataclass
class NablaConfig:
    """Main application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    last_repos: list[str] = field(default_factory=list)

    def save(self) -> None:
        """Save configuration to disk."""
        config_path = get_config_path()
        data = {
            "llm": asdict(self.llm),
            "hf": asdict(self.hf),
            "last_repos": self.last_repos[-10:],  # Keep last 10
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "NablaConfig":
        """Load configuration from disk."""
        config_path = get_config_path()

        if not config_path.exists():
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                llm=LLMConfig(**data.get("llm", {})),
                hf=HFConfig(**data.get("hf", {})),
                last_repos=data.get("last_repos", []),
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupted config, return default
            return cls()

    def get_effective_api_key(self, provider: str) -> Optional[str]:
        """Get API key from config or environment."""
        # Environment variables take precedence
        env_keys = {
            "openai": "OPENAI_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "xai": "XAI_API_KEY",
            "zai": "ZAI_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "azure": "AZURE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        env_var = env_keys.get(provider)
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]

        # Fall back to stored config
        if self.llm.provider == provider and self.llm.api_key:
            return self.llm.api_key

        return None

    def get_effective_hf_token(self) -> Optional[str]:
        """Get HF token from config or environment."""
        return os.environ.get("HF_TOKEN") or self.hf.token


# Provider configurations
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview", "o1-mini"],
        "env_var": "OPENAI_API_KEY",
        "needs_key": True,
    },
    "openai_compatible": {
        "name": "OpenAI Compatible (Custom)",
        "models": ["Custom..."],
        "env_var": "OPENAI_API_KEY",
        "needs_key": True,
        "needs_base_url": True,
        "default_base": "http://localhost:8000/v1",
    },
    "anthropic": {
        "name": "Anthropic",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        "env_var": "ANTHROPIC_API_KEY",
        "needs_key": True,
    },
    "xai": {
        "name": "xAI (Grok)",
        "models": [
            "xai/grok-4-1-fast-reasoning",
            "xai/grok-4-1-fast-non-reasoning",
            "xai/grok-4",
            "xai/grok-4-fast-reasoning",
            "xai/grok-3",
            "xai/grok-3-mini",
            "xai/grok-code-fast",
            "xai/grok-2",
            "xai/grok-2-vision-latest",
        ],
        "env_var": "XAI_API_KEY",
        "needs_key": True,
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini/gemini-pro", "gemini/gemini-1.5-pro"],
        "env_var": "GEMINI_API_KEY",
        "needs_key": True,
    },
    "zai": {
        "name": "Z.AI International",
        "models": [
            "glm-4.5-flash",
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5v",
            "glm-4.6",
            "glm-4.7",
        ],
        "env_var": "ZAI_API_KEY",
        "needs_key": True,
        "needs_base_url": True,
        "default_base": "https://api.z.ai/api/paas/v4",
        "use_openai_compat": True,
    },
    "zhipu": {
        "name": "Zhipu AI (China)",
        "models": [
            "glm-4-plus",
            "glm-4-flash",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-long",
            "glm-4",
            "glm-4v-plus",
            "glm-4v",
        ],
        "env_var": "ZHIPU_API_KEY",
        "needs_key": True,
        "needs_base_url": True,
        "default_base": "https://open.bigmodel.cn/api/paas/v4",
        "use_openai_compat": True,
    },
    "ollama": {
        "name": "Ollama (Local)",
        "models": ["ollama/llama3.2", "ollama/mistral", "ollama/codellama", "ollama/qwen2.5"],
        "env_var": None,
        "needs_key": False,
        "default_base": "http://localhost:11434",
    },
    "lmstudio": {
        "name": "LM Studio",
        "models": ["Custom..."],
        "env_var": None,
        "needs_key": False,
        "needs_base_url": True,
        "default_base": "http://localhost:1234/v1",
    },
    "azure": {
        "name": "Azure OpenAI",
        "models": ["azure/gpt-4", "azure/gpt-35-turbo"],
        "env_var": "AZURE_API_KEY",
        "needs_key": True,
    },
    "openrouter": {
        "name": "OpenRouter (Free Models)",
        "models": [],  # Fetched dynamically
        "env_var": "OPENROUTER_API_KEY",
        "needs_key": True,
        "fetch_models": True,  # Indicates models should be fetched
    },
}


def fetch_openrouter_free_models() -> list[str]:
    """
    Fetch free models from OpenRouter API.
    
    Returns:
        List of model IDs ending with ':free'
    """
    if not HAS_HTTPX:
        # Fallback to urllib if httpx not available
        import urllib.request
        import ssl
        
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"User-Agent": "Nabla/0.1"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return ["openrouter/auto"]  # Fallback
    else:
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"User-Agent": "Nabla/0.1"}
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return ["openrouter/auto"]  # Fallback
    
    # Filter for free models (ending with :free)
    free_models = []
    for model in data.get("data", []):
        model_id = model.get("id", "")
        if model_id.endswith(":free"):
            # Add openrouter/ prefix for LiteLLM routing
            free_models.append(f"openrouter/{model_id}")
    
    # Sort by model name
    free_models.sort()
    
    return free_models if free_models else ["openrouter/auto"]
