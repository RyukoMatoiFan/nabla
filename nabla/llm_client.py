"""Abstracted LLM client using LiteLLM for multi-provider support."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from pydantic import BaseModel, model_validator

# Suppress LiteLLM debug info messages
litellm.suppress_debug_info = True

# Default retry configuration
DEFAULT_RETRIES = 0  # No retry by default
RETRY_DELAY = 2  # seconds



class JustifiedValue(BaseModel):
    """A value with its justification (flat structure)."""
    value: Any = None
    confidence: str = "unknown"  # Allow any string, normalize in validator
    source: str = "not_found"
    
    @model_validator(mode='before')
    @classmethod
    def handle_raw_values(cls, data):
        """Handle cases where LLM returns raw value instead of structured object."""
        if data is None:
            return {"value": None, "confidence": "unknown", "source": "not_found"}
        if not isinstance(data, dict):
            # LLM returned a raw value (int, str, bool, etc.) instead of structure
            return {"value": data, "confidence": "derived", "source": "llm_raw"}
        
        # Normalize confidence to valid values
        if isinstance(data, dict):
            conf = data.get("confidence", "unknown")
            if conf not in ("explicit", "derived", "unknown", "formula"):
                data["confidence"] = "unknown"
        return data


class ModelIdentity(BaseModel):
    model_name: JustifiedValue = JustifiedValue()
    repo_id: JustifiedValue = JustifiedValue()



class Architecture(BaseModel):
    architecture_class: JustifiedValue = JustifiedValue()
    pretrained_config_architecture: JustifiedValue = JustifiedValue()


class ParameterAnalysis(BaseModel):
    vocab_size: JustifiedValue = JustifiedValue()
    d_model: JustifiedValue = JustifiedValue()
    num_layers: JustifiedValue = JustifiedValue()
    num_heads: JustifiedValue = JustifiedValue()


class ContextWindow(BaseModel):
    max_position_embeddings: JustifiedValue = JustifiedValue()
    sliding_window: JustifiedValue = JustifiedValue()


class Modalities(BaseModel):
    visual_encoder_type: JustifiedValue = JustifiedValue()
    audio_encoder_type: JustifiedValue = JustifiedValue()
    video_support_detected: JustifiedValue = JustifiedValue()


class SpecialTokens(BaseModel):
    chat_template_exists: JustifiedValue = JustifiedValue()
    eos_token: JustifiedValue = JustifiedValue()


class ModelCapabilities(BaseModel):
    """Structured output for model capability analysis with justifications."""
    
    model_identity: ModelIdentity = ModelIdentity()
    architecture: Architecture = Architecture()
    parameter_analysis: ParameterAnalysis = ParameterAnalysis()
    context_window: ContextWindow = ContextWindow()
    modalities: Modalities = Modalities()
    special_tokens: SpecialTokens = SpecialTokens()


@dataclass
class LLMResponse:
    """Response from LLM inference."""

    content: str
    model: str
    usage: dict[str, int]
    raw_response: Any = None


class LLMClient:
    """Abstracted LLM client supporting multiple providers via LiteLLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        retries: int = DEFAULT_RETRIES,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-opus", "ollama/llama2")
            api_key: API key (or set via environment variable)
            api_base: Custom API base URL (for local/OpenAI-compatible models)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            retries: Number of retry attempts on failure (default: 0)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

        # For OpenAI-compatible endpoints with custom base URL,
        # ensure model uses openai/ prefix for LiteLLM routing
        if api_base and not any(model.startswith(p) for p in [
            "openai/", "ollama/", "azure/", "gemini/", "anthropic/", "xai/", "openrouter/"
        ]):
            # Use openai/ prefix for custom OpenAI-compatible endpoints
            self.model = f"openai/{model}"

        # Set API key if provided
        if api_key:
            # For custom base URLs, always set OPENAI_API_KEY
            if api_base:
                os.environ["OPENAI_API_KEY"] = api_key
            # Detect provider from model name and set appropriate env var
            elif model.startswith("gpt") or model.startswith("o1"):
                os.environ["OPENAI_API_KEY"] = api_key
            elif model.startswith("claude"):
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif model.startswith("gemini"):
                os.environ["GEMINI_API_KEY"] = api_key
            elif model.startswith("xai"):
                os.environ["XAI_API_KEY"] = api_key
            elif model.startswith("zai"):
                os.environ["ZAI_API_KEY"] = api_key
            elif model.startswith("openrouter"):
                os.environ["OPENROUTER_API_KEY"] = api_key

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a completion request to the LLM with automatic retry.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to litellm

        Returns:
            LLMResponse with the model's response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        last_error = None
        max_attempts = self.retries + 1  # retries=0 means 1 attempt
        for attempt in range(max_attempts):
            try:
                # Build completion kwargs
                completion_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **kwargs,
                }
                
                # Only pass api_base for non-OpenRouter models
                # OpenRouter is handled internally by LiteLLM
                if self.api_base and not self.model.startswith("openrouter/"):
                    completion_kwargs["api_base"] = self.api_base
                
                response = litellm.completion(**completion_kwargs)

                # Validate response
                content = response.choices[0].message.content
                if content is None:
                    content = ""

                return LLMResponse(
                    content=content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                    raw_response=response,
                )
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    # Wait before retry with exponential backoff
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                raise last_error

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_schema: Optional[dict] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a completion request expecting JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_schema: Optional JSON schema for structured output
            **kwargs: Additional arguments

        Returns:
            LLMResponse with JSON content
        """
        # Add JSON instruction to system prompt
        json_instruction = "\nRespond ONLY with valid JSON. No markdown, no code blocks, just raw JSON."
        if system_prompt:
            system_prompt = system_prompt + json_instruction
        else:
            system_prompt = json_instruction.strip()

        # Try to use native JSON mode if supported
        extra_kwargs = {}
        if self.model.startswith("gpt-4") or self.model.startswith("gpt-3.5"):
            extra_kwargs["response_format"] = {"type": "json_object"}

        return self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            **extra_kwargs,
            **kwargs,
        )

    async def acomplete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Async version of complete."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Build completion kwargs
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        
        # Only pass api_base for non-OpenRouter models
        if self.api_base and not self.model.startswith("openrouter/"):
            completion_kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(**completion_kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw_response=response,
        )
