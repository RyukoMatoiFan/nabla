"""Prompt templates for HuggingFace model capability analysis."""

SYSTEM_PROMPT = """You are generating a human-readable summary from an existing
STRICT factual analysis of a HuggingFace model.

You MUST:
- Use ONLY information present in the provided analysis JSON.
- Rephrase facts, do NOT infer new capabilities or suitability.
- If information is unknown, state it as "Unknown".

You must NOT:
- Guess performance or quality
- Infer use cases beyond stated capabilities

"""


ANALYSIS_PROMPT_TEMPLATE = """Analyze the following HuggingFace model configuration artifacts.

{config_text}

Produce a STRICT factual analysis using the JSON schema below.

For EACH field:
- Provide the value
- Provide a justification object with:
  - "source": file name + JSON path OR "not_found"
  - "confidence": "explicit" | "derived" | "unknown" || "formula"

If a value cannot be determined strictly from the artifacts,
set the value to null and confidence to "unknown".

OUTPUT SCHEMA (do NOT add or remove fields):

{{
  "model_identity": {{
    "model_name": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }},
    "repo_id": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }}
  }},

  "architecture": {{
    "architecture_class": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }},
    "pretrained_config_architecture": {{
       "value": null,
       "confidence": "unknown",
       "source": "not_found"
    }}
  }},

  "parameter_analysis": {{
    "vocab_size": {{
        "value": null,
        "confidence": "unknown",
        "source": "not_found"
    }},
    "d_model": {{
        "value": null,
        "confidence": "unknown",
        "source": "not_found"
    }},
    "num_layers": {{
        "value": null,
        "confidence": "unknown",
        "source": "not_found"
    }},
    "num_heads": {{
        "value": null,
        "confidence": "unknown",
        "source": "not_found"
    }}
  }},

  "context_window": {{
    "max_position_embeddings": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }},
    "sliding_window": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }}
  }},

  "modalities": {{
    "visual_encoder_type": {{ "value": null, "confidence": "unknown", "source": "not_found" }},
    "audio_encoder_type": {{ "value": null, "confidence": "unknown", "source": "not_found" }},
    "video_support_detected": {{ "value": null, "confidence": "unknown", "source": "not_found" }}
  }},

  "special_tokens": {{
    "chat_template_exists": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }},
    "eos_token": {{
      "value": null,
      "confidence": "unknown",
      "source": "not_found"
    }}
  }}
}}

IMPORTANT CONSTRAINTS:
- Do NOT include "recommended use cases".
- Do NOT describe performance, quality, or suitability.
- Do NOT include assumptions about training data.
- Output ONLY valid JSON. No prose. No markdown.
"""


ROUTER_PROMPT = """You are a Model Type Router.
Your goal is to determine the primary architecture type of a model based on its configuration files.

{config_text}

Analyze the configurations (architectures, model_type, special tokens, processors) to determine the model type.

Possible `primary_type` values:
- text_only_lm
- image_text_vlm
- video_text_vlm
- text_to_video_generator
- image_to_video_generator
- audio_text_model
- multimodal_general
- unknown

OUTPUT format (JSON):
{{
  "primary_type": "...",
  "secondary_types": ["..."],
  "evidence": [
    "file -> json.path -> reason"
  ]
}}

Rules:
- Rely ONLY on explicit signals (e.g. `LlavaForConditionalGeneration` -> image_text_vlm).
- If `video` word appears in processor configs or architectures, check if it is generation or understanding.
- Do not guess based on model name alone.
"""


COMPARISON_PROMPT_TEMPLATE = """You are comparing multiple STRICT model analyses produced under the same schema.

{analysis_jsons}

Compare ONLY fields with confidence != "unknown".

OUTPUT FORMAT:

{{
  "models": [
    {{
      "repo_id": "string",
      "explicit_capabilities": ["list of confirmed features"],
      "unknown_areas": ["capabilities not determinable from configs"],
      "structural_differences": ["verifiable architectural differences"]
    }}
  ],
  "comparison_constraints": [
    "Statements that cannot be made due to missing data"
  ]
}}

Rules:
- Do NOT declare winners.
- Do NOT infer suitability.
- Prefer stating uncertainty over ranking.
"""


QUICK_SUMMARY_PROMPT = """Based ONLY on the following STRICT model analysis JSON:

{analysis_json}

Produce a concise summary in this format:

- Model: <name or repo_id>
- Architecture: <architecture_class or "Unknown">
- Modalities: <comma-separated confirmed modalities>
- Context Length: <tokens or "Unknown">
- Scale: <parameter count or "Unknown">
- Notable Structural Features: <confirmed features only>
- Known Constraints: <list or "Not determinable from configs">

Rules:
- Do NOT include "Best for"
- Do NOT infer tasks or domains
- Use "Unknown" where appropriate
- Keep the summary factual and compact

"""



# =============================================================================
# STEP 2: TYPE-SPECIFIC INTERPRETERS
# =============================================================================

INTERPRETER_SYSTEM_PROMPT = """You are an expert model architecture analyst.
You do NOT extract new facts.
You ONLY interpret already verified architectural parameters.
"""

BASE_INTERPRETER_TEMPLATE = """ROLE:
You are an expert {model_type} architecture analyst.
You do NOT extract new facts.
You ONLY interpret already verified architectural parameters.

INPUT:
I will provide you with a structured analysis of a {model_type} Model.
Each field has:
- value
- confidence: explicit | derived | unknown
- source: file -> path

STRICT RULES:
1. Use ONLY fields with confidence = explicit or derived.
2. Do NOT add any new numbers or assumptions.
3. Do NOT use phrases like "usually", "likely", "in practice".
4. Every conclusion must reference the exact fields it is based on.
5. If information is insufficient, state this explicitly.

TASK:
Interpret the architecture for {model_type} specifically.

ANALYSIS AXES:
{axes}

Here is the factual analysis to interpret:

{analysis_json}

OUTPUT FORMAT:

### Architectural Conclusions (5 bullets)

Each bullet:
- 1-2 sentences
- plain technical language
- format:
  Conclusion sentence.
  (Based on: field_A, field_B)

FORBIDDEN:
- quality claims ("good", "better")
- training speculation
- references to other models
"""

TEXT_LM_INTERPRETER = BASE_INTERPRETER_TEMPLATE.format(
    model_type="Text-Only LLM",
    analysis_json="{analysis_json}",
    axes="""
- context handling (short vs long context mechanisms)
- memory vs compute tradeoffs (GQA, MoE)
- positional encoding constraints
- vocabulary and tokenization features
"""
)

IMAGE_TEXT_VLM_INTERPRETER = BASE_INTERPRETER_TEMPLATE.format(
    model_type="Image-Text VLM",
    analysis_json="{analysis_json}",
    axes="""
- visual encoder integration
- resolution handling
- cross-attention mechanisms
- multimodal connector type
"""
)

VIDEO_TEXT_VLM_INTERPRETER = BASE_INTERPRETER_TEMPLATE.format(
    model_type="Video-Text VLM",
    analysis_json="{analysis_json}",
    axes="""
- spatial density
- temporal density
- vision budget behavior
- annotation granularity
- limitations for frame-level tasks
"""
)

T2V_INTERPRETER = BASE_INTERPRETER_TEMPLATE.format(
    model_type="Text-to-Video Generator",
    analysis_json="{analysis_json}",
    axes="""
- max duration mechanics
- resolution limits
- temporal coherence constraints
- latent vs pixel diffusion
"""
)

MULTIMODAL_INTERPRETER = BASE_INTERPRETER_TEMPLATE.format(
    model_type="Multimodal General Model",
    analysis_json="{analysis_json}",
    axes="""
- modality fusion strategy
- shared vs separate backbones
- unified vocabulary characteristics
"""
)

