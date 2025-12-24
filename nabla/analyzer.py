"""Main analyzer orchestrating config collection and LLM inference."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable

from rich.console import Console

from .config_collector import ConfigCollector, RepoConfigs
from .llm_client import LLMClient, ModelCapabilities
from .prompts import (
    SYSTEM_PROMPT,
    ANALYSIS_PROMPT_TEMPLATE,
    COMPARISON_PROMPT_TEMPLATE,
    QUICK_SUMMARY_PROMPT,
    ROUTER_PROMPT,
    INTERPRETER_SYSTEM_PROMPT,
    TEXT_LM_INTERPRETER,
    IMAGE_TEXT_VLM_INTERPRETER,
    VIDEO_TEXT_VLM_INTERPRETER,
    T2V_INTERPRETER,
    MULTIMODAL_INTERPRETER,
)

console = Console()


@dataclass
class AnalysisResult:
    """Complete analysis result for a HuggingFace model."""

    repo_id: str
    configs: RepoConfigs
    model_type: Optional[str] = None  # NEW: Step 0 result
    router_evidence: Optional[list] = None  # NEW: Step 0 evidence
    capabilities: Optional[ModelCapabilities] = None
    raw_response: Optional[str] = None
    llm_usage: Optional[dict] = None
    interpreted_analysis: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "repo_id": self.repo_id,
            "model_type": self.model_type,
            "router_evidence": self.router_evidence,
            "configs": self.configs.to_dict(),
            "capabilities": self.capabilities.model_dump() if self.capabilities else None,
            "raw_response": self.raw_response,
            "llm_usage": self.llm_usage,
            "interpreted_analysis": self.interpreted_analysis,
            "error": self.error,
        }


class HFAnalyzer:
    """Orchestrates HuggingFace model analysis using config collection and LLM inference."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config_collector: Optional[ConfigCollector] = None,
        hf_token: Optional[str] = None,
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            llm_client: LLM client for inference (creates default if None)
            config_collector: Config collector (creates default if None)
            hf_token: HuggingFace token for gated repos
            verbose: If True, print detailed progress
            log_callback: Optional callback for log messages
        """
        self.llm = llm_client or LLMClient()
        self.collector = config_collector or ConfigCollector(token=hf_token, verbose=verbose)
        self.verbose = verbose
        self.log_callback = log_callback

    def _log(self, message: str, style: str = "") -> None:
        """Log a message."""
        if self.verbose:
            if style:
                console.print(message, style=style)
            else:
                console.print(message)
        if self.log_callback:
            self.log_callback(message)

    def analyze(self, repo_id: str) -> AnalysisResult:
        """
        Analyze a single HuggingFace model repository.

        Args:
            repo_id: HuggingFace repository ID (e.g., "microsoft/phi-2")

        Returns:
            AnalysisResult with capabilities and metadata
        """
        self._log(f"\n[bold blue]━━━ Analyzing: {repo_id} ━━━[/bold blue]\n")
        
        # Collect configs
        self._log("[bold]Step 1:[/bold] Collecting configuration files...")
        configs = self.collector.collect(repo_id)

        if configs.error:
            self._log(f"[red]✗ Error: {configs.error}[/red]")
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"Config collection failed: {configs.error}",
            )

        if not configs.configs:
            self._log("[yellow]⚠ No configuration files found in repository[/yellow]")
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error="No configuration files found in repository",
            )


        # Step 0: Model Type Router
        self._log(f"\n[bold]Step 0:[/bold] Routing model type...")
        config_text = configs.to_prompt_text()
        
        # We use a separate router call
        router_result = self._determine_model_type(config_text)
        primary_type = router_result.get("primary_type", "unknown")
        self._log(f"  Type identified: [bold cyan]{primary_type}[/bold cyan]")

        # Step 1: Strict Factual Extraction
        self._log(f"\n[bold]Step 1:[/bold] Extracting facts ({self.llm.model})...")
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(config_text=config_text)
        
        try:
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
            
            self._log(f"  [green]✓ Facts extracted[/green]")

            # Step 1.5: Parse capabilities (Factual Analysis)
            capabilities = self._parse_capabilities(response.content)
            
            # Step 2: Type-Specific Interpretation
            self._log(f"\n[bold]Step 2:[/bold] Interpreting for {primary_type}...")
            interpreted_analysis = self._interpret_analysis(response.content, primary_type)
            
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                model_type=primary_type,
                router_evidence=router_result.get("evidence", []),
                capabilities=capabilities,
                raw_response=response.content,
                interpreted_analysis=interpreted_analysis,
                llm_usage=response.usage,
            )

        except Exception as e:
            self._log(f"  [red]✗ Error: {type(e).__name__}: {e}[/red]")
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"LLM analysis failed: {type(e).__name__}: {e}",
            )

    def analyze_many(self, repo_ids: list[str]) -> list[AnalysisResult]:
        """Analyze multiple repositories."""
        return [self.analyze(repo_id) for repo_id in repo_ids]

    def analyze_from_configs(self, repo_id: str, configs: RepoConfigs) -> AnalysisResult:
        """
        Analyze a model using pre-collected configs.

        Args:
            repo_id: HuggingFace repository ID
            configs: Pre-collected RepoConfigs

        Returns:
            AnalysisResult with capabilities and metadata
        """
        self._log(f"\n[bold blue]━━━ Analyzing: {repo_id} ━━━[/bold blue]\n")

        if configs.error:
            self._log(f"[red]✗ Error: {configs.error}[/red]")
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"Config collection failed: {configs.error}",
            )

        # Count selected files
        selected = [c for c in configs.configs if c.selected]
        if not selected and not configs.include_readme:
            self._log("[yellow]⚠ No files selected for analysis[/yellow]")
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error="No files selected for analysis",
            )

        # Format configs for LLM
        config_text = configs.to_prompt_text(selected_only=True)
        
        # Step 0: Model Type Router
        self._log(f"[bold]Step 0:[/bold] Routing model type...")
        router_result = self._determine_model_type(config_text)
        primary_type = router_result.get("primary_type", "unknown")
        self._log(f"  Type identified: [bold cyan]{primary_type}[/bold cyan]")

        # Step 1: Strict Factual Extraction
        self._log(f"[bold]Step 1:[/bold] Extracting facts ({self.llm.model})...")
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(config_text=config_text)
        
        try:
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
            
            self._log(f"  [green]✓ Response received[/green]")
            if response.usage:
                self._log(f"  Tokens used: {response.usage.get('total_tokens', 'N/A')}", "dim")

            # Parse response into structured format
            self._log(f"Parsing LLM response...")
            capabilities = self._parse_capabilities(response.content)
            self._log(f"  [green]✓ Successfully parsed capabilities[/green]")
            
            # Step 2: Type-Specific Interpretation
            self._log(f"[bold]Step 2:[/bold] Interpreting for {primary_type}...")
            interpreted_analysis = self._interpret_analysis(response.content, primary_type)
            
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                model_type=primary_type,
                router_evidence=router_result.get("evidence", []),
                capabilities=capabilities,
                raw_response=response.content,
                interpreted_analysis=interpreted_analysis,
                llm_usage=response.usage,
            )

        except Exception as e:
            self._log(f"  [red]✗ Error: {type(e).__name__}: {e}[/red]")
            if 'response' in locals() and response and response.content:
                preview = response.content[:500].replace("\n", " ")
                self._log(f"  [dim]Raw response preview: {preview}...[/dim]")
                
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"LLM analysis failed: {type(e).__name__}: {e}",
            )

    def compare(self, repo_ids: list[str]) -> dict:
        """
        Compare multiple models.

        Args:
            repo_ids: List of repository IDs to compare

        Returns:
            Comparison analysis dictionary
        """
        # Collect all configs
        all_configs = self.collector.collect_many(repo_ids)

        # Format for comparison
        analysis_jsons = "\n\n---\n\n".join(
            cfg.to_prompt_text() for cfg in all_configs if not cfg.error
        )

        prompt = COMPARISON_PROMPT_TEMPLATE.format(analysis_jsons=analysis_jsons)

        try:
            response = self.llm.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )
            return json.loads(response.content)
        except Exception as e:
            return {"error": str(e)}

    def quick_summary(self, repo_id: str) -> str:
        """
        Get a quick text summary of a model.

        Args:
            repo_id: Repository ID

        Returns:
            Text summary string
        """
        configs = self.collector.collect(repo_id)
        if configs.error:
            return f"Error: {configs.error}"

        analysis_json = configs.to_prompt_text()
        prompt = QUICK_SUMMARY_PROMPT.format(analysis_json=analysis_json)

        try:
            response = self.llm.complete(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            return response.content
        except Exception as e:
            return f"Error: {e}"

    def _determine_model_type(self, config_text: str) -> dict:
        """Step 0: Determine model type strictly from configs."""
        prompt = ROUTER_PROMPT.format(config_text=config_text)
        try:
            response = self.llm.complete_json(prompt=prompt, system_prompt="You are a strict classifier.")
            return json.loads(response.content)
        except Exception as e:
            self._log(f"[yellow]⚠ Router failed: {e}[/yellow]")
            if 'response' in locals():
                 self._log(f"[dim]Raw response: {response.content}[/dim]")
            self._log(f"[yellow]Defaulting to unknown[/yellow]")
            return {"primary_type": "unknown", "evidence": []}

    def _interpret_analysis(self, factual_analysis: str, model_type: str) -> str:
        """
        Step 2: Interpret the factual analysis using type-specific prompts.
        """
        try:
            # Select interpreter
            if model_type == "text_only_lm":
                template = TEXT_LM_INTERPRETER
            elif model_type == "image_text_vlm":
                template = IMAGE_TEXT_VLM_INTERPRETER
            elif model_type == "video_text_vlm":
                template = VIDEO_TEXT_VLM_INTERPRETER
            elif model_type == "text_to_video_generator":
                template = T2V_INTERPRETER
            elif model_type == "audio_text_model":
                template = MULTIMODAL_INTERPRETER # fallback to general for now
            elif model_type == "multimodal_general":
                template = MULTIMODAL_INTERPRETER
            else:
                template = MULTIMODAL_INTERPRETER # Default fallback
            
            formatted_prompt = template.format(analysis_json=factual_analysis)
            
            # Get interpretation from LLM
            response = self.llm.complete(
                prompt=formatted_prompt,
                system_prompt=INTERPRETER_SYSTEM_PROMPT,
            )
            
            return response.content
            
        except Exception as e:
            self._log(f"  [yellow]⚠ Interpretation failed: {e}[/yellow]")
            return f"Interpretation failed: {e}"

    def _parse_capabilities(self, json_str: str) -> ModelCapabilities:
        """Parse LLM response into ModelCapabilities."""
        if not json_str:
            raise ValueError("Empty response from LLM")
            
        content = json_str.strip()
        
        # Try to extract JSON from markdown code blocks
        if "```" in content:
            # Find JSON block
            pattern = r'```(?:json)?\s*\n?(.+?)\n?```'
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                content = matches[0].strip()
        
        # Try to find JSON object in response
        if not content.startswith("{"):
            # Look for first { and last }
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                content = content[start:end + 1]
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to provide helpful error
            preview = content[:200] if len(content) > 200 else content
            raise ValueError(f"Invalid JSON from LLM. Preview: {preview!r}") from e
            
        return ModelCapabilities(**data)

    async def analyze_async(self, repo_id: str) -> AnalysisResult:
        """Async version of analyze."""
        configs = self.collector.collect(repo_id)

        if configs.error:
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"Config collection failed: {configs.error}",
            )

        if not configs.configs:
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error="No configuration files found in repository",
            )

        config_text = configs.to_prompt_text()

        try:
            # Step 0: Router
            router_result = self._determine_model_type(config_text)
            primary_type = router_result.get("primary_type", "unknown")

            # Step 1: Extraction
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(config_text=config_text)
            response = await self.llm.acomplete(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            capabilities = self._parse_capabilities(response.content)
            
            # Step 2: Interpretation
            interpreted_analysis = self._interpret_analysis(response.content, primary_type)

            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                model_type=primary_type,
                router_evidence=router_result.get("evidence", []),
                capabilities=capabilities,
                raw_response=response.content,
                interpreted_analysis=interpreted_analysis,
                llm_usage=response.usage,
            )

        except Exception as e:
            return AnalysisResult(
                repo_id=repo_id,
                configs=configs,
                error=f"LLM analysis failed: {type(e).__name__}: {e}",
            )