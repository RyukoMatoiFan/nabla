"""Collect JSON configuration files from HuggingFace repositories."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Callable

from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console

console = Console()


DEFAULT_CONFIG_FILES = [
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "image_processor_config.json",
    "chat_template.json",
    "tokenizer.json",
]


@dataclass
class ConfigFile:
    """Represents a single config file from a HF repo."""

    filename: str
    content: str
    is_valid_json: bool
    size_bytes: int = 0
    parsed: Optional[dict] = None
    error: Optional[str] = None
    selected: bool = True  # For UI selection


@dataclass
class RepoConfigs:
    """All collected configs from a single HF repository."""

    repo_id: str
    configs: list[ConfigFile] = field(default_factory=list)
    readme: Optional[str] = None  # README content
    readme_size: int = 0
    include_readme: bool = False  # Whether to include in analysis
    error: Optional[str] = None

    def to_prompt_text(self, selected_only: bool = True) -> str:
        """Format configs for LLM prompt."""
        lines = [f"# Repository: {self.repo_id}\n"]

        if self.error:
            lines.append(f"Error: {self.error}\n")
            return "\n".join(lines)

        # Include README if selected
        if self.include_readme and self.readme:
            lines.append("\n## README.md")
            lines.append("```markdown")
            # Truncate very long READMEs
            readme_text = self.readme[:8000] + "\n... (truncated)" if len(self.readme) > 8000 else self.readme
            lines.append(readme_text)
            lines.append("```")

        for cfg in self.configs:
            if selected_only and not cfg.selected:
                continue
            lines.append(f"\n## {cfg.filename}")
            if cfg.is_valid_json and cfg.parsed:
                # Pretty print JSON for better LLM understanding
                lines.append("```json")
                lines.append(json.dumps(cfg.parsed, indent=2))
                lines.append("```")
            else:
                lines.append(f"(Invalid JSON: {cfg.error})")
                lines.append("```")
                lines.append(cfg.content[:1000] + "..." if len(cfg.content) > 1000 else cfg.content)
                lines.append("```")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "repo_id": self.repo_id,
            "configs": [
                {
                    "filename": c.filename,
                    "is_valid_json": c.is_valid_json,
                    "parsed": c.parsed,
                    "error": c.error,
                }
                for c in self.configs
            ],
            "error": self.error,
        }


class ConfigCollector:
    """Collects JSON configuration files from HuggingFace repositories."""

    def __init__(
        self,
        token: Optional[str] = None,
        include_files: Optional[list[str]] = None,
        include_all_json: bool = False,
        max_files: int = 50,
        verbose: bool = True,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the config collector.

        Args:
            token: HuggingFace API token for gated repos
            include_files: List of specific filenames to include
            include_all_json: If True, include all JSON files
            max_files: Maximum number of files to collect per repo
            verbose: If True, print detailed progress
            log_callback: Optional callback for log messages
        """
        self.token = token or os.environ.get("HF_TOKEN")
        self.include_files = set(include_files or DEFAULT_CONFIG_FILES)
        self.include_all_json = include_all_json
        self.max_files = max_files
        self.verbose = verbose
        self.log_callback = log_callback
        self.api = HfApi(token=self.token)

    def _log(self, message: str, style: str = "") -> None:
        """Log a message."""
        if self.verbose:
            if style:
                console.print(f"  {message}", style=style)
            else:
                console.print(f"  {message}")
        if self.log_callback:
            self.log_callback(message)

    def _should_include(self, filepath: str) -> bool:
        """Check if a file should be included."""
        filename = os.path.basename(filepath)
        if self.include_all_json:
            return filename.lower().endswith(".json")
        return filename in self.include_files

    def _read_file(self, path: str) -> str:
        """Read file content."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def collect(self, repo_id: str) -> RepoConfigs:
        """
        Collect all matching config files from a HuggingFace repository.

        Args:
            repo_id: The HuggingFace repository ID (e.g., "microsoft/phi-2")

        Returns:
            RepoConfigs object containing all collected configurations
        """
        result = RepoConfigs(repo_id=repo_id)
        
        self._log(f"[bold cyan]Fetching file list from {repo_id}...[/bold cyan]")

        try:
            files = self.api.list_repo_files(repo_id=repo_id)
            self._log(f"Found {len(files)} total files in repository", "dim")
        except Exception as e:
            result.error = f"Failed to list repo files: {type(e).__name__}: {e}"
            self._log(f"[red]Error: {result.error}[/red]")
            return result

        wanted = sorted([f for f in files if self._should_include(f)])
        self._log(f"Matched {len(wanted)} config files to download", "dim")

        if self.include_all_json and len(wanted) > self.max_files:
            self._log(f"[yellow]Limiting to {self.max_files} files[/yellow]")
            wanted = wanted[: self.max_files]

        for i, filepath in enumerate(wanted, 1):
            self._log(f"[{i}/{len(wanted)}] Downloading: [cyan]{filepath}[/cyan]")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filepath,
                    token=self.token,
                )
                content = self._read_file(local_path)
                file_size = len(content)
                
                # Try to parse JSON
                try:
                    parsed = json.loads(content)
                    num_keys = len(parsed) if isinstance(parsed, dict) else len(parsed) if isinstance(parsed, list) else 1
                    self._log(f"    ✓ Valid JSON ({file_size:,} bytes, {num_keys} top-level keys)", "green")
                    cfg = ConfigFile(
                        filename=filepath,
                        content=content,
                        is_valid_json=True,
                        size_bytes=file_size,
                        parsed=parsed,
                    )
                except json.JSONDecodeError as je:
                    self._log(f"    ⚠ Invalid JSON: {je}", "yellow")
                    cfg = ConfigFile(
                        filename=filepath,
                        content=content,
                        is_valid_json=False,
                        size_bytes=file_size,
                        error=str(je),
                    )

                result.configs.append(cfg)

            except Exception as e:
                self._log(f"    ✗ Download failed: {e}", "red")
                result.configs.append(
                    ConfigFile(
                        filename=filepath,
                        content="",
                        is_valid_json=False,
                        error=f"Download failed: {type(e).__name__}: {e}",
                    )
                )

        self._log(f"[bold green]Collected {len(result.configs)} config files[/bold green]")

        # Try to download README
        readme_files = ["README.md", "readme.md", "Readme.md"]
        for readme_name in readme_files:
            if readme_name in files:
                self._log(f"Downloading: [cyan]{readme_name}[/cyan]")
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=readme_name,
                        token=self.token,
                    )
                    result.readme = self._read_file(local_path)
                    result.readme_size = len(result.readme)
                    self._log(f"    ✓ README ({result.readme_size:,} bytes)", "green")
                    break
                except Exception as e:
                    self._log(f"    ✗ README download failed: {e}", "yellow")

        return result

    def collect_many(self, repo_ids: list[str]) -> list[RepoConfigs]:
        """Collect configs from multiple repositories."""
        return [self.collect(repo_id) for repo_id in repo_ids]
