"""CLI entry point with TUI menu support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import pyperclip

try:
    import questionary
    from questionary import Style

    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

from .analyzer import HFAnalyzer, AnalysisResult
from .config_collector import ConfigCollector, RepoConfigs
from .llm_client import LLMClient
from .config import NablaConfig, PROVIDERS, get_config_dir, fetch_openrouter_free_models

console = Console()

# Custom style for questionary
MENU_STYLE = Style([
    ("qmark", "fg:ansibrightcyan bold"),
    ("question", "fg:ansiwhite bold"),
    ("answer", "fg:ansicyan bold"),
    ("pointer", "fg:ansibrightmagenta bold"),
    ("highlighted", "fg:ansibrightcyan bold"),
    ("selected", "fg:ansicyan"),
]) if HAS_QUESTIONARY else None


def print_header() -> None:
    """Print the application header."""
    # ASCII art logo
    console.print()
    console.print("[cyan]    _   _       _     _       [/cyan]")
    console.print("[cyan]   | \\| | __ _| |__ | | __ _ [/cyan]")
    console.print("[cyan]   |  \\| |/ _` | '_ \\| |/ _` |[/cyan]")
    console.print("[cyan]   | |\\  | (_| | |_) | | (_| |[/cyan]")
    console.print("[cyan]   |_| \\_|\\__,_|_.__/|_|\\__,_|[/cyan]")
    console.print()
    console.print("  [dim]HuggingFace Model Analyzer[/dim]")
    console.print("  [dim]Analyze model capabilities using LLM inference[/dim]\n")


def print_current_config(config: NablaConfig) -> None:
    """Print current configuration status."""
    table = Table(title="Current Configuration", show_header=False, border_style="dim")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # LLM Config
    llm_status = config.llm.display_name
    api_key = config.get_effective_api_key(config.llm.provider)
    if api_key:
        llm_status = f"[green]✓[/green] {llm_status}"
    else:
        llm_status = f"[yellow]![/yellow] {config.llm.model} [dim](no key)[/dim]"

    table.add_row("LLM Endpoint", llm_status)

    # HF Config
    hf_token = config.get_effective_hf_token()
    if hf_token:
        table.add_row("HuggingFace", f"[green]✓[/green] Token configured")
    else:
        table.add_row("HuggingFace", "[dim]Public repos only[/dim]")

    # Last analyzed
    if config.last_repos:
        table.add_row("Recent", ", ".join(config.last_repos[-3:]))

    console.print(table)
    console.print()


def format_value(jv, default="Unknown"):
    """Format a JustifiedValue for display."""
    if jv.value is None:
        return f"[dim]{default}[/dim]"
    
    conf = jv.confidence
    if conf == "explicit":
        return f"[green]{jv.value}[/green]"
    elif conf == "derived":
        return f"[yellow]{jv.value}[/yellow]"
    else:
        return f"[dim]{jv.value}[/dim]"


def format_bool_value(jv, label=""):
    """Format a boolean JustifiedValue."""
    if jv.value is True:
        conf = jv.confidence
        if conf == "explicit":
            return f"[green]✓ {label}[/green]"
        elif conf == "derived":
            return f"[yellow]✓ {label}[/yellow]"
        else:
            return f"[dim]✓ {label}[/dim]"
    elif jv.value is False:
        return f"[dim]✗ {label}[/dim]"
    return f"[dim]? {label}[/dim]"


def print_result(result: AnalysisResult) -> None:
    """Pretty print an analysis result."""
    if result.error:
        console.print(f"[red]Error:[/red] {result.error}")
        return

    if not result.capabilities:
        console.print("[yellow]No capabilities extracted[/yellow]")
        return

    cap = result.capabilities
    identity = cap.model_identity
    arch = cap.architecture
    params = cap.parameter_analysis
    ctx = cap.context_window
    mod = cap.modalities
    tokens = cap.special_tokens

    # Header
    model_name = identity.model_name.value or result.repo_id
    console.print(Panel(
        f"[bold cyan]{model_name}[/bold cyan]\n"
        f"[dim]{result.repo_id}[/dim]"
        + (f"\n[dim]Type: {result.model_type}[/dim]" if result.model_type else ""),
        title="Model Analysis",
    ))

    # Legend
    console.print("[dim]Legend: [green]● explicit[/green] | [yellow]● derived[/yellow] | [dim]● unknown[/dim][/dim]\n")

    # Architecture info
    table = Table(title="Architecture", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    table.add_row("Architecture Class", format_value(arch.architecture_class), arch.architecture_class.source)
    table.add_row("Config Architecture", format_value(arch.pretrained_config_architecture), arch.pretrained_config_architecture.source)
    table.add_row("Vocab Size", format_value(params.vocab_size), params.vocab_size.source)
    table.add_row("Hidden Size (d_model)", format_value(params.d_model), params.d_model.source)
    table.add_row("Layers", format_value(params.num_layers), params.num_layers.source)
    table.add_row("Attention Heads", format_value(params.num_heads), params.num_heads.source)
    table.add_row("Max Position Embeddings", format_value(ctx.max_position_embeddings), ctx.max_position_embeddings.source)
    if ctx.sliding_window.value:
        table.add_row("Sliding Window", format_value(ctx.sliding_window), ctx.sliding_window.source)

    console.print(table)

    # Modalities
    console.print("\n[bold]Modalities:[/bold]")
    modalities_list = []
    if mod.visual_encoder_type.value:
        modalities_list.append(f"[green]✓ Vision ({mod.visual_encoder_type.value})[/green]")
    if mod.audio_encoder_type.value:
        modalities_list.append(f"[green]✓ Audio ({mod.audio_encoder_type.value})[/green]")
    if mod.video_support_detected.value:
        modalities_list.append("[green]✓ Video[/green]")
    console.print("  " + " | ".join(modalities_list) if modalities_list else "  [dim]Text only (no multimodal encoders detected)[/dim]")

    # Special Tokens
    tokens_table = Table(title="Special Tokens", show_header=False)
    tokens_table.add_column("Feature")
    tokens_table.add_column("Status")
    tokens_table.add_column("Source", style="dim")

    tokens_table.add_row("Chat Template", format_bool_value(tokens.chat_template_exists), tokens.chat_template_exists.source)
    tokens_table.add_row("EOS Token", format_value(tokens.eos_token), tokens.eos_token.source)

    console.print(tokens_table)

    # Interpreted Analysis (if available)
    if result.interpreted_analysis:
        console.print("\n[bold magenta]Architectural Interpretation:[/bold magenta]")
        console.print(result.interpreted_analysis)

    # Token usage
    if result.llm_usage:
        console.print(f"\n[dim]Tokens used: {result.llm_usage.get('total_tokens', 'N/A')}[/dim]")


def configure_llm(config: NablaConfig) -> None:
    """Configure LLM endpoint."""
    console.print("\n[bold]Configure LLM Endpoint[/bold]\n")

    # Show current config
    console.print(f"Current: {config.llm.display_name}")
    if config.llm.api_base:
        console.print(f"Base URL: {config.llm.api_base}")
    console.print()

    # Select provider - use plain text symbols (questionary doesn't support Rich markup)
    provider_choices = []
    for key, info in PROVIDERS.items():
        env_var = info.get("env_var")
        has_key = bool(os.environ.get(env_var)) if env_var else True
        
        # Use plain Unicode symbols
        if key == config.llm.provider:
            status = "●"  # Current selection
        elif has_key or not info.get("needs_key", True):
            status = "✓"  # Key available or not needed
        else:
            status = "○"  # No key
        
        provider_choices.append(questionary.Choice(
            f"{status} {info['name']}",
            value=key,
        ))

    provider = questionary.select(
        "Select LLM provider:",
        choices=provider_choices,
        default=config.llm.provider,
        style=MENU_STYLE,
    ).ask()

    if not provider:
        return

    provider_info = PROVIDERS[provider]

    # Select model - fetch dynamically for OpenRouter
    if provider == "openrouter":
        console.print("[dim]Fetching free models from OpenRouter...[/dim]")
        models = fetch_openrouter_free_models()
        console.print(f"[dim]Found {len(models)} free models[/dim]\n")
    else:
        models = provider_info["models"]
    
    if "Custom..." not in models:
        models = models + ["Custom..."]
    
    model = questionary.select(
        "Select model:",
        choices=models,
        style=MENU_STYLE,
    ).ask()

    if not model:
        return

    if model == "Custom...":
        model = questionary.text(
            "Enter model identifier (e.g., 'gpt-4', 'llama-3.1-8b'):",
            style=MENU_STYLE,
        ).ask()
        if not model:
            return

    # Configure API base URL if needed (OpenAI-compatible, LM Studio, Ollama, Z.AI, Zhipu)
    api_base = None
    if provider_info.get("needs_base_url") or provider in ["ollama", "openai_compatible", "lmstudio", "zai", "zhipu"]:
        default_base = provider_info.get("default_base", "http://localhost:8000/v1")
        current_base = config.llm.api_base or default_base
        
        console.print(f"\n[dim]Current base URL: {current_base}[/dim]")
        api_base = questionary.text(
            "API base URL:",
            default=current_base,
            style=MENU_STYLE,
        ).ask()

    # Configure API key if needed
    api_key = None
    if provider_info.get("needs_key", False):
        env_var = provider_info.get("env_var")
        current_key = os.environ.get(env_var) if env_var else None
        current_key = current_key or config.llm.api_key

        if current_key:
            masked = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
            console.print(f"\n[dim]Current key: {masked}[/dim]")

        change_key = questionary.confirm(
            "Set/change API key?",
            default=not bool(current_key),
            style=MENU_STYLE,
        ).ask()

        if change_key:
            api_key = questionary.password(
                f"Enter API key:",
                style=MENU_STYLE,
            ).ask()

    # Update config
    config.llm.provider = provider
    config.llm.model = model
    if api_key:
        config.llm.api_key = api_key
    
    # Set or clear api_base depending on provider
    if api_base:
        config.llm.api_base = api_base
    elif not provider_info.get("needs_base_url") and provider not in ["ollama", "openai_compatible", "lmstudio", "zai", "zhipu"]:
        # Clear api_base for providers that don't need it (like OpenRouter, OpenAI, Anthropic)
        config.llm.api_base = None

    config.save()
    console.print("\n[green]✓ LLM configuration saved![/green]\n")


def configure_hf(config: NablaConfig) -> None:
    """Configure HuggingFace settings."""
    console.print("\n[bold]Configure HuggingFace[/bold]\n")

    # Show current config
    console.print(f"Current: {config.hf.display_name}\n")

    # Token
    current_token = os.environ.get("HF_TOKEN") or config.hf.token
    if current_token:
        masked = current_token[:8] + "..." if len(current_token) > 8 else "***"
        console.print(f"[dim]Current token: {masked}[/dim]")

    change_token = questionary.confirm(
        "Set/change HuggingFace token?",
        default=not bool(current_token),
        style=MENU_STYLE,
    ).ask()

    if change_token:
        token = questionary.password(
            "Enter HuggingFace token (for gated repos):",
            style=MENU_STYLE,
        ).ask()
        if token:
            config.hf.token = token

    # Include all JSON
    config.hf.include_all_json = questionary.confirm(
        "Include all JSON files by default?",
        default=config.hf.include_all_json,
        style=MENU_STYLE,
    ).ask()

    config.save()
    console.print("[green]✓ HuggingFace configuration saved![/green]\n")


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def select_files_ui(repo_configs: RepoConfigs) -> bool:
    """Interactive UI for selecting files to include. Returns False if cancelled."""
    console.print("\n[bold]Files found in repository:[/bold]\n")
    
    # Build choice list
    choices = []
    
    # Add README if available (always pre-select)
    if repo_configs.readme:
        choices.append(questionary.Choice(
            f"README.md ({format_size(repo_configs.readme_size)})",
            value="readme",
            checked=True,  # Always pre-select README
        ))
    
    # Add config files
    for cfg in repo_configs.configs:
        status = "[green]✓[/green]" if cfg.is_valid_json else "[yellow]⚠[/yellow]"
        size_str = format_size(cfg.size_bytes)
        choices.append(questionary.Choice(
            f"{cfg.filename} ({size_str})",
            value=cfg.filename,
            checked=cfg.selected,
        ))
    
    if not choices:
        console.print("[yellow]No files found to select.[/yellow]")
        return False
    
    selected = questionary.checkbox(
        "Select files to include (Space to toggle, Enter to confirm):",
        choices=choices,
        style=MENU_STYLE,
    ).ask()
    
    if selected is None:
        return False  # Cancelled
    
    # Update selection state
    repo_configs.include_readme = "readme" in selected
    for cfg in repo_configs.configs:
        cfg.selected = cfg.filename in selected
    
    return True


def analyze_model(config: NablaConfig) -> bool:
    """Analyze a HuggingFace model. Returns True if user wants to go back."""
    console.print("\n[bold]Analyze HuggingFace Model[/bold]\n")

    # Build choices from recent repos (last 5, reversed so most recent first)
    recent_repos = list(reversed(config.last_repos[-5:])) if config.last_repos else []
    
    if recent_repos:
        # Show select menu with option to enter new first, then recent repos
        choices = [questionary.Choice("✏️  Enter new repo ID...", value="__NEW__")]
        choices.extend([questionary.Choice(repo, value=repo) for repo in recent_repos])
        choices.append(questionary.Choice("← Back", value="__BACK__"))
        
        selection = questionary.select(
            "Select a model or enter new:",
            choices=choices,
            style=MENU_STYLE,
        ).ask()
        
        if selection == "__BACK__" or selection is None:
            return True
        elif selection == "__NEW__":
            repo_input = questionary.text(
                "Enter HuggingFace repo ID:",
                style=MENU_STYLE,
            ).ask()
            if not repo_input or repo_input.strip() == "":
                return True
            repo_id = repo_input.strip()
        else:
            repo_id = selection
    else:
        # No recent repos, just ask for input
        console.print("[dim]Press Enter with empty input to go back[/dim]\n")
        repo_input = questionary.text(
            "Enter HuggingFace repo ID:",
            style=MENU_STYLE,
        ).ask()
        
        if not repo_input or repo_input.strip() == "":
            return True
        repo_id = repo_input.strip()



    # Update last repos
    if repo_id not in config.last_repos:
        config.last_repos.append(repo_id)
    config.save()

    # Step 1: Collect files
    console.print("\n[bold cyan]Step 1: Fetching files...[/bold cyan]")
    collector = ConfigCollector(
        token=config.get_effective_hf_token(),
        include_all_json=config.hf.include_all_json,
        verbose=True,
    )
    repo_configs = collector.collect(repo_id)

    if repo_configs.error:
        console.print(f"[red]Error: {repo_configs.error}[/red]")
        return False

    if not repo_configs.configs and not repo_configs.readme:
        console.print("[yellow]No config files or README found in repository.[/yellow]")
        return False

    # Notify when only README is available
    if not repo_configs.configs and repo_configs.readme:
        console.print("[yellow]⚠ No config files found, but README is available for analysis.[/yellow]")

    # Step 2: Let user select files
    console.print("\n[bold cyan]Step 2: Select files to include[/bold cyan]")
    if not select_files_ui(repo_configs):
        return True  # Cancelled

    # Count selected
    selected_count = sum(1 for c in repo_configs.configs if c.selected)
    if repo_configs.include_readme:
        selected_count += 1
    
    if selected_count == 0:
        console.print("[yellow]No files selected.[/yellow]")
        return True

    # Generate prompt text
    prompt_text = repo_configs.to_prompt_text(selected_only=True)
    prompt_size = len(prompt_text)
    estimated_tokens = len(prompt_text.split()) * 1.3  # Rough token estimate
    
    console.print(f"\n[dim]Selected {selected_count} files ({format_size(prompt_size)}, ~{int(estimated_tokens):,} tokens)[/dim]")

    # Warn if prompt is too large
    MAX_SAFE_TOKENS = 100000  # Most models max at 128K
    if estimated_tokens > MAX_SAFE_TOKENS:
        console.print(f"\n[bold yellow]⚠ Warning: Prompt is very large (~{int(estimated_tokens):,} tokens)[/bold yellow]")
        console.print("[yellow]Most LLMs have a limit of 8K-128K tokens. This will likely fail.[/yellow]")
        console.print("[yellow]Consider deselecting large files like tokenizer.json[/yellow]\n")
        
        proceed = questionary.confirm(
            "Proceed anyway?",
            default=False,
            style=MENU_STYLE,
        ).ask()
        if not proceed:
            return True  # Go back to file selection

    # Step 3: Choose action
    console.print("\n[bold cyan]Step 3: Choose action[/bold cyan]")
    
    action_choices = [
        questionary.Choice("🤖 Send to LLM for analysis", value="analyze"),
        questionary.Choice("📋 Copy all contents to clipboard", value="copy"),
        questionary.Choice("← Back", value="back"),
    ]
    
    action = questionary.select(
        "What would you like to do?",
        choices=action_choices,
        style=MENU_STYLE,
    ).ask()

    if action == "back" or action is None:
        return True

    if action == "copy":
        try:
            pyperclip.copy(prompt_text)
            console.print(f"[green]✓ Copied {format_size(prompt_size)} to clipboard![/green]")
        except Exception as e:
            console.print(f"[red]Failed to copy to clipboard: {e}[/red]")
        return False

    # Analyze with LLM
    # Check if LLM is configured
    api_key = config.get_effective_api_key(config.llm.provider)
    if PROVIDERS[config.llm.provider]["needs_key"] and not api_key:
        console.print(f"[yellow]Warning: No API key set for {config.llm.provider}[/yellow]")
        configure_first = questionary.confirm(
            "Configure LLM endpoint first?",
            default=True,
            style=MENU_STYLE,
        ).ask()
        if configure_first:
            configure_llm(config)
            api_key = config.get_effective_api_key(config.llm.provider)
        else:
            return True  # Go back

    console.print("\n[bold cyan]Step 4: Analyzing with LLM...[/bold cyan]")
    
    llm_client = LLMClient(
        model=config.llm.model,
        api_key=api_key,
        api_base=config.llm.api_base,
    )
    analyzer = HFAnalyzer(llm_client=llm_client, config_collector=collector, verbose=True)
    
    # Pass the already-collected configs to analyzer
    result = analyzer.analyze_from_configs(repo_id, repo_configs)

    # Output results
    console.print("\n" + "═" * 60)
    console.print("[bold cyan]ANALYSIS RESULTS[/bold cyan]")
    console.print("═" * 60 + "\n")
    
    print_result(result)
    console.print("\n" + "─" * 60 + "\n")
    
    return False  # Don't go back automatically


def show_settings(config: NablaConfig) -> bool:
    """Show and manage settings. Returns True if user wants to go back."""
    console.print("\n[bold]Settings[/bold]\n")

    config_dir = get_config_dir()
    console.print(f"Config location: [dim]{config_dir}[/dim]\n")

    action = questionary.select(
        "What would you like to do?",
        choices=[
            questionary.Choice("View current config", value="view"),
            questionary.Choice("Reset to defaults", value="reset"),
            questionary.Choice("Clear saved API keys", value="clear_keys"),
            questionary.Choice("← Back", value="back"),
        ],
        style=MENU_STYLE,
    ).ask()

    if action == "back" or action is None:
        return True

    if action == "view":
        console.print_json(json.dumps({
            "llm": {
                "provider": config.llm.provider,
                "model": config.llm.model,
                "api_key": "***" if config.llm.api_key else None,
                "api_base": config.llm.api_base,
            },
            "hf": {
                "token": "***" if config.hf.token else None,
                "include_all_json": config.hf.include_all_json,
            },
            "last_repos": config.last_repos,
        }, indent=2))

    elif action == "reset":
        if questionary.confirm("Reset all settings to defaults?", default=False).ask():
            config = NablaConfig()
            config.save()
            console.print("[green]Settings reset![/green]")

    elif action == "clear_keys":
        if questionary.confirm("Clear all saved API keys?", default=False).ask():
            config.llm.api_key = None
            config.hf.token = None
            config.save()
            console.print("[green]API keys cleared![/green]")
    
    return False


def interactive_menu() -> None:
    """Main interactive TUI menu."""
    if not HAS_QUESTIONARY:
        console.print("[red]TUI requires 'questionary'. Install with: pip install questionary[/red]")
        sys.exit(1)

    config = NablaConfig.load()

    while True:
        console.clear()
        print_header()
        print_current_config(config)

        # Main menu
        choice = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("🔍  Analyze HuggingFace Model", value="analyze"),
                questionary.Choice("🤖  Configure LLM Endpoint", value="llm"),
                questionary.Choice("🤗  Configure HuggingFace", value="hf"),
                questionary.Choice("⚙️   Settings", value="settings"),
                questionary.Choice("❌  Exit", value="exit"),
            ],
            style=MENU_STYLE,
        ).ask()

        if choice == "analyze":
            went_back = analyze_model(config)
            if not went_back:
                questionary.press_any_key_to_continue("Press any key to return to menu...").ask()

        elif choice == "llm":
            configure_llm(config)

        elif choice == "hf":
            configure_hf(config)

        elif choice == "settings":
            went_back = show_settings(config)
            if not went_back:
                questionary.press_any_key_to_continue("Press any key to return to menu...").ask()

        elif choice == "exit" or choice is None:
            console.print("\n[dim]Goodbye![/dim]\n")
            break

        # Reload config in case it was modified
        config = NablaConfig.load()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze HuggingFace model capabilities using LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nabla                                    # Interactive TUI menu
  nabla --repos microsoft/phi-2            # Analyze single model
  nabla --repos Qwen/Qwen2-VL-7B --model xai/grok-2
  nabla --repos meta-llama/Llama-3.1-8B,mistralai/Mistral-7B  # Multiple models
        """,
    )

    parser.add_argument(
        "--repos",
        nargs="+",
        help="HuggingFace repo IDs to analyze",
    )
    parser.add_argument(
        "--model",
        help="LLM model for analysis (uses saved config if not specified)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace token for gated repos",
    )
    parser.add_argument(
        "--all-json",
        action="store_true",
        help="Include all JSON files from repos",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed logging output",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models instead of individual analysis",
    )

    args = parser.parse_args()

    # Load config
    config = NablaConfig.load()

    # Interactive menu if no repos specified
    if not args.repos:
        interactive_menu()
        return 0

    # CLI mode - show logo first
    print_header()

    # CLI mode - use args or fall back to config
    model = args.model or config.llm.model
    api_key = args.api_key or config.get_effective_api_key(config.llm.provider)
    hf_token = args.hf_token or config.get_effective_hf_token()
    verbose = not args.quiet

    llm_client = LLMClient(
        model=model,
        api_key=api_key,
        api_base=config.llm.api_base,
    )
    collector = ConfigCollector(
        token=hf_token,
        include_all_json=args.all_json or config.hf.include_all_json,
        verbose=verbose,
    )
    analyzer = HFAnalyzer(llm_client=llm_client, config_collector=collector, verbose=verbose)

    if args.compare and len(args.repos) > 1:
        # Comparison mode
        console.print("[bold]Comparing models...[/bold]\n")
        comparison = analyzer.compare(args.repos)
        console.print_json(json.dumps(comparison, indent=2))
    else:
        # Individual analysis
        results = []
        for repo_id in args.repos:
            result = analyzer.analyze(repo_id)
            results.append(result)

        console.print("\n" + "═" * 60)
        console.print("[bold cyan]ANALYSIS RESULTS[/bold cyan]")
        console.print("═" * 60 + "\n")
        
        for result in results:
            print_result(result)
            console.print("\n" + "─" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
