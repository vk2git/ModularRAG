import argparse
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()


def ensure_documents_folder():
    """Checks if documents folder exists, creates it if not."""
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        console.print(f"⚠️  '[bold]{docs_dir}[/bold]' folder not found. Creating it...")
        os.makedirs(docs_dir)
        console.print(f"👉 Put your documents (PDF, TXT, MD, etc.) in '[bold]{docs_dir}[/bold]' and run with --ingest.")
        return False
    
    if not os.listdir(docs_dir):
        console.print(f"⚠️  '[bold]{docs_dir}[/bold]' folder is empty.")
        console.print(f"👉 Add documents to '[bold]{docs_dir}[/bold]' before ingesting.")
        return False
        
    return True


def run_ingestion():
    """Runs the ingestion process."""
    console.print("\n🚀 [bold]Starting Ingestion Process...[/bold]")
    if not ensure_documents_folder():
        return

    try:
        from src.core.ingestion.manager import IngestionManager
        manager = IngestionManager()
        manager.run_ingestion()
        console.print("\n✅ [bold green]Ingestion Complete![/bold green]")
    except Exception as e:
        console.print(f"\n❌ [bold red]Ingestion Failed:[/bold red] {e}")


def list_architectures():
    """Display all available RAG architectures in a rich table."""
    from src.core.registry import ArchitectureRegistry
    from src.utils.config_loader import get_active_architecture

    registry = ArchitectureRegistry()
    architectures = registry.list_architectures()
    active = get_active_architecture()

    table = Table(
        title="🏗️  Available RAG Architectures",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_blue",
    )

    table.add_column("Architecture", style="bold", min_width=18)
    table.add_column("Description", min_width=40)
    table.add_column("Dependencies", min_width=15)
    table.add_column("Status", min_width=12)

    for arch in architectures:
        name = arch["name"]
        if name == active:
            name_display = f"▶ {arch['display_name']} [bold green](active)[/bold green]"
        else:
            name_display = f"  {arch['display_name']}"

        deps = ", ".join(arch["requires"]) if arch["requires"] else "none"
        
        table.add_row(
            name_display,
            arch["description"],
            deps,
            arch["status"],
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Use [bold]--arch <name>[/bold] to run with a specific architecture[/dim]")
    console.print("[dim]Use [bold]--select[/bold] to interactively choose and save[/dim]")
    console.print()


def select_architecture():
    """Interactive architecture selector."""
    from src.core.registry import ArchitectureRegistry
    from src.utils.config_loader import set_active_architecture, get_active_architecture

    registry = ArchitectureRegistry()
    architectures = registry.list_architectures()
    current = get_active_architecture()

    console.print()
    console.print(Panel(
        "[bold]Select a RAG Architecture[/bold]\n"
        "This will update your config/settings.yaml",
        border_style="cyan",
    ))

    # Show options
    for i, arch in enumerate(architectures, 1):
        marker = " [bold green]← current[/bold green]" if arch["name"] == current else ""
        status = "✅" if arch["available"] else "⚠️"
        console.print(f"  {status} [bold]{i}.[/bold] {arch['display_name']} — {arch['description']}{marker}")

    console.print()

    # Get selection
    while True:
        choice = Prompt.ask(
            "Enter number or architecture name",
            default=current,
        )

        # Handle numeric input
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(architectures):
                selected = architectures[idx]["name"]
                break
        except ValueError:
            pass

        # Handle name input
        names = [a["name"] for a in architectures]
        if choice.lower() in names:
            selected = choice.lower()
            break

        console.print(f"[red]Invalid choice. Try a number 1-{len(architectures)} or a name.[/red]")

    # Confirm
    selected_arch = next(a for a in architectures if a["name"] == selected)
    if not selected_arch["available"]:
        console.print(f"\n⚠️  [yellow]{selected_arch['display_name']} has missing dependencies.[/yellow]")
        console.print(f"   Install with: [bold]uv pip install {' '.join(selected_arch['requires'])}[/bold]")

        if not Confirm.ask("Set as active anyway?", default=False):
            return

    # Save
    set_active_architecture(selected)
    console.print(f"\n✅ Active architecture set to [bold green]{selected_arch['display_name']}[/bold green]")
    console.print(f"   Config file: [dim]config/architectures/{selected}.yaml[/dim]")


def show_config(arch_name: str = None):
    """Display the current configuration for an architecture."""
    from src.utils.config_loader import load_config, load_architecture_config, get_active_architecture

    if arch_name is None:
        arch_name = get_active_architecture()

    global_config = load_config()
    arch_config = load_architecture_config(arch_name)

    console.print()
    console.print(Panel(f"[bold]Configuration: {arch_name}[/bold]", border_style="cyan"))

    # Show global settings
    console.print("[bold]Global Settings:[/bold]")
    console.print(f"  LLM:        {global_config.get('llm', {}).get('mode', 'local')}")
    console.print(f"  Embeddings: {global_config.get('embedding', {}).get('provider', 'huggingface')}")
    console.print(f"  Vector DB:  {global_config.get('vector_db', {}).get('provider', 'chroma')}")
    console.print(f"  Memory:     {global_config.get('memory', {}).get('type', 'window')}")
    console.print(f"  Guardrails: {'enabled' if global_config.get('guardrails', {}).get('enabled', True) else 'disabled'}")
    console.print(f"  Web Search: {'enabled' if global_config.get('web_search', {}).get('enabled', True) else 'disabled'}")

    console.print()
    console.print(f"[bold]Architecture-Specific ({arch_name}):[/bold]")
    if arch_config:
        for key, value in arch_config.items():
            if isinstance(value, dict):
                console.print(f"  {key}:")
                for k, v in value.items():
                    console.print(f"    {k}: {v}")
            else:
                console.print(f"  {key}: {value}")
    else:
        console.print("  [dim]No architecture-specific overrides. Using global defaults.[/dim]")

    console.print()
    console.print(f"[dim]Edit config at: config/architectures/{arch_name}.yaml[/dim]")
    console.print(f"[dim]Edit global config at: config/settings.yaml[/dim]")


def toggle_web_search():
    """Toggle web search on/off."""
    from src.utils.config_loader import load_config
    import yaml

    config = load_config()
    current = config.get("web_search", {}).get("enabled", True)
    new_state = not current

    # Update config file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    full_path = os.path.join(base_dir, "config", "settings.yaml")

    with open(full_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if "web_search" not in raw_config:
        raw_config["web_search"] = {}
    raw_config["web_search"]["enabled"] = new_state

    with open(full_path, "w") as f:
        yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    state_str = "[bold green]ON[/bold green]" if new_state else "[bold red]OFF[/bold red]"
    console.print(f"🔍 Web search: {state_str}")


def run_chat(architecture: str = None, verbose: bool = False):
    """Runs the interactive chat loop with the selected architecture."""
    from src.core.runner import PipelineRunner
    from src.utils.config_loader import get_active_architecture

    arch_name = architecture or get_active_architecture()

    console.print()
    console.print(Panel(
        f"[bold]ModularRAG Chat[/bold]\n"
        f"Architecture: [cyan]{arch_name}[/cyan]\n"
        f"Type [bold]'exit'[/bold] to quit, [bold]'switch <name>'[/bold] to change architecture,\n"
        f"[bold]'list'[/bold] to see architectures, [bold]'websearch'[/bold] to toggle web search",
        border_style="bright_blue",
    ))
    console.print()

    try:
        runner = PipelineRunner(architecture_name=arch_name, verbose=verbose)

        while True:
            try:
                query = input("\nYou: ")

                if query.lower().strip() in ["exit", "quit", "bye"]:
                    console.print("👋 [bold]Goodbye![/bold]")
                    break

                if not query.strip():
                    continue

                # In-chat commands
                if query.lower().strip() == "list":
                    list_architectures()
                    continue

                if query.lower().strip() == "websearch":
                    toggle_web_search()
                    continue

                if query.lower().strip().startswith("switch "):
                    new_arch = query.strip().split(" ", 1)[1].strip()
                    try:
                        runner.switch_architecture(new_arch)
                    except ValueError as e:
                        console.print(f"[red]{e}[/red]")
                    continue

                if query.lower().strip() == "health":
                    runner.check_health()
                    continue

                if query.lower().strip() == "info":
                    info = runner.get_current_architecture()
                    console.print(f"[cyan]Architecture:[/cyan] {info['display_name']}")
                    console.print(f"[cyan]Description:[/cyan] {info['description']}")
                    continue

                # Run query
                response = runner.run(query)
                console.print(f"\n[bold]Bot:[/bold] {response}")
                
            except KeyboardInterrupt:
                console.print("\n👋 [bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
                
    except Exception as e:
        console.print(f"[red]❌ Failed to start chatbot: {e}[/red]")
        console.print("   Hint: Did you run ingestion? (uv run main.py --ingest)")


def main():
    parser = argparse.ArgumentParser(
        description="ModularRAG — Multi-Architecture RAG Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                   Start chat with configured architecture
  uv run main.py --list            List all available architectures
  uv run main.py --select          Interactively choose an architecture
  uv run main.py --arch advanced   Start chat with Advanced RAG
  uv run main.py --config          Show current configuration
  uv run main.py --ingest          Ingest documents into vector store
  uv run main.py --verbose         Enable verbose/debug output
        """
    )

    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--list", action="store_true", help="List available RAG architectures")
    parser.add_argument("--select", action="store_true", help="Interactively select an architecture")
    parser.add_argument("--arch", type=str, metavar="NAME", help="Run with a specific architecture")
    parser.add_argument("--config", nargs="?", const="__default__", metavar="NAME",
                        help="Show configuration (optionally for a specific architecture)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--health", action="store_true", help="Run health checks")

    args = parser.parse_args()
    
    if args.list:
        list_architectures()
    elif args.select:
        select_architecture()
    elif args.config is not None:
        arch = args.config if args.config != "__default__" else None
        show_config(arch)
    elif args.ingest:
        run_ingestion()
    elif args.health:
        from src.core.runner import PipelineRunner
        runner = PipelineRunner(architecture_name=args.arch, verbose=args.verbose)
        runner.check_health()
    else:
        run_chat(architecture=args.arch, verbose=args.verbose)


if __name__ == "__main__":
    main()
