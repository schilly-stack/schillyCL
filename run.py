import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint
import pyfiglet

console = Console()

# ── Helpers ──────────────────────────────────────────────────────────────────

def clear():
    console.clear()

def header():
    ascii_art = pyfiglet.figlet_format("schillyCL", font="slant")
    console.print(f"[bold cyan]{ascii_art}[/bold cyan]")
    console.print(Panel(
        "[bold]WSJ Manga Scan Restoration[/bold]",
        style="cyan",
        expand=False
    ))
    console.print()

def back_prompt():
    Prompt.ask("\n[dim]Press Enter to return to menu[/dim]", default="")

# ── Folder selection ──────────────────────────────────────────────────────────

def select_folder(label, default):
    console.print(f"[yellow]{label}[/yellow] [dim](default: {default})[/dim]")
    val = Prompt.ask("Path", default=str(default))
    p = Path(val)
    if not p.exists():
        console.print(f"[red]Folder not found: {p}[/red]")
        if Confirm.ask("Create it?"):
            p.mkdir(parents=True, exist_ok=True)
        else:
            return None
    return p

def select_weight(default="weights/schillyCL.pth"):
    console.print(f"[yellow]Model weights[/yellow] [dim](default: {default})[/dim]")
    val = Prompt.ask("Path", default=default)
    p = Path(val)
    if not p.exists():
        console.print(f"[red]Weight file not found: {p}[/red]")
        return None
    return p

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(filtered=False):
    clear()
    header()
    script = "filtered_inference.py" if filtered else "inference.py"
    label  = "Filtered inference" if filtered else "Standard inference"
    console.print(Panel(f"[bold]{label}[/bold]", style="cyan", expand=False))
    console.print()

    weight = select_weight()
    if weight is None:
        back_prompt()
        return

    input_dir = select_folder("Input folder", Path("data/test"))
    if input_dir is None:
        back_prompt()
        return

    output_dir = select_folder("Output folder", Path("data/test_output"))
    if output_dir is None:
        back_prompt()
        return

    console.print()
    console.print(f"[dim]Weight:  {weight}[/dim]")
    console.print(f"[dim]Input:   {input_dir}[/dim]")
    console.print(f"[dim]Output:  {output_dir}[/dim]")
    console.print()

    if not Confirm.ask("Run?"):
        back_prompt()
        return

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running inference...", total=None)
        result = subprocess.run(
            [
                sys.executable, script,
                "--weight",  str(weight),
                "--input",   str(input_dir),
                "--output",  str(output_dir)
            ],
            capture_output=False
        )
        progress.update(task, completed=True)

    if result.returncode == 0:
        console.print("\n[bold green]Done.[/bold green]")
        files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        console.print(f"[dim]{len(files)} files saved to {output_dir}[/dim]")
    else:
        console.print("\n[bold red]Inference failed.[/bold red]")

    back_prompt()

# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset():
    clear()
    header()
    console.print(Panel("[bold]Dataset Preparation[/bold]", style="cyan", expand=False))
    console.print()

    table = Table(show_header=False, box=None)
    table.add_row("[cyan]1[/cyan]", "Rename files (sequential or flatten)")
    table.add_row("[cyan]2[/cyan]", "Align scans and generate crops")
    table.add_row("[cyan]3[/cyan]", "Synthetic degradation")
    table.add_row("[cyan]4[/cyan]", "Clean up orphan crops")
    table.add_row("[cyan]b[/cyan]", "Back")
    console.print(table)
    console.print()

    choice = Prompt.ask("Select", choices=["1","2","3","4","b"])

    scripts = {
        "1": "scripts/prepare/rename.py",
        "2": "scripts/prepare/align_and_crop.py",
        "3": "scripts/prepare/degrade.py",
        "4": "scripts/prepare/cleanup_orphans.py",
    }

    if choice == "b":
        return

    script = scripts[choice]
    console.print(f"\n[dim]Running {script}...[/dim]\n")
    subprocess.run([sys.executable, script])
    back_prompt()

# ── Inspect checkpoint ────────────────────────────────────────────────────────

def inspect_checkpoint():
    clear()
    header()
    console.print(Panel("[bold]Inspect Checkpoint[/bold]", style="cyan", expand=False))
    console.print()

    weight = select_weight()
    if weight is None:
        back_prompt()
        return

    console.print()
    subprocess.run([sys.executable, "scripts/utils/inspect_checkpoint.py", str(weight)])
    back_prompt()

# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    while True:
        clear()
        header()

        table = Table(show_header=False, box=None)
        table.add_row("[cyan]1[/cyan]", "Run inference")
        table.add_row("[cyan]2[/cyan]", "Run filtered inference")
        table.add_row("[cyan]3[/cyan]", "Prepare dataset")
        table.add_row("[cyan]4[/cyan]", "Inspect checkpoint")
        table.add_row("[cyan]5[/cyan]", "Exit")
        console.print(table)
        console.print()

        choice = Prompt.ask("Select", choices=["1","2","3","4","5"])

        if choice == "1":
            run_inference(filtered=False)
        elif choice == "2":
            run_inference(filtered=True)
        elif choice == "3":
            prepare_dataset()
        elif choice == "4":
            inspect_checkpoint()
        elif choice == "5":
            clear()
            console.print("[cyan]Bye.[/cyan]")
            break

if __name__ == "__main__":
    main()