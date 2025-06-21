import typer
from rich import print

from .main import fem

app = typer.Typer()


@app.command()
def main(n1: int, n2: int) -> None:
    """Add the arguments and print the result."""
    print(fem(n1, n2))
