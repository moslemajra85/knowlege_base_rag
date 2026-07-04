import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# Downloading and loading the mini blog dataset
url = "https://raw.githubusercontent.com/AlaFalaki/tutorial_notebooks/main/data/mini-llama-articles.csv"
mini_dataset = pd.read_csv(url)

console = Console()

def render_dataset_preview(dataframe, rows=10):
    table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY)
    for column in dataframe.columns:
        table.add_column(str(column), overflow="fold")

    for row in dataframe.head(rows).itertuples(index=False):
        table.add_row(*[str(value) for value in row])

    console.print(table)

 