from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

zip_path = Path(__file__).parent / "wine.zip"
if not zip_path.exists():
    with open(zip_path, "wb") as f:
        f.write(
            requests.get(
                "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
            ).content
        )

data_path = Path(__file__).parent / "wine.csv"
if not data_path.exists():
    with ZipFile(zip_path, "r") as f:
        with f.open("winequality-white.csv") as data_f:
            df = pd.read_csv(data_f, sep=";", header=None)
            df = df.drop(index=0)
            df.to_csv(data_path, index=False)
