from pathlib import Path
from zipfile import ZipFile

import requests

zip_path = Path(__file__).parent / "ml-100k.zip"
if not zip_path.exists():
    with open(zip_path, "wb") as f:
        f.write(
            requests.get(
                "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
            ).content
        )

data_path = Path(__file__).parent / "u.data"
if not data_path.exists():
    with ZipFile(zip_path, "r") as f:
        with f.open("ml-100k/u.data") as data_f:
            with open(data_path, "wb") as out_f:
                out_f.write(data_f.read())
