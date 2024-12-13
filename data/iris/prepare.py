from pathlib import Path
from zipfile import ZipFile

import requests

zip_path = Path(__file__).parent / "iris.zip"
if not zip_path.exists():
    with open(zip_path, "wb") as f:
        f.write(
            requests.get(
                "https://archive.ics.uci.edu/static/public/53/iris.zip"
            ).content
        )

data_path = Path(__file__).parent / "iris.data"
if not data_path.exists():
    with ZipFile(zip_path, "r") as f:
        with f.open("iris.data") as data_f:
            with open(data_path, "wb") as out_f:
                out_f.write(data_f.read())
