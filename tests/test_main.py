import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import split_data


def test_chronological_split():
    df = pd.DataFrame({
        "TIME_PERIOD": [2015, 2016, 2017, 2018],
        "OBS_VALUE": [20.0, 25.0, 30.0, 35.0],
    })

    train, test = split_data(df)

    assert train["TIME_PERIOD"].tolist() == [2015, 2016]
    assert test["TIME_PERIOD"].tolist() == [2017, 2018]
