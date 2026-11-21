
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "isoc_ci_ifp_iu__custom_524328_linear_2_0.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "regression.png"


def load_data():
    """Load and prepare Italian internet adoption data."""
    df = pd.read_csv(DATA_PATH)

    filtered = df.loc[
        (df["geo"] == "IT")
        & (df["ind_type"] == "Y65_74")
        & (df["indic_is"] == "I_ILT12")
        & (df["unit"] == "PC_IND"),
        ["TIME_PERIOD", "OBS_VALUE"],
    ].copy()

    if filtered.empty:
        raise ValueError("No observations found.")

    if filtered.isna().any().any():
        raise ValueError("Missing values detected.")

    if filtered["TIME_PERIOD"].duplicated().any():
        raise ValueError("Duplicate years detected.")

    return filtered.sort_values("TIME_PERIOD").reset_index(drop=True)


def split_data(df):
    """Split observations chronologically."""
    train = df[df["TIME_PERIOD"] <= 2016]
    test = df[df["TIME_PERIOD"] > 2016]

    if train.empty or test.empty:
        raise ValueError("Training and test sets must not be empty.")

    return train, test


def train_model(train):
    """Train a linear regression model."""
    X_train = train[["TIME_PERIOD"]]
    y_train = train["OBS_VALUE"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, train, test):
    """Compare model predictions with a naive baseline."""
    X_test = test[["TIME_PERIOD"]]
    y_test = test["OBS_VALUE"]

    predictions = model.predict(X_test)

    baseline_predictions = [train["OBS_VALUE"].iloc[-1]] * len(test)

    mae = mean_absolute_error(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    return predictions, mae, baseline_mae


def plot_results(df, train, test, model):
    """Save the observed data and fitted regression line."""
    plt.figure(figsize=(10, 6))

    plt.scatter(
        train["TIME_PERIOD"], train["OBS_VALUE"],
        label="Training"
    )
    plt.scatter(
        test["TIME_PERIOD"], test["OBS_VALUE"],
        label="Test"
    )

    full_predictions = model.predict(df[["TIME_PERIOD"]])

    plt.plot(
        df["TIME_PERIOD"], full_predictions,
        linestyle="--", label="Linear Regression"
    )

    plt.axvline(2016.5, linestyle=":", label="Train/Test Split")

    plt.xlabel("Year")
    plt.ylabel("Internet usage (%)")
    plt.title("Digital Generations - Internet Adoption")
    plt.legend()
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=160)
    plt.close()


def main():
    df = load_data()
    train, test = split_data(df)

    model = train_model(train)
    predictions, mae, baseline_mae = evaluate_model(model, train, test)

    print(f"Linear Regression MAE: {mae:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")

    for year, actual, predicted in zip(
        test["TIME_PERIOD"], test["OBS_VALUE"], predictions
    ):
        print(f"{year}: actual={actual:.2f}%, predicted={predicted:.2f}%")

    plot_results(df, train, test, model)


if __name__ == "__main__":
    main()
