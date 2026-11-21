import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


model = LinearRegression()

# Defining the DataSet
df = pd.read_csv("isoc_ci_ifp_iu__custom_524328_linear_2_0.csv.gz")

# Filtering for our case
df_filtered = df[
    (df["geo"] == "IT") &
    (df["ind_type"] == "Y65_74") &
    (df["indic_is"] == "I_ILT12") &
    (df["unit"] == "PC_IND")
]

# Printing a tuple with number of rows and columns
print(df_filtered.shape)

# Prints the name of the columns
print(df_filtered.columns)

# Print the percentage for each year, from 2010 to 2019
print(df_filtered[["TIME_PERIOD", "OBS_VALUE"]])

# Verify missing values
print(df_filtered[["TIME_PERIOD", "OBS_VALUE"]].isna().sum())

# Make datas go cronologically, from 2010 to 2019
df_model = df_filtered[["TIME_PERIOD", "OBS_VALUE"]].sort_values("TIME_PERIOD")
print(df_model)

# Feature and target: x - TIME_PERIOD, y - OBS_VALUE
X = df_model[["TIME_PERIOD"]]
y = df_model["OBS_VALUE"]

print(X.shape)
print(y.shape)

# Training: 2010-2016
train_df = df_model[df_model["TIME_PERIOD"] <= 2016]

# Test: 2017-2019
test_df = df_model[df_model["TIME_PERIOD"] > 2016]

print(train_df)
print(test_df)

# Prepared the training set
X_train = train_df[["TIME_PERIOD"]]
y_train = train_df["OBS_VALUE"]

model.fit(X_train, y_train)

# Parameters where our model learned from the datas
print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

# Preparing the test set
X_test = test_df[["TIME_PERIOD"]]
y_test = test_df["OBS_VALUE"]

predictions = model.predict(X_test)
print("Real values:", y_test.to_numpy())
print("Predictions:", predictions)

mae = sum(abs((y_test - predictions))) / len(predictions)
print(mae)

last_value = train_df["OBS_VALUE"].iloc[-1]
baseline_predictions = [last_value] * len(y_test)

print(baseline_predictions)

baseline_mae = sum(abs((y_test - baseline_predictions))) / len(baseline_predictions)

print("MAE Linear Regression:", mae)
print("MAE Baseline:", baseline_mae)
print("Difference:", mae - baseline_mae)

plt.scatter(X_train["TIME_PERIOD"], y_train, label="Training")
plt.scatter(X_test["TIME_PERIOD"], y_test, label="Test")

X_full = df_model[["TIME_PERIOD"]]
full_predictions = model.predict(X_full)

plt.axvline(
    x=2016.5,
    color="gray",
    linestyle=":",
    label="Train/Test Split"
)

plt.plot(
    X_full["TIME_PERIOD"],
    full_predictions,
    label = "Linear Regression",
    linestyle="--"
)

plt.xlabel("Year")
plt.ylabel("Internet usage (%)")
plt.title("Digital Generations - Internet Adoption")
plt.legend()
plt.show()

