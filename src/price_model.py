import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_and_evaluate(data_path: str = "data/alberletek.csv"):
    df = pd.read_csv(data_path)
    print(f"Betöltött sorok száma: {len(df)}")
    print("Hiányzó értékek betöltéskor:\n", df.isnull().sum()[df.isnull().sum() > 0])

    # Jellemzők (X) és célváltozó (y) szétválasztása
    feature_cols = [
        "kerulet",
        "alapterulet_nm",
        "szobak_szama",
        "allapot",
        "erkely",
        "lift",
        "butorozott",
    ]
    X = df[feature_cols]
    y = df["ar_ft"]

    numeric_features = ["alapterulet_nm", "szobak_szama", "erkely", "lift", "butorozott"]
    categorical_features = ["kerulet", "allapot"]

    # Előfeldolgozó pipeline (adattisztítás + skálázás + One-Hot kódolás)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Lineáris Regresszió": LinearRegression(),
        "Random Forest Regresszor": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=42
        ),
    }

    results = {}
    best_model = None
    best_r2 = -np.inf
    best_name = ""

    print("\n--- Modell kiértékelési eredmények (Teszt halmaz) ---")
    for name, regressor in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"y_pred": y_pred, "MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"{name:25s} | R2: {r2:.4f} | MAE: {mae:,.0f} Ft | RMSE: {rmse:,.0f} Ft")

        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe
            best_name = name

    # Legjobb modell mentése
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "price_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nLegjobb modell mentve ({best_name}): {model_path}")

    # Kiértékelő ábra készítése a dokumentációhoz
    os.makedirs("reports", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Valós vs Becsült ár scatter plot
    best_preds = results[best_name]["y_pred"]
    axes[0].scatter(y_test / 1000, best_preds / 1000, alpha=0.5, color="#2563eb", edgecolors="k", linewidths=0.3)
    min_val = min(y_test.min(), best_preds.min()) / 1000
    max_val = max(y_test.max(), best_preds.max()) / 1000
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideális becslés (y = x)")
    axes[0].set_title(f"{best_name}\nValós vs. Becsült bérleti díj (R² = {best_r2:.3f})")
    axes[0].set_xlabel("Valós hirdetési ár (ezer Ft)")
    axes[0].set_ylabel("MI által becsült ár (ezer Ft)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Modellek MAE összehasonlítása
    model_names = list(results.keys())
    maes = [results[m]["MAE"] / 1000 for m in model_names]
    bars = axes[1].bar(model_names, maes, color=["#64748b", "#16a34a"], width=0.5)
    axes[1].set_title("Átlagos abszolút hiba (MAE) összehasonlítása")
    axes[1].set_ylabel("MAE (ezer Ft)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f} eFt", ha="center", fontweight="bold")

    plt.tight_layout()
    plot_path = os.path.join("reports", "model_evaluation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Kiértékelő grafikon elmentve: {plot_path}")


if __name__ == "__main__":
    train_and_evaluate()