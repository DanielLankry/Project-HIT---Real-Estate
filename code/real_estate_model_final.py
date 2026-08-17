"""Train a leakage-safe baseline for cross-sectional property-price prediction.

This model estimates the current listing price from property and location features.
It is not a future-market forecasting model because the repository does not yet
contain a historical time series or aligned macroeconomic indicators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "scraped_real_estate_model_features.csv"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "output" / "real_estate_model_results.png"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "output" / "real_estate_model_metrics.json"
TARGET_COLUMN = "price_usd"

NUMERIC_FEATURE_CANDIDATES = [
    "covered_area_sqm",
    "effective_area_sqm",
    "total_area_sqm",
    "bedrooms",
    "bathrooms",
    "total_rooms",
    "area_per_room_sqm",
    "amenity_count",
    "parking_spaces",
    "expenses_usd",
    "age_years",
    "floor_number",
    "floors_in_building",
    "distance_to_sea_blocks",
    "is_apartment",
    "is_house",
    "is_new_construction",
    "is_under_construction",
    "has_balcony",
    "has_terrace",
    "has_garden",
    "has_patio",
    "has_pool",
    "has_elevator",
    "has_security",
    "has_air_conditioning",
    "has_heating",
    "has_laundry_room",
    "has_storage_room",
    "has_gym",
    "has_grill",
    "is_furnished",
    "is_gated_community",
    "is_near_beach",
    "is_near_park",
    "is_near_sea",
    "is_near_subway",
    "pets_allowed",
    "mortgage_eligible",
]

CATEGORICAL_FEATURE_CANDIDATES = [
    "property_type",
    "property_subtype",
    "construction_stage",
    "country",
    "province",
    "city",
    "neighborhood",
    "location_key",
    "area_bucket",
    "room_bucket",
]


def parse_args() -> argparse.Namespace:
    """Parse paths and validation settings for a reproducible model run.

    Explicit CLI options let local development and CI use the same training
    entry point without relying on the caller's working directory.
    """

    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-safe real-estate listing-price model."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the compact model dataset and validate its target.

    Failing early produces a clear error when the wrong CSV or an incomplete
    scrape is supplied instead of allowing training to fail much later.
    """

    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' is missing from {path}")
    return frame


def prepare_feature_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build the model matrix from target-independent listing attributes.

    The function deliberately excludes price-derived fields such as
    ``price_per_sqm``. Two aggregate features are calculated only from amenity
    flags, so they remain available at real prediction time.
    """

    valid = frame.loc[frame[TARGET_COLUMN].notna() & frame[TARGET_COLUMN].gt(0)].copy()
    if len(valid) < 20:
        raise ValueError("At least 20 positive-price rows are required for model validation")

    numeric_features = [
        column
        for column in NUMERIC_FEATURE_CANDIDATES
        if column in valid.columns and valid[column].notna().any()
    ]
    categorical_features = [
        column
        for column in CATEGORICAL_FEATURE_CANDIDATES
        if column in valid.columns and valid[column].notna().any()
    ]

    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        raise ValueError("No usable prediction features were found in the dataset")

    features = valid[feature_columns].copy()

    luxury_columns = [
        column
        for column in ("has_pool", "has_gym", "has_grill", "has_security")
        if column in features.columns
    ]
    proximity_columns = [
        column
        for column in ("is_near_beach", "is_near_sea", "is_near_park", "is_near_subway")
        if column in features.columns
    ]
    if luxury_columns:
        features["luxury_amenity_count"] = features[luxury_columns].fillna(0).sum(axis=1)
        numeric_features.append("luxury_amenity_count")
    if proximity_columns:
        features["location_proximity_count"] = features[proximity_columns].fillna(0).sum(axis=1)
        numeric_features.append("location_proximity_count")

    target = valid[TARGET_COLUMN].astype(float)
    return features, target, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Create train-fitted numeric and categorical preprocessing.

    Keeping imputers, scaling, and encoding inside the estimator pipeline means
    every cross-validation fold learns preprocessing only from its training rows.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models(random_state: int) -> dict[str, object]:
    """Return a small, interpretable set of regression baselines.

    The candidates cover linear, regularized-linear, bagged-tree, and boosted-
    tree behavior without introducing a large hyperparameter search.
    """

    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }


def build_estimator(
    model: object, numeric_features: list[str], categorical_features: list[str]
) -> TransformedTargetRegressor:
    """Combine preprocessing and a regressor with a log-transformed target.

    Real-estate prices are strongly right-skewed, so fitting on ``log1p(price)``
    reduces domination by a few expensive listings while predictions and metrics
    remain expressed in original US dollars.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", clone(model)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def select_model_with_cross_validation(
    features: pd.DataFrame,
    target: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    cv_folds: int,
    random_state: int,
) -> tuple[str, TransformedTargetRegressor, pd.DataFrame]:
    """Select the best candidate using training-only cross-validation.

    The held-out test set is intentionally absent from this function so it
    cannot influence model selection or preprocessing decisions.
    """

    if not 2 <= cv_folds <= len(features):
        raise ValueError("cv_folds must be between 2 and the number of training rows")

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | str]] = []
    estimators: dict[str, TransformedTargetRegressor] = {}

    for name, model in candidate_models(random_state).items():
        estimator = build_estimator(model, numeric_features, categorical_features)
        scores = cross_validate(
            estimator,
            features,
            target,
            cv=splitter,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            n_jobs=-1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "cv_r2_mean": float(scores["test_r2"].mean()),
                "cv_r2_std": float(scores["test_r2"].std()),
                "cv_mae_usd_mean": float(-scores["test_mae"].mean()),
            }
        )
        estimators[name] = estimator

    results = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    return best_name, estimators[best_name], results


def evaluate_predictions(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    """Calculate held-out regression metrics in original price units.

    Reporting R2 together with absolute and percentage errors prevents a single
    score from being mistaken for classification-style accuracy.
    """

    return {
        "r2": float(r2_score(target, predictions)),
        "mae_usd": float(mean_absolute_error(target, predictions)),
        "rmse_usd": float(np.sqrt(mean_squared_error(target, predictions))),
        "mape": float(mean_absolute_percentage_error(target, predictions)),
    }


def extract_feature_importance(
    estimator: TransformedTargetRegressor,
) -> pd.DataFrame | None:
    """Extract comparable feature weights from the fitted winning pipeline.

    Tree importances are preferred; linear coefficients are converted to
    absolute magnitudes so the diagnostic plot works for every candidate.
    """

    fitted_pipeline = estimator.regressor_
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        return None

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_diagnostics_plot(
    cv_results: pd.DataFrame,
    best_name: str,
    target: pd.Series,
    predictions: np.ndarray,
    metrics: dict[str, float],
    importance: pd.DataFrame | None,
    path: Path,
) -> None:
    """Save a compact visual report of validation and held-out performance.

    The figure separates cross-validation model selection from the final test
    result so readers can see exactly where each reported number came from.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    ordered = cv_results.sort_values("cv_r2_mean")
    colors = ["#d4a72c" if name == best_name else "#3b82b8" for name in ordered["model"]]
    axes[0, 0].barh(
        ordered["model"],
        ordered["cv_r2_mean"],
        xerr=ordered["cv_r2_std"],
        color=colors,
        alpha=0.9,
    )
    axes[0, 0].set_title("Training-only cross-validation R2")
    axes[0, 0].set_xlabel("Mean R2 (error bars: 1 standard deviation)")
    axes[0, 0].grid(axis="x", alpha=0.25)

    axes[0, 1].scatter(target, predictions, alpha=0.55, s=24, color="#2563a6")
    minimum = float(min(target.min(), predictions.min()))
    maximum = float(max(target.max(), predictions.max()))
    axes[0, 1].plot([minimum, maximum], [minimum, maximum], "--", color="#c43d3d")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(f"Held-out test: {best_name}")
    axes[0, 1].set_xlabel("Actual price (USD, log scale)")
    axes[0, 1].set_ylabel("Predicted price (USD, log scale)")
    axes[0, 1].grid(alpha=0.25)

    residuals = target.to_numpy() - predictions
    axes[1, 0].hist(residuals, bins=35, color="#7b5ea7", alpha=0.8, edgecolor="white")
    axes[1, 0].axvline(0, linestyle="--", color="#c43d3d")
    axes[1, 0].set_title("Held-out residuals")
    axes[1, 0].set_xlabel("Actual - predicted price (USD)")
    axes[1, 0].set_ylabel("Listings")
    axes[1, 0].grid(axis="y", alpha=0.25)

    if importance is not None and not importance.empty:
        top = importance.head(12).sort_values("importance")
        axes[1, 1].barh(top["feature"], top["importance"], color="#16826c")
        axes[1, 1].set_title("Top model feature weights")
        axes[1, 1].set_xlabel("Relative importance")
        axes[1, 1].grid(axis="x", alpha=0.25)
    else:
        axes[1, 1].axis("off")

    figure.suptitle(
        "Leakage-safe listing-price baseline\n"
        f"Test R2={metrics['r2']:.3f} | MAE=${metrics['mae_usd']:,.0f} | "
        f"MAPE={metrics['mape']:.1%}",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_metrics(
    path: Path,
    best_name: str,
    metrics: dict[str, float],
    cv_results: pd.DataFrame,
    row_counts: dict[str, int],
    feature_counts: dict[str, int],
) -> None:
    """Write machine-readable validation evidence beside the diagnostic plot.

    Persisted metrics make the PR claim auditable without parsing console output
    or inferring values from chart pixels.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "cross-sectional listing-price prediction, not future forecasting",
        "selection_method": "training-only shuffled K-fold cross-validation by mean R2",
        "best_model": best_name,
        "held_out_test_metrics": metrics,
        "row_counts": row_counts,
        "feature_counts": feature_counts,
        "cross_validation": cv_results.to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_training(
    data_path: Path,
    plot_path: Path,
    metrics_path: Path,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Train, select, evaluate, and persist the complete model workflow.

    A single stratification-free random holdout is appropriate for the current
    cross-sectional snapshot. Future temporal data should use time-based splits
    in a separate forecasting workflow.
    """

    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5")

    frame = load_dataset(data_path)
    features, target, numeric_features, categorical_features = prepare_feature_frame(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    best_name, best_estimator, cv_results = select_model_with_cross_validation(
        x_train,
        y_train,
        numeric_features,
        categorical_features,
        cv_folds,
        random_state,
    )
    best_estimator.fit(x_train, y_train)
    predictions = np.maximum(best_estimator.predict(x_test), 0)
    metrics = evaluate_predictions(y_test, predictions)
    importance = extract_feature_importance(best_estimator)

    row_counts = {
        "usable": int(len(features)),
        "train": int(len(x_train)),
        "test": int(len(x_test)),
    }
    feature_counts = {
        "numeric": len(numeric_features),
        "categorical": len(categorical_features),
        "total_input": len(numeric_features) + len(categorical_features),
    }
    save_diagnostics_plot(
        cv_results,
        best_name,
        y_test,
        predictions,
        metrics,
        importance,
        plot_path,
    )
    save_metrics(
        metrics_path,
        best_name,
        metrics,
        cv_results,
        row_counts,
        feature_counts,
    )

    return {
        "best_model": best_name,
        "metrics": metrics,
        "cv_results": cv_results,
        "row_counts": row_counts,
        "feature_counts": feature_counts,
    }


def main() -> None:
    """Run the command-line training workflow and print its audit summary."""

    args = parse_args()
    result = run_training(
        data_path=args.data,
        plot_path=args.output,
        metrics_path=args.metrics_output,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    print("\nCross-validation results (training split only):")
    print(result["cv_results"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {result['best_model']}")
    print("Held-out test metrics:")
    for metric, value in result["metrics"].items():
        if metric.endswith("_usd"):
            print(f"  {metric}: ${value:,.0f}")
        elif metric == "mape":
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value:.4f}")
    print(f"\nSaved plot: {args.output}")
    print(f"Saved metrics: {args.metrics_output}")


if __name__ == "__main__":
    main()
