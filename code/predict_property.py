"""Train the validated Ridge baseline and predict one listing from terminal input.

The script intentionally saves no binary model file. It reloads the current
model CSV, fits the same leakage-safe pipeline as the notebook, and asks simple
English questions so a user can estimate a current online asking price.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "scraped_real_estate_model_features.csv"
TARGET_COLUMN = "price_usd"
RIDGE_ALPHA = 10.0

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

BINARY_FEATURES = {
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
}

AMENITY_ALIASES = {
    "balcony": "has_balcony",
    "terrace": "has_terrace",
    "garden": "has_garden",
    "patio": "has_patio",
    "pool": "has_pool",
    "elevator": "has_elevator",
    "security": "has_security",
    "air conditioning": "has_air_conditioning",
    "heating": "has_heating",
    "laundry room": "has_laundry_room",
    "storage room": "has_storage_room",
    "gym": "has_gym",
    "grill": "has_grill",
    "furnished": "is_furnished",
    "gated community": "is_gated_community",
    "near beach": "is_near_beach",
    "near park": "is_near_park",
    "near sea": "is_near_sea",
    "near subway": "is_near_subway",
    "pets allowed": "pets_allowed",
    "mortgage eligible": "mortgage_eligible",
}


def prepare_training_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build the exact target-independent feature frame used by the notebook.

    Keeping preprocessing and aggregate feature construction aligned with the
    validated notebook prevents the terminal predictor from using a different
    model definition.
    """

    valid = frame.loc[
        frame[TARGET_COLUMN].notna() & frame[TARGET_COLUMN].gt(0)
    ].copy()
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
    features = valid[numeric_features + categorical_features].copy()
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
    features["luxury_amenity_count"] = features[luxury_columns].fillna(0).sum(axis=1)
    features["location_proximity_count"] = (
        features[proximity_columns].fillna(0).sum(axis=1)
    )
    numeric_features += ["luxury_amenity_count", "location_proximity_count"]
    return features, valid[TARGET_COLUMN].astype(float), numeric_features, categorical_features


def build_estimator(numeric_features: list[str], categorical_features: list[str]):
    """Create the train-fitted preprocessing and Ridge price estimator.

    Missing values, scaling, and one-hot encoding stay inside the pipeline so
    terminal input is treated exactly like a new unseen listing.
    """

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", Ridge(alpha=RIDGE_ALPHA))]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def prompt_choice(label: str, choices: dict[str, str]) -> str:
    """Prompt until the user enters one of the documented English choices."""

    while True:
        value = input(label).strip().casefold()
        if value in choices:
            return choices[value]
        print(f"Please enter one of: {', '.join(sorted(choices))}")


def prompt_number(label: str, *, optional: bool = False, minimum: float = 0) -> float:
    """Read a bounded numeric value while allowing documented blank inputs."""

    while True:
        raw = input(label).strip()
        if optional and not raw:
            return np.nan
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            print("Please enter a number." + (" Leave blank if unknown." if optional else ""))
            continue
        if value < minimum:
            print(f"Please enter a value of at least {minimum:g}.")
            continue
        return value


def bucket_area(area: float) -> str:
    """Convert effective area into the same simple buckets as the scraper."""

    for upper_bound, label in (
        (30, "01_tiny"),
        (50, "02_small"),
        (80, "03_medium"),
        (120, "04_large"),
        (200, "05_extra_large"),
        (10_000, "06_luxury_scale"),
    ):
        if area <= upper_bound:
            return label
    return "06_luxury_scale"


def bucket_rooms(bedrooms: float, total_rooms: float) -> str:
    """Convert room counts into the categorical groups used during training."""

    if total_rooms == 1:
        return "01_studio"
    if bedrooms <= 1:
        return "02_one_bedroom"
    if bedrooms == 2:
        return "03_two_bedroom"
    if bedrooms == 3:
        return "04_three_bedroom"
    return "05_four_plus_bedroom"


def known_or_original(frame: pd.DataFrame, column: str, value: str) -> tuple[str, bool]:
    """Reuse known category spelling and report whether a location is unseen."""

    if not value:
        return "", True
    known_values = frame[column].dropna().astype(str).unique()
    matches = {known.casefold(): known for known in known_values}
    matched = matches.get(value.casefold())
    return (matched, True) if matched is not None else (value, False)


def parse_amenities(raw: str) -> tuple[set[str], list[str]]:
    """Map comma-separated English amenity names to model feature columns."""

    selected: set[str] = set()
    unknown: list[str] = []
    for item in raw.split(","):
        name = " ".join(item.strip().casefold().replace("_", " ").split())
        if not name:
            continue
        column = AMENITY_ALIASES.get(name)
        if column:
            selected.add(column)
        else:
            unknown.append(item.strip())
    return selected, unknown


def collect_property_input(training_frame: pd.DataFrame) -> pd.DataFrame:
    """Collect one understandable English terminal form and derive model fields."""

    print("\nEnter the property details in English. Required fields cannot be blank.\n")
    country = prompt_choice(
        "Country code [AR/UY]: ",
        {"ar": "AR", "argentina": "AR", "uy": "UY", "uruguay": "UY"},
    )
    property_type = prompt_choice(
        "Property type [Apartment/House]: ",
        {"apartment": "Apartment", "house": "House"},
    )
    province_input = input("Province [optional]: ").strip()
    city_input = input("City [optional]: ").strip()
    neighborhood_input = input("Neighborhood [optional]: ").strip()
    covered_area = prompt_number("Covered area in sqm: ", minimum=1)
    total_area = prompt_number("Total area in sqm [optional]: ", optional=True, minimum=1)
    while not np.isnan(total_area) and total_area < covered_area:
        print("Total area cannot be smaller than covered area.")
        total_area = prompt_number(
            "Total area in sqm [optional]: ", optional=True, minimum=1
        )
    bedrooms = prompt_number("Bedrooms: ", minimum=0)
    bathrooms = prompt_number("Bathrooms: ", minimum=1)
    total_rooms = prompt_number(
        "Total rooms [optional; defaults to bedrooms + 1]: ", optional=True, minimum=1
    )
    while not np.isnan(total_rooms) and total_rooms < bedrooms:
        print("Total rooms cannot be smaller than bedrooms.")
        total_rooms = prompt_number(
            "Total rooms [optional; defaults to bedrooms + 1]: ",
            optional=True,
            minimum=1,
        )
    if np.isnan(total_rooms):
        total_rooms = bedrooms + 1
    parking_spaces = prompt_number("Parking spaces [optional]: ", optional=True, minimum=0)
    expenses_usd = prompt_number(
        "Monthly expenses in USD [optional]: ", optional=True, minimum=0
    )
    age_years = prompt_number("Property age in years [optional]: ", optional=True, minimum=0)
    construction_stage = input(
        "Construction stage [new/under_construction/renovated/resale, optional]: "
    ).strip().casefold()
    allowed_stages = {"", "new", "under_construction", "renovated", "resale"}
    while construction_stage not in allowed_stages:
        print("Please enter new, under_construction, renovated, resale, or leave blank.")
        construction_stage = input("Construction stage: ").strip().casefold()

    print("Available amenities: " + ", ".join(AMENITY_ALIASES))
    selected_amenities, unknown_amenities = parse_amenities(
        input("Amenities mentioned, comma-separated [optional]: ")
    )
    if unknown_amenities:
        print("Ignored unknown amenities: " + ", ".join(unknown_amenities))

    province, known_province = known_or_original(training_frame, "province", province_input)
    city, known_city = known_or_original(training_frame, "city", city_input)
    neighborhood, known_neighborhood = known_or_original(
        training_frame, "neighborhood", neighborhood_input
    )
    for label, value, is_known in (
        ("province", province, known_province),
        ("city", city, known_city),
        ("neighborhood", neighborhood, known_neighborhood),
    ):
        if value and not is_known:
            print(f"Warning: {label} '{value}' was not seen during training.")

    effective_area = covered_area
    record: dict[str, object] = {
        column: 0 for column in BINARY_FEATURES
    }
    record.update(
        {
            "covered_area_sqm": covered_area,
            "effective_area_sqm": effective_area,
            "total_area_sqm": total_area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "total_rooms": total_rooms,
            "area_per_room_sqm": effective_area / total_rooms,
            "parking_spaces": parking_spaces,
            "expenses_usd": expenses_usd,
            "age_years": age_years,
            "is_apartment": int(property_type == "Apartment"),
            "is_house": int(property_type == "House"),
            "is_new_construction": int(construction_stage == "new"),
            "is_under_construction": int(construction_stage == "under_construction"),
            "property_type": property_type,
            "property_subtype": np.nan,
            "construction_stage": construction_stage or np.nan,
            "country": country,
            "province": province or np.nan,
            "city": city or np.nan,
            "neighborhood": neighborhood or np.nan,
            "location_key": " | ".join(
                part for part in (country, province, city, neighborhood) if part
            ),
            "area_bucket": bucket_area(effective_area),
            "room_bucket": bucket_rooms(bedrooms, total_rooms),
        }
    )
    for column in selected_amenities:
        record[column] = 1
    record["amenity_count"] = sum(
        int(record[column]) for column in BINARY_FEATURES if column.startswith("has_")
    )
    record["luxury_amenity_count"] = sum(
        int(record[column]) for column in ("has_pool", "has_gym", "has_grill", "has_security")
    )
    record["location_proximity_count"] = sum(
        int(record[column])
        for column in ("is_near_beach", "is_near_sea", "is_near_park", "is_near_subway")
    )
    return pd.DataFrame([record])


def main() -> None:
    """Fit the current full-data baseline and print one terminal prediction."""

    if not MODEL_DATA_PATH.exists():
        raise SystemExit(f"Model data not found: {MODEL_DATA_PATH}")
    model_frame = pd.read_csv(MODEL_DATA_PATH)
    features, target, numeric_features, categorical_features = prepare_training_data(
        model_frame
    )
    estimator = build_estimator(numeric_features, categorical_features)
    estimator.fit(features, target)
    prediction_input = collect_property_input(model_frame)
    prediction_input = prediction_input.reindex(columns=features.columns)
    predicted_price = max(float(estimator.predict(prediction_input)[0]), 0)

    print("\nPrediction result")
    print("-----------------")
    print(f"Estimated current listing price: ${predicted_price:,.0f} USD")
    print(f"Model trained on {len(features):,} cleaned listings.")
    print("This is an asking-price estimate, not a future forecast or confirmed sale price.")
    if prediction_input["is_house"].iloc[0] == 1:
        print("Caution: house estimates are less stable than apartment estimates in validation.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nPrediction cancelled.")
