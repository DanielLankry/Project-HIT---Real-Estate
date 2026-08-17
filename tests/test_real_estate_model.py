"""Regression tests for the leakage-safe real-estate model workflow."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "code" / "real_estate_model_final.py"
SPEC = importlib.util.spec_from_file_location("real_estate_model_final", MODEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load model module from {MODEL_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def synthetic_listings(row_count: int = 30) -> pd.DataFrame:
    """Create deterministic property rows for fast preprocessing tests.

    The fixture includes target, numeric, amenity, and location fields so tests
    exercise the same feature paths as the real compact dataset.
    """

    index = np.arange(row_count)
    return pd.DataFrame(
        {
            "price_usd": 80_000 + index * 7_500,
            "covered_area_sqm": 45 + index * 2,
            "effective_area_sqm": 48 + index * 2,
            "bedrooms": 1 + index % 4,
            "bathrooms": 1 + index % 2,
            "amenity_count": index % 6,
            "has_pool": index % 2,
            "has_gym": (index + 1) % 2,
            "has_grill": index % 3 == 0,
            "has_security": index % 4 == 0,
            "is_near_beach": index % 5 == 0,
            "is_near_sea": index % 5 == 0,
            "is_near_park": index % 3 == 0,
            "is_near_subway": index % 7 == 0,
            "property_type": np.where(index % 2 == 0, "Apartment", "House"),
            "country": np.where(index % 3 == 0, "AR", "UY"),
            "city": np.where(index % 3 == 0, "Buenos Aires", "Montevideo"),
        }
    )


class FeaturePreparationTests(unittest.TestCase):
    """Protect the model boundary from target leakage and category failures."""

    def test_target_and_price_derived_features_are_excluded(self) -> None:
        """Ensure no target information enters the prediction matrix."""

        features, target, numeric, categorical = MODEL.prepare_feature_frame(
            synthetic_listings()
        )

        self.assertNotIn("price_usd", features.columns)
        self.assertNotIn("price_per_sqm", features.columns)
        self.assertEqual(len(features), len(target))
        self.assertIn("luxury_amenity_count", numeric)
        self.assertIn("city", categorical)

    def test_pipeline_handles_missing_and_unseen_categories(self) -> None:
        """Verify train-fitted preprocessing accepts realistic new listings."""

        features, target, numeric, categorical = MODEL.prepare_feature_frame(
            synthetic_listings()
        )
        train = features.iloc[:-1].copy()
        test = features.iloc[[-1]].copy()
        test.loc[:, "property_type"] = "Previously unseen type"
        test["covered_area_sqm"] = test["covered_area_sqm"].astype(float)
        test.loc[:, "covered_area_sqm"] = np.nan

        estimator = MODEL.build_estimator(Ridge(alpha=10.0), numeric, categorical)
        estimator.fit(train, target.iloc[:-1])
        prediction = estimator.predict(test)

        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction[0]))
        self.assertGreater(prediction[0], 0)


if __name__ == "__main__":
    unittest.main()
