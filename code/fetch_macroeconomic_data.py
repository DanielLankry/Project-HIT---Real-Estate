"""Download basic World Bank context data for Argentina and Uruguay.

The output is a small long-format CSV used by the EDA notebook. Economic values
provide country-level context only and are not joined to individual listings.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "macroeconomic_indicators.csv"
START_YEAR = 2015
END_YEAR = 2024

COUNTRIES = {"ARG": "Argentina", "URY": "Uruguay"}
INDICATORS = {
    "FP.CPI.TOTL.ZG": ("Inflation, consumer prices", "annual %"),
    "NY.GDP.MKTP.KD.ZG": ("GDP growth", "annual %"),
}


def fetch_indicator(indicator_code: str) -> tuple[list[dict[str, object]], str]:
    """Return available annual observations and the exact World Bank API URL."""

    query = urlencode(
        {
            "format": "json",
            "source": 2,
            "date": f"{START_YEAR}:{END_YEAR}",
            "per_page": 1000,
        }
    )
    api_url = (
        "https://api.worldbank.org/v2/country/ARG;URY/indicator/"
        f"{indicator_code}?{query}"
    )
    request = Request(api_url, headers={"User-Agent": "HIT-real-estate-project/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank response for {indicator_code}")
    return payload[1], api_url


def build_rows() -> list[dict[str, object]]:
    """Build sorted output rows for the two selected indicators."""

    rows: list[dict[str, object]] = []
    retrieved_on = date.today().isoformat()
    for indicator_code, (indicator_name, unit) in INDICATORS.items():
        observations, source_url = fetch_indicator(indicator_code)
        for observation in observations:
            value = observation.get("value")
            country_code = observation.get("countryiso3code")
            if value is None or country_code not in COUNTRIES:
                continue
            rows.append(
                {
                    "country_code": country_code,
                    "country": COUNTRIES[country_code],
                    "indicator_code": indicator_code,
                    "indicator": indicator_name,
                    "year": int(observation["date"]),
                    "value": round(float(value), 4),
                    "unit": unit,
                    "source": "World Bank, World Development Indicators",
                    "source_url": source_url,
                    "retrieved_on": retrieved_on,
                }
            )
    return sorted(rows, key=lambda row: (row["indicator_code"], row["country_code"], row["year"]))


def main() -> None:
    """Download the data and write the project CSV."""

    rows = build_rows()
    if not rows:
        raise RuntimeError("No World Bank observations were returned")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
