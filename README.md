# Latin America Real Estate Data Project

מדריך מלא בעברית — מה הושלם, מה נשאר, מבנה התיקיות, הוראות הפעלה
והסבר מדדי המודל — נמצא ב־[`docs/PROJECT-GUIDE-HE.md`](docs/PROJECT-GUIDE-HE.md).

This project collects residential sale listings from Argentina and Uruguay,
explores the data, and builds a basic model that estimates listing prices in
US dollars.

The current model predicts prices for listings like those in the dataset. It
does **not** predict future market prices because the listing data is a snapshot,
not a historical time series.

## Project Status

| Part | Status | Main file |
| --- | --- | --- |
| Web scraping and cleaning | Complete | `code/scrape_real_estate_training_data.py` |
| Full and model-ready datasets | Complete | `data/*.csv` |
| Exploratory data analysis | Complete | `code/eda_real_estate.ipynb` |
| Basic listing-price model | Complete | `code/real_estate_model.ipynb` |
| English terminal price estimate | Complete | `code/predict_property.py` |
| Economic context | Complete | `data/macroeconomic_indicators.csv` |
| Project specification | Complete | `docs/Project-Specification.docx` |
| Final presentation | Complete | `docs/Final-Project-Presentation.pptx` |
| Future market-trend forecasting | Future work | Requires historical listing snapshots |

## Folders and Files

### `code`

- `scrape_real_estate_training_data.py` crawls approved real-estate sites,
  extracts listing fields, removes duplicates, and writes the datasets.
- `fetch_macroeconomic_data.py` downloads basic inflation and GDP-growth data
  for Argentina and Uruguay from the World Bank API.
- `eda_real_estate.ipynb` explains the dataset, data quality, price distribution,
  correlations, and economic context with simple charts.
- `real_estate_model.ipynb` prepares the data, compares basic regression models,
  evaluates the selected model, and explains the results.
- `predict_property.py` asks for one property's details in English, trains the
  current Ridge baseline on the model CSV, and prints an estimated price in USD.
  It does not create or require a separate saved-model file.

### `data`

- `scraped_real_estate_training.csv` is the full cleaned extraction dataset. It
  keeps detailed fields for checking scraper results.
- `scraped_real_estate_model_features.csv` is the smaller model-ready dataset.
  It keeps residential rows with usable price and size information.
- `scraped_real_estate_raw.jsonl` stores raw page evidence for debugging. It is
  large and is intentionally not tracked by Git.
- `macroeconomic_indicators.csv` contains annual inflation and GDP growth for
  2015-2024 where values are available. It provides context only; it is not
  joined to individual listings.

### `docs`

- `Project-Proposal-Template-HIT-project-center Maiora.docx.pdf` is the original
  proposal.
- `Project-Specification.docx` explains the goal, data, method, results,
  limitations, and how to run the project.
- `Final-Project-Presentation.pptx` summarizes the project for presentation.
- `sitelist.md` lists the approved real-estate websites.
- `CrawlSteps.pdf` contains earlier crawl notes.

### Other folders

- `.github/workflows` contains the automatic notebook validation workflow.
- `requirements.txt` lists the Python packages needed by the scraper, notebooks,
  and terminal predictor.

## Dataset Summary

- Full dataset: **1,777 rows and 161 columns**.
- Model dataset: **1,689 rows and 63 columns**.
- Countries: **1,110 Uruguay rows** and **579 Argentina rows** in the model data.
- Property types: **1,261 apartments**, **427 houses**, and **1 PH** in the
  model data.
- Full-data sources: **1,185 InfoCasas**, **581 ZonaProp**, and **11 ArgenProp**
  rows.
- Gallito blocks normal requests with a Cloudflare challenge, so it currently
  contributes no rows.

Important: a binary amenity value of `0` means that the listing did not mention
the amenity. It does not prove that the property does not have it.

## Basic Results

The notebook selects Ridge Regression using training-only cross-validation. Its
mean training cross-validation R2 is **0.628**. On the held-out test set it reports:

- R2: **0.475**
- Mean absolute error: **$105,473**
- Root mean squared error: **$307,037**
- Mean absolute percentage error: **27.2%**

These are baseline regression results for the collected listings, not a claim
about the whole Argentina and Uruguay real-estate markets. The extra rows improve
Argentina-house coverage, but the broader held-out set is harder and the headline
score did not improve merely because more data was added.

## How to Run

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Refresh the economic context:

```powershell
python code/fetch_macroeconomic_data.py
```

Rebuild the compact model dataset without crawling:

```powershell
python code/scrape_real_estate_training_data.py --rebuild-model-only
```

Open and run these notebooks from top to bottom:

1. `code/eda_real_estate.ipynb`
2. `code/real_estate_model.ipynb`

Estimate one current listing price by entering the details in English:

```powershell
python code/predict_property.py
```

Use `AR` or `UY` for country and `Apartment` or `House` for property type. The
script explains which fields are optional and warns when a location was not seen
during training. It retrains from the current CSV each time, so newly collected
and rebuilt data is included automatically.

## Data Sources

- Real-estate websites are listed in `docs/sitelist.md`.
- Inflation indicator: World Bank `FP.CPI.TOTL.ZG`.
- GDP-growth indicator: World Bank `NY.GDP.MKTP.KD.ZG`.
- World Bank API documentation: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Main Limitations and Next Step

The sample is dominated by Uruguay, InfoCasas, and apartments. Prices are asking
prices rather than confirmed sale prices. The apartment test segment is stronger
than the house segment (MAPE 21.0% versus 47.0%), so house estimates need extra
caution. Some Argentina location labels inherited inconsistent source hierarchy,
so an unseen-location warning means the predictor falls back to the other
features. The economic indicators are annual country-level context and are not
used as listing features.

The next meaningful step is to collect dated snapshots of the same markets over
time. After enough history exists, the project can test true future-trend
forecasting without pretending that a snapshot is a time series.
