# Latin America Real Estate Price Estimation

**Core implementation status: complete.** This project collects residential
sale listings from Argentina and Uruguay, explores the resulting data, and
trains a leakage-safe baseline model that estimates an online asking price in
US dollars. The local submission materials are prepared; committing or
uploading the latest files is a separate handoff step.

- Repository: https://github.com/DanielLankry/Project-HIT---Real-Estate
- Full Hebrew guide: [`docs/PROJECT-GUIDE-HE.md`](docs/PROJECT-GUIDE-HE.md)
- Recorded walkthrough: `Explaining Video.mov`

> **Scope note:** the original proposal included sale and rental data, direct
> macroeconomic integration, and future-price forecasting. The implemented,
> defensible scope is sale listings only, with macroeconomic indicators used as
> context and a model that estimates current listing prices. The available data
> are a cross-sectional snapshot, so future-trend forecasting requires dated
> snapshots collected over time. This scope adjustment should be disclosed to
> and accepted by the course instructor.

## Submission Deliverables

| Deliverable | File or link | Status |
| --- | --- | --- |
| Specification document | `docs/Project-Specification.docx` | Complete |
| Code, datasets, and model | [GitHub repository](https://github.com/DanielLankry/Project-HIT---Real-Estate) | Complete locally; publish latest changes if required |
| Final presentation | `docs/Final-Project-Presentation.pptx` | Prepared locally |
| Recorded explanation (supplementary) | `Explaining Video.mov` | Prepared locally |
| Video slide deck (supplementary) | `HIT_Real_Estate_6_Slides.pptx` | Prepared locally |

## Results at a Glance

| Item | Current result |
| --- | ---: |
| Full cleaned dataset | 1,777 rows × 161 columns |
| Model-ready dataset | 1,689 rows × 63 columns |
| Training/test split | 1,351 / 338 rows |
| Selected model | Ridge Regression (`alpha=10`) |
| Mean 5-fold training CV R² | 0.628 |
| Held-out test R² | 0.475 |
| Held-out MAE | $105,473 |
| Held-out RMSE | $307,037 |
| Held-out MAPE | 27.2% |

The model is stronger for apartments than houses: held-out MAPE is 21.0% for
apartments and 47.0% for houses. These are regression errors, not an “accuracy
percentage,” and they describe the collected sample rather than both national
markets as a whole.

## Project Workflow

1. `code/scrape_real_estate_training_data.py` collects approved listings,
   normalizes fields, removes duplicates, and writes full and compact datasets.
2. `code/fetch_macroeconomic_data.py` retrieves annual inflation and GDP-growth
   context from the World Bank.
3. `code/eda_real_estate.ipynb` checks data quality, distributions, outliers,
   market coverage, correlations, and economic context.
4. `code/real_estate_model.ipynb` performs preprocessing inside scikit-learn
   pipelines, compares four regression baselines on training-only
   cross-validation, and evaluates the selected model once on the held-out set.
5. `code/predict_property.py` retrains the notebook-aligned Ridge model from the
   current compact CSV and estimates one property from terminal input.

Target-derived fields such as `price_per_sqm` are excluded from training to
prevent leakage. Numeric imputation, scaling, categorical imputation, and
one-hot encoding are fitted only on training data.

## Repository Layout

### `code`

- `scrape_real_estate_training_data.py` — collection, extraction, cleaning, and
  compact-dataset rebuilding.
- `fetch_macroeconomic_data.py` — World Bank context refresh.
- `eda_real_estate.ipynb` — exploratory analysis only.
- `real_estate_model.ipynb` — modeling source of truth, including validation
  tables, held-out metrics, and charts.
- `predict_property.py` — terminal inference entrypoint; no binary model artifact
  is required or saved.

### `data`

- `scraped_real_estate_training.csv` — 1,777-row detailed cleaned extraction.
- `scraped_real_estate_model_features.csv` — 1,689-row model-ready table.
- `scraped_real_estate_raw.jsonl` — raw page evidence for extraction debugging;
  intentionally not tracked because of its size.
- `macroeconomic_indicators.csv` — annual country-level context for 2015–2024.

The model data contains 1,110 Uruguay rows and 579 Argentina rows: 1,261
apartments, 427 houses, and 1 PH. The full extraction contains 1,185 InfoCasas,
581 ZonaProp, and 11 ArgenProp rows. Gallito contributes no rows because it
returns a Cloudflare challenge to normal requests.

### `docs`

- `Project-Specification.docx` — final goal, methodology, results, limitations,
  reproducibility instructions, and submission references.
- `Final-Project-Presentation.pptx` — final presentation.
- `PROJECT-GUIDE-HE.md` — full Hebrew project guide.
- `Project-Proposal-Template-HIT-project-center Maiora.docx.pdf` — original
  project proposal.
- `sitelist.md` — approved listing websites.
- `CrawlSteps.pdf` — early collection notes.

## Run the Project

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the saved analysis from top to bottom:

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace `
  --ExecutePreprocessor.timeout=900 `
  code/eda_real_estate.ipynb code/real_estate_model.ipynb
```

Estimate one current listing price:

```powershell
python code/predict_property.py
```

Use `AR` or `UY` for country and `Apartment` or `House` for property type. The
script identifies optional fields and warns when a location was not seen during
training.

Optional data refresh commands:

```powershell
python code/fetch_macroeconomic_data.py
python code/scrape_real_estate_training_data.py --rebuild-model-only
```

The scraper also supports targeted `--sites`, `--apartments-only`, and
incremental `--merge-existing` runs. It respects `robots.txt` by default; use
only approved sources and crawl limits appropriate to the site.

## Data Sources

- Approved listing sites: `docs/sitelist.md`
- World Bank inflation indicator: `FP.CPI.TOTL.ZG`
- World Bank real GDP-growth indicator: `NY.GDP.MKTP.KD.ZG`
- World Bank API documentation:
  https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

## Limitations and Next Research Step

- Prices are advertised asking prices, not completed transaction prices.
- The sample is dominated by Uruguay, InfoCasas, and apartments.
- A binary amenity value of `0` means “not mentioned,” not confirmed absence.
- Some Argentina location labels inherit inconsistent source hierarchies.
- Annual macroeconomic indicators provide context and are not listing-level
  predictors in the current model.

The project’s natural extension is scheduled collection of dated listing
snapshots with stable listing identifiers. Once a sufficiently long history
exists, prices can be aggregated by market segment, aligned with economic
indicators, and evaluated with honest time-based forecasting splits.

## Author

Daniel Lankry — `lankrydaniel7@gmail.com`
