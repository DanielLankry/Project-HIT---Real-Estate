# Agent Notes

## Project Patterns
- Project files are currently organized in English folders: `docs` for docs/sitelist, `data` for data outputs, and `code` for Python code.
- The existing ML schema in `data/real estate data.csv` uses one USD price target plus many binary amenity columns.
- `code/scrape_real_estate_training_data.py` supports targeted crawl tests with `--sites <name>`.
- `code/scrape_real_estate_training_data.py` supports apartment-focused datasets with `--apartments-only`.

## Known Issues
- `pdftotext` via MiKTeX may fail in this sandbox because first-run setup writes outside the workspace.
- Data output files can be large, so stage them only when the dataset itself is meant to be published.
- Gallito returns Cloudflare HTTP 403 challenge pages to plain `requests`, so it is reported as blocked instead of scraped.

## Architecture Decisions
- The live scraper writes both a model-ready CSV and raw JSONL so extraction misses can be debugged without immediately re-crawling.
- The scraper respects robots.txt by default and uses `--ignore-robots` only when permission exists.
- Detail URLs are prioritized ahead of listing/search pages so small crawl limits collect actual property rows.
- The scraper intentionally avoids `price_per_sqm` in the training CSV because it is derived from the target price and would leak the label.
