# Tasks

## Completed
- [x] Improve apartment scraping and ML price features — 2026-06-18
  - Files modified: `code/scrape_real_estate_training_data.py`, `TASKS.md`, `AGENTS.md`
  - Summary: Added apartment-only crawl mode, more apartment seed pages, richer non-leaky ML features, numeric sanity checks, and cleaner location/expense extraction.
- [x] Fix zero-property crawl results for supported sites — 2026-06-18
  - Files modified: `code/scrape_real_estate_training_data.py`, `TASKS.md`, `AGENTS.md`
  - Summary: Improved site URL rules, link discovery, queue priority, blocked-page diagnostics, and deduplication so scrapeable sites return property rows and blocked sites are clearly reported.
- [x] Create readable real-estate scraper for model training — 2026-06-18
  - Files modified: `code/scrape_real_estate_training_data.py`, `TASKS.md`, `AGENTS.md`
  - Summary: Added a sitelist-driven scraper that crawls supported real-estate sites and writes a model-ready CSV plus raw JSONL debug data.

## In Progress

## Planned
