"""
Simple real-estate scraper for model-training data.

What it does:
1. Reads the web sites from ../docs/sitelist.md
2. Crawls sale/listing pages from those sites
3. Extracts useful model features: price, rooms, area, location, amenities, etc.
4. Writes:
   - ../data/scraped_real_estate_training.csv
   - ../data/scraped_real_estate_raw.jsonl

Install dependencies:
    pip install requests beautifulsoup4

Example:
    python code/scrape_real_estate_training_data.py --max-pages-per-site 300

Use a larger number when you want more training data:
    python code/scrape_real_estate_training_data.py --max-pages-per-site 2000 --delay 1.5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html as html_lib
import json
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag
from urllib.robotparser import RobotFileParser

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: pip install requests beautifulsoup4"
    ) from exc


# Approximate conversion rates. Update these before making a final dataset.
ARS_PER_USD = 1500
UYU_PER_USD = 40

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITELIST = PROJECT_ROOT / "docs" / "sitelist.md"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "scraped_real_estate_training.csv"
DEFAULT_RAW_JSONL = PROJECT_ROOT / "data" / "scraped_real_estate_raw.jsonl"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class SiteConfig:
    """Describe one supported site and its crawl rules.

    The scraper uses this to keep site-specific URLs and detail-page patterns
    separate from the generic crawling and extraction logic.
    """

    name: str
    domain_key: str
    country: str
    sale_paths: tuple[str, ...]
    apartment_paths: tuple[str, ...]
    detail_patterns: tuple[re.Pattern[str], ...]
    useful_words: tuple[str, ...]


SITE_CONFIGS = (
    SiteConfig(
        name="argenprop",
        domain_key="argenprop.com",
        country="AR",
        sale_paths=(
            "/departamentos/venta/argentina",
            "/departamentos/venta/capital-federal",
            "/departamentos/venta/buenos-aires",
            "/departamentos/venta/gba-norte",
            "/departamentos/venta/gba-sur",
            "/departamentos/venta/gba-oeste",
            "/departamentos/venta/palermo",
            "/departamentos/venta/belgrano",
            "/departamentos/venta/caballito",
            "/casas/venta/argentina",
            "/terrenos/venta/argentina",
            "/departamentos-en-venta",
            "/casas-en-venta",
            "/terrenos-en-venta",
            "/inmuebles-en-venta",
            "/",
        ),
        apartment_paths=(
            "/departamentos/venta/argentina",
            "/departamentos/venta/capital-federal",
            "/departamentos/venta/buenos-aires",
            "/departamentos/venta/gba-norte",
            "/departamentos/venta/gba-sur",
            "/departamentos/venta/gba-oeste",
            "/departamentos/venta/palermo",
            "/departamentos/venta/belgrano",
            "/departamentos/venta/caballito",
            "/departamentos-en-venta",
        ),
        detail_patterns=(re.compile(r"--\d+(?:$|[/?])"),),
        useful_words=("venta", "departamento", "casa", "terreno", "inmueble"),
    ),
    SiteConfig(
        name="zonaprop",
        domain_key="zonaprop.com.ar",
        country="AR",
        sale_paths=(
            "/departamentos-venta.html",
            "/departamentos-venta-capital-federal.html",
            "/departamentos-venta-gba-norte.html",
            "/departamentos-venta-gba-sur.html",
            "/departamentos-venta-gba-oeste.html",
            "/departamentos-venta-palermo.html",
            "/departamentos-venta-belgrano.html",
            "/departamentos-venta-caballito.html",
            "/casas-venta.html",
            "/terrenos-venta.html",
            "/inmuebles-venta.html",
        ),
        apartment_paths=(
            "/departamentos-venta.html",
            "/departamentos-venta-capital-federal.html",
            "/departamentos-venta-gba-norte.html",
            "/departamentos-venta-gba-sur.html",
            "/departamentos-venta-gba-oeste.html",
            "/departamentos-venta-palermo.html",
            "/departamentos-venta-belgrano.html",
            "/departamentos-venta-caballito.html",
        ),
        detail_patterns=(
            re.compile(r"/propiedades/.*\.html(?:$|[/?])"),
            re.compile(r"/.*-\d{6,}\.html(?:$|[/?])"),
        ),
        useful_words=("venta", "departamento", "casa", "terreno", "propiedades"),
    ),
    SiteConfig(
        name="infocasas",
        domain_key="infocasas.com.uy",
        country="UY",
        sale_paths=(
            "/venta/apartamentos",
            "/venta/apartamentos/montevideo",
            "/venta/apartamentos/punta-del-este",
            "/venta/apartamentos/pocitos",
            "/venta/apartamentos/cordon",
            "/venta/apartamentos/malvin",
            "/venta/apartamentos/carrasco",
            "/venta/inmuebles",
            "/venta/casas",
            "/venta/terrenos",
        ),
        apartment_paths=(
            "/venta/apartamentos",
            "/venta/apartamentos/montevideo",
            "/venta/apartamentos/punta-del-este",
            "/venta/apartamentos/pocitos",
            "/venta/apartamentos/cordon",
            "/venta/apartamentos/malvin",
            "/venta/apartamentos/carrasco",
        ),
        detail_patterns=(re.compile(r"/\d{6,}(?:$|[/?])"),),
        useful_words=("venta", "apartamento", "casa", "terreno", "inmuebles"),
    ),
    SiteConfig(
        name="gallito",
        domain_key="gallito.com.uy",
        country="UY",
        sale_paths=(
            "/inmuebles/apartamentos/venta",
            "/inmuebles/apartamentos/venta/montevideo",
            "/inmuebles/apartamentos/venta/punta-del-este",
            "/inmuebles/casas/venta",
            "/inmuebles/terrenos/venta",
            "/inmuebles/venta",
        ),
        apartment_paths=(
            "/inmuebles/apartamentos/venta",
            "/inmuebles/apartamentos/venta/montevideo",
            "/inmuebles/apartamentos/venta/punta-del-este",
        ),
        detail_patterns=(
            re.compile(r"/[^/?#]+-inmuebles-\d+(?:$|[/?])"),
            re.compile(r"/inmuebles/.+\d+(?:$|[/?])"),
            re.compile(r"/aviso/.+\d+(?:$|[/?])"),
        ),
        useful_words=("venta", "apartamento", "casa", "terreno", "inmuebles"),
    ),
)


BASE_COLUMNS = [
    "source_url",
    "source_domain",
    "listing_id",
    "listing_title",
    "description",
    "scraped_at",
    "data_source",
    "country",
    "operation_type",
    "property_type",
    "property_subtype",
    "publication_type",
    "construction_stage",
    "province",
    "city",
    "neighborhood",
    "location_key",
    "street",
    "balcony_type",
    "view_type",
    "flooring_type",
    "orientation",
    "price_usd",
    "price_original",
    "price_currency",
    "expenses_usd",
    "expenses_original",
    "expenses_currency",
    "total_rooms",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "toilets",
    "covered_area_sqm",
    "total_area_sqm",
    "semicovered_area_sqm",
    "uncovered_area_sqm",
    "lot_frontage_sqm",
    "lot_length_sqm",
    "age_years",
    "number_of_floors_in_unit",
    "floors_in_building",
    "units_per_floor",
    "floor_number",
    "latitude",
    "longitude",
    "distance_to_sea_blocks",
    "effective_area_sqm",
    "area_per_room_sqm",
    "bedrooms_per_bathroom",
    "bathrooms_per_bedroom",
    "amenity_count",
    "core_feature_count",
    "title_length_chars",
    "description_length_chars",
    "title_word_count",
    "description_word_count",
    "area_bucket",
    "age_bucket",
    "floor_bucket",
    "room_bucket",
]

AMENITY_COLUMNS = [
    "accepts_swap",
    "has_apartment_balcony_view",
    "has_air_conditioning",
    "has_alarm",
    "has_attic",
    "has_back_garden",
    "has_backyard",
    "has_balcony",
    "has_basement",
    "has_bbq_house",
    "has_breakfast_room",
    "has_cable_tv",
    "has_cleaning_service",
    "has_closet",
    "has_coordinates",
    "has_cooling",
    "has_description",
    "has_dining_room",
    "has_doorman",
    "has_double_circulation",
    "has_dressing_room",
    "has_electricity",
    "has_elevator",
    "has_extra_bedroom",
    "has_exact_address",
    "has_expenses",
    "has_football_court",
    "has_front_garden",
    "has_gallery",
    "has_garage_room",
    "has_garden",
    "has_grill",
    "has_gym",
    "has_hall",
    "has_heating",
    "has_hot_water",
    "has_internet",
    "has_jacuzzi",
    "has_kitchen",
    "has_kitchen_dining",
    "has_laundry_room",
    "has_library",
    "has_living_dining_room",
    "has_living_room",
    "has_maid_room",
    "has_master_suite",
    "has_multipurpose_room",
    "has_natural_gas",
    "has_office",
    "has_patio",
    "has_phone_line",
    "has_play_room",
    "has_pool",
    "has_private_landing",
    "has_property_tax_service",
    "has_radiant_floor",
    "has_reception",
    "has_rentas_paid",
    "has_running_water",
    "has_sauna",
    "has_sea_view",
    "has_security",
    "has_storage_room",
    "has_studio_room",
    "has_tennis_court",
    "has_terrace",
    "has_toilette_room",
    "has_walk_in_closet",
    "has_water_heater",
    "has_wood_stove",
    "has_virtual_tour",
    "is_accessible",
    "is_apartment",
    "is_back_facing",
    "is_bright",
    "is_currently_occupied",
    "is_development_project",
    "is_duplex",
    "is_high_floor",
    "is_front_facing",
    "is_furnished",
    "is_gated_community",
    "is_negotiable",
    "is_new_construction",
    "is_near_beach",
    "is_near_park",
    "is_near_sea",
    "is_near_subway",
    "is_owner_direct",
    "is_opportunity",
    "is_penthouse",
    "is_social_housing",
    "is_studio_apartment",
    "is_under_construction",
    "mortgage_eligible",
    "pets_allowed",
    "professional_use_allowed",
]

CSV_COLUMNS = BASE_COLUMNS + AMENITY_COLUMNS
AMENITY_COUNT_EXCLUDE = {
    "has_coordinates",
    "has_description",
    "has_exact_address",
    "has_expenses",
}
NUMERIC_COLUMNS = {
    "price_usd",
    "price_original",
    "total_rooms",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    "toilets",
    "covered_area_sqm",
    "total_area_sqm",
    "semicovered_area_sqm",
    "uncovered_area_sqm",
    "lot_frontage_sqm",
    "lot_length_sqm",
    "age_years",
    "number_of_floors_in_unit",
    "floors_in_building",
    "units_per_floor",
    "floor_number",
    "latitude",
    "longitude",
    "distance_to_sea_blocks",
    "expenses_usd",
    "expenses_original",
    "effective_area_sqm",
    "area_per_room_sqm",
    "bedrooms_per_bathroom",
    "bathrooms_per_bedroom",
    "amenity_count",
    "core_feature_count",
    "title_length_chars",
    "description_length_chars",
    "title_word_count",
    "description_word_count",
}

PROPERTY_TYPE_PATTERNS = (
    (re.compile(r"\b(departamento|apartamento|apartment|monoambiente)\b", re.I), "Apartment"),
    (re.compile(r"\b(casa|house|chalet)\b", re.I), "House"),
    (re.compile(r"\b(ph)\b", re.I), "PH"),
    (re.compile(r"\b(terreno|lote|land)\b", re.I), "Land"),
    (re.compile(r"\b(oficina|office)\b", re.I), "Office"),
    (re.compile(r"\b(local comercial|local|retail)\b", re.I), "Commercial Space"),
    (re.compile(r"\b(cochera|garage|garaje)\b", re.I), "Parking Space"),
    (re.compile(r"\b(campo|chacra|farm)\b", re.I), "Farm"),
)

AMENITY_PATTERNS = {
    "has_apartment_balcony_view": r"balc[oó]n\s+(?:al\s+)?(?:frente|contrafrente|vista)",
    "has_air_conditioning": r"aire\s+acondicionado|air\s*conditioning",
    "has_alarm": r"\balarma\b|alarm",
    "has_attic": r"\baltillo\b|attic",
    "has_back_garden": r"jard[ií]n\s+fondo",
    "has_backyard": r"\bfondo\b|backyard",
    "has_balcony": r"balc[oó]n|balcony",
    "has_basement": r"s[oó]tano|basement",
    "has_bbq_house": r"\bbarbacoa\b|\bquincho\b",
    "has_breakfast_room": r"comedor\s+diario",
    "has_cable_tv": r"cable\s*tv|video\s*cable",
    "has_cleaning_service": r"servicio\s+de\s+limpieza|limpieza",
    "has_closet": r"\bplacard(?:s)?\b|closet",
    "has_cooling": r"refrigeraci[oó]n",
    "has_dining_room": r"\bcomedor\b|dining",
    "has_doorman": r"portero|conserje|encargado|porter[ií]a",
    "has_double_circulation": r"doble\s+circulaci[oó]n",
    "has_dressing_room": r"\bvestuario\b",
    "has_electricity": r"electricidad|electricity",
    "has_elevator": r"ascensor(?:es)?|elevator",
    "has_extra_bedroom": r"dormitorio\s+extra",
    "has_football_court": r"cancha\s+de\s+f[uú]tbol",
    "has_front_garden": r"jard[ií]n\s+frente",
    "has_gallery": r"\bgaler[ií]a\b",
    "has_garage_room": r"\bcochera\b|\bgaraje\b|garage",
    "has_garden": r"jard[ií]n|garden",
    "has_grill": r"parrillero|parrilla|grill",
    "has_gym": r"gimnasio|\bgym\b",
    "has_hall": r"\bhall\b",
    "has_heating": r"calefacci[oó]n|heating",
    "has_hot_water": r"agua\s+caliente",
    "has_internet": r"internet|wifi|wi-fi",
    "has_jacuzzi": r"hidromasaje|jacuzzi",
    "has_kitchen": r"\bcocina\b|kitchen",
    "has_kitchen_dining": r"cocina\s+comedor",
    "has_laundry_room": r"lavadero|laundry",
    "has_library": r"biblioteca|library",
    "has_living_dining_room": r"living\s+comedor",
    "has_living_room": r"\bliving\b",
    "has_maid_room": r"dependencia\s+de\s+servicio|dependencia",
    "has_master_suite": r"suite|dormitorio\s+en\s+suite",
    "has_multipurpose_room": r"\bsum\b|sal[oó]n\s+de\s+usos\s+m[uú]ltiples|sal[oó]n\s+comunal",
    "has_natural_gas": r"gas\s+natural",
    "has_office": r"escritorio|estudio|home\s*office",
    "has_patio": r"\bpatio\b",
    "has_phone_line": r"tel[eé]fono|phone",
    "has_play_room": r"playroom|sala\s+de\s+juegos",
    "has_pool": r"piscina|pileta|pool",
    "has_private_landing": r"\bpalier\b",
    "has_property_tax_service": r"\babl\b|contribuci[oó]n",
    "has_radiant_floor": r"losa\s+radiante",
    "has_reception": r"recepci[oó]n",
    "has_rentas_paid": r"\brentas\b",
    "has_running_water": r"agua\s+corriente",
    "has_sauna": r"\bsauna\b",
    "has_sea_view": r"vista\s+al\s+mar|sea\s+view",
    "has_security": r"seguridad|vigilancia|24\s*hs|security",
    "has_storage_room": r"baulera|dep[oó]sito|storage",
    "has_studio_room": r"\bestudio\b",
    "has_tennis_court": r"cancha\s+de\s+tenis",
    "has_terrace": r"terraza|terrace",
    "has_toilette_room": r"toilette",
    "has_virtual_tour": r"tour\s+virtual|recorrido\s+virtual|video\s+tour",
    "has_walk_in_closet": r"vestidor|walk[-\s]?in",
    "has_water_heater": r"calef[oó]n|termotanque|water\s+heater",
    "has_wood_stove": r"estufa\s+a\s+le[nñ]a",
    "is_accessible": r"acceso\s+para\s+personas|movilidad\s+reducida|accessible",
    "is_back_facing": r"contrafrente",
    "is_bright": r"luminos[oa]|bright",
    "is_currently_occupied": r"propiedad\s+ocupada",
    "is_development_project": r"emprendimiento|desarrollo|proyecto",
    "is_duplex": r"\bd[uú]plex\b|\bduplex\b",
    "is_front_facing": r"\bfrente\b",
    "is_furnished": r"amoblado|amueblado|furnished",
    "is_gated_community": r"barrio\s+privado|country\s+club",
    "is_near_beach": r"playa|beach",
    "is_near_park": r"parque|plaza|park",
    "is_near_sea": r"\bmar\b|sea",
    "is_near_subway": r"subte|metro|estaci[oó]n",
    "is_negotiable": r"negociable",
    "is_opportunity": r"oportunidad",
    "is_penthouse": r"penthouse|[áa]tico",
    "is_social_housing": r"vivienda\s+social",
    "is_studio_apartment": r"monoambiente|studio",
    "mortgage_eligible": r"apto\s+cr[eé]dito|acepta\s+banco|financiaci[oó]n",
    "pets_allowed": r"mascotas|pet\s*friendly",
    "professional_use_allowed": r"apto\s+profesional|apto\s+para\s+oficina",
    "accepts_swap": r"acepta\s+permuta|permuta",
}

BLOCKED_PAGE_PATTERNS = (
    re.compile(r"verify that you(?:'| a)?re not a robot", re.I),
    re.compile(r"captcha", re.I),
    re.compile(r"just a moment", re.I),
    re.compile(r"cloudflare", re.I),
    re.compile(r"javascript is disabled", re.I),
    re.compile(r"access denied", re.I),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for crawl size, paths, and politeness.

    These options make it easy to collect a small test dataset or a larger
    model-training dataset without editing the script.
    """

    parser = argparse.ArgumentParser(
        description="Scrape real-estate sites from sitelist.md into an ML-ready CSV."
    )
    parser.add_argument("--sitelist", type=Path, default=DEFAULT_SITELIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_JSONL)
    parser.add_argument("--max-pages-per-site", type=int, default=300)
    parser.add_argument("--max-properties-per-site", type=int, default=200)
    parser.add_argument(
        "--apartments-only",
        action="store_true",
        help="Use apartment-focused seeds and keep only apartment listings.",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Optional site names/domains to crawl, for example: --sites gallito argenprop",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt. Use only if you have permission.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sites and seed URLs without downloading pages.",
    )
    return parser.parse_args()


def read_site_urls(path: Path) -> list[str]:
    """Read unique HTTP URLs from the Markdown sitelist file.

    The project keeps the target sites in a simple document, so this function
    lets non-code edits change the crawl targets.
    """

    text = path.read_text(encoding="utf-8")
    urls = re.findall(r"https?://[^\s)]+", text)
    return sorted(dict.fromkeys(url.strip().rstrip("/") for url in urls))


def wanted_site(url: str, config: SiteConfig, selected_sites: list[str] | None) -> bool:
    """Return True when a site passes the optional --sites filter.

    Targeted crawls make debugging one website faster without changing
    sitelist.md or the configured site rules.
    """

    if not selected_sites:
        return True
    choices = {site.lower().removeprefix("www.") for site in selected_sites}
    host = comparable_host(url)
    return config.name.lower() in choices or config.domain_key.lower() in choices or host in choices


def config_for_url(url: str) -> SiteConfig | None:
    """Find the matching site configuration for a base URL.

    Unsupported sites are skipped instead of failing the entire scrape.
    """

    host = urlparse(url).netloc.lower()
    for config in SITE_CONFIGS:
        if config.domain_key in host:
            return config
    return None


def apartment_word_in_url(url: str) -> bool:
    """Return True when a URL looks apartment-related.

    This gives --apartments-only a cheap URL filter before downloading pages.
    """

    lower = url.lower()
    return any(word in lower for word in ("departamento", "departamentos", "apartamento", "apartamentos"))


def build_seed_urls(base_url: str, config: SiteConfig, apartments_only: bool = False) -> list[str]:
    """Create the first sale/listing pages to crawl for one site.

    The sitelist contains domains only, so the scraper adds known sale paths
    for apartments, houses, land, and broad real-estate searches.
    """

    paths = config.apartment_paths if apartments_only else config.sale_paths
    return [urljoin(base_url.rstrip("/") + "/", path.lstrip("/")) for path in paths]


def make_session() -> requests.Session:
    """Create a requests session with stable headers.

    Reusing one session is faster and keeps the crawler identity consistent.
    """

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es,en;q=0.8",
        }
    )
    return session


def load_robot_parser(
    session: requests.Session, base_url: str, timeout: float
) -> RobotFileParser | None:
    """Load robots.txt rules for a site when available.

    Respecting robots.txt keeps the crawler conservative and avoids collecting
    pages the site has explicitly disallowed.
    """

    robots_url = urljoin(base_url.rstrip("/") + "/", "robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = session.get(robots_url, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code >= 400:
        return None
    parser.parse(response.text.splitlines())
    return parser


def normalize_url(url: str) -> str:
    """Normalize a discovered link before adding it to the queue.

    This removes fragments and tracking queries so the crawler does not waste
    requests on duplicate versions of the same page.
    """

    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    host = parsed.netloc.lower()
    path = re.sub(r"/+", "/", parsed.path)

    # Keep only pagination-style query parameters. Tracking queries create duplicates.
    query = parsed.query if re.search(r"\b(page|pagina|pag|offset)=", parsed.query, re.I) else ""
    return urlunparse((parsed.scheme, host, path, "", query, ""))


def comparable_host(url: str) -> str:
    """Normalize a URL host for same-site comparisons.

    Real sites often mix www and non-www links, so removing only the leading
    www keeps those links crawlable without allowing unrelated domains.
    """

    return urlparse(url).netloc.lower().removeprefix("www.")


def same_domain(url: str, base_url: str) -> bool:
    """Return True when a link stays inside the current site.

    The crawler is intentionally limited to the sites listed in sitelist.md.
    """

    host = comparable_host(url)
    base_host = comparable_host(base_url)
    return host == base_host or host.endswith("." + base_host)


def looks_like_detail_url(url: str, config: SiteConfig) -> bool:
    """Identify whether a URL probably belongs to one property listing.

    Detail pages are the pages that can become rows in the training CSV.
    """

    return any(pattern.search(url) for pattern in config.detail_patterns)


def looks_useful_url(url: str, config: SiteConfig, apartments_only: bool = False) -> bool:
    """Keep only listing/search/detail URLs worth crawling.

    This avoids assets, login pages, and unrelated site sections.
    """

    lower = url.lower()
    if any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf")):
        return False
    if any(bad in lower for bad in ("/login", "/signin", "/contact", "/contacto", "/ayuda")):
        return False
    if apartments_only and not apartment_word_in_url(url):
        return False
    if looks_like_detail_url(url, config):
        return "alquiler" not in lower
    return "venta" in lower and any(word in lower for word in config.useful_words)


def clean_discovered_url(raw_value: Any, current_url: str) -> str:
    """Convert a raw HTML/JSON URL value into an absolute normalized URL.

    Listing pages often hide detail URLs in image links, data attributes, or
    escaped JSON, so this helper makes those values usable by the crawler.
    """

    if raw_value is None:
        return ""
    value = html_lib.unescape(str(raw_value)).replace("\\/", "/").strip()
    value = value.strip("\"'()[]{}<>.,;")
    if not value or value.startswith(("mailto:", "tel:", "javascript:")):
        return ""
    if any(character.isspace() for character in value):
        return ""
    if not value.startswith(("http://", "https://", "/", "./", "../")):
        return ""
    return normalize_url(urljoin(current_url, value))


def raw_html_urls(html: str) -> list[str]:
    """Extract URL-looking strings from raw HTML.

    This catches real-estate card URLs that appear inside scripts or image
    metadata instead of regular anchor tags.
    """

    text = html_lib.unescape(html).replace("\\/", "/")
    absolute_urls = re.findall(r"https?://[^\"'<>\s)]+", text)
    relative_urls = re.findall(r"""["'](/[^"'<>]*?(?:venta|inmuebles)[^"'<>]*?)["']""", text, re.I)
    return absolute_urls + relative_urls


def polite_sleep(delay: float) -> None:
    """Pause between requests with small random jitter.

    The delay reduces pressure on the source sites during larger scrapes.
    """

    if delay <= 0:
        return
    time.sleep(delay + random.uniform(0, delay * 0.35))


def fetch_html(
    session: requests.Session, url: str, timeout: float
) -> tuple[str | None, int | None]:
    """Download one HTML page and return its text plus status code.

    Network errors are logged and converted to None so one bad URL does not
    stop the whole crawl.
    """

    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"  fetch failed: {url} ({exc})")
        return None, None
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        print(f"  skipped non-HTML response: {url} ({response.status_code}, {content_type})")
        return None, response.status_code
    return response.text, response.status_code


def soup_text(soup: BeautifulSoup) -> str:
    """Extract clean visible page text from parsed HTML.

    Feature extraction works mostly from readable Spanish text, so scripts and
    style blocks are removed first.
    """

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def is_blocked_page(page_text: str) -> bool:
    """Detect anti-bot or JavaScript-required pages.

    A blocked page has no usable property links, so the crawler reports it
    separately from a normal zero-result crawl.
    """

    sample = page_text[:2000]
    return any(pattern.search(sample) for pattern in BLOCKED_PAGE_PATTERNS)


def meta_content(soup: BeautifulSoup, *names: str) -> str:
    """Read the first matching meta tag content value.

    Real-estate pages often expose better titles and descriptions in OpenGraph
    or SEO meta tags than in visible page text.
    """

    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return clean_text(tag["content"])
    return ""


def clean_text(value: Any) -> str:
    """Normalize whitespace in extracted text values.

    CSV cells should stay readable even when source HTML has irregular spacing.
    """

    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_json_blocks(soup: BeautifulSoup) -> list[Any]:
    """Parse JSON and JSON-LD script blocks from a page.

    Structured data often contains price, address, and geolocation fields that
    are more reliable than plain text extraction.
    """

    blocks: list[Any] = []
    for script in soup.find_all("script", type=re.compile(r"ld\+json|json", re.I)):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return blocks


def walk_json(value: Any) -> list[Any]:
    """Flatten nested JSON into a list of every nested value.

    This keeps the extractor simple because sites place useful fields at
    different depths inside JSON-LD or application JSON.
    """

    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(walk_json(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(walk_json(child))
    return values


def json_values(json_blocks: list[Any], wanted_keys: set[str]) -> list[Any]:
    """Collect values for selected keys from all parsed JSON blocks.

    The helper supports generic extraction across several website formats.
    """

    found: list[Any] = []
    for node in walk_json(json_blocks):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if key.lower() in wanted_keys:
                found.append(value)
    return found


def first_json_value(json_blocks: list[Any], *keys: str) -> str:
    """Return the first scalar JSON value for any of the requested keys.

    Most fields only need one good structured value, such as price or title.
    """

    wanted = {key.lower() for key in keys}
    for value in json_values(json_blocks, wanted):
        if isinstance(value, (str, int, float)) and str(value).strip():
            return clean_text(value)
    return ""


def parse_number(value: Any) -> float | None:
    """Parse the first decimal-like number from text.

    Spanish pages mix comma and dot separators, so the parser accepts both.
    """

    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"-?\d[\d.,]*", text)
    if not match:
        return None
    number = match.group(0)
    if "," in number and "." in number:
        number = number.replace(".", "").replace(",", ".")
    elif "," in number:
        number = number.replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    """Parse a rounded integer from text.

    Room counts, floors, and areas are stored as integers for model features.
    """

    number = parse_number(value)
    if number is None:
        return None
    return int(round(number))


def parse_price_amount(value: Any) -> int | None:
    """Parse a money amount while treating separators as thousands markers.

    Prices such as 110.000 should become 110000 instead of 110.
    """

    if value is None:
        return None
    text = str(value)
    match = re.search(r"\d[\d.,]*", text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(0))
    return int(digits) if digits else None


def convert_to_usd(amount: int, currency: str, country: str) -> int:
    """Convert original listing prices into approximate USD.

    The model CSV uses one comparable target price column across Argentina and
    Uruguay, while keeping the original amount and currency too.
    """

    currency = currency.upper().replace("U$S", "USD").replace("US$", "USD")
    if currency in {"USD", "US DOLLAR", "DOLLAR"}:
        return amount
    if currency in {"ARS", "AR$", "$"} and country == "AR":
        return round(amount / ARS_PER_USD)
    if currency in {"UYU", "$U", "$"} and country == "UY":
        return round(amount / UYU_PER_USD)
    return amount


def extract_price(text: str, json_blocks: list[Any], country: str) -> dict[str, Any]:
    """Extract price fields from structured JSON first, then visible text.

    Price is the likely training target, so the scraper skips pages where no
    usable sale price can be found.
    """

    json_price = first_json_value(json_blocks, "price", "lowPrice", "highPrice")
    json_currency = first_json_value(json_blocks, "priceCurrency", "currency")
    if json_price:
        amount = parse_price_amount(json_price)
        if amount:
            currency = json_currency or "USD"
            return {
                "price_original": amount,
                "price_currency": currency.upper(),
                "price_usd": convert_to_usd(amount, currency, country),
            }

    price_patterns = (
        (r"(?:U\$S|US\$|USD)\s*([\d.,]+)", "USD"),
        (r"(?:ARS|AR\$)\s*([\d.,]+)", "ARS"),
        (r"(?:UYU|\$U)\s*([\d.,]+)", "UYU"),
        (r"\$\s*([\d.,]{5,})", "$"),
    )
    for pattern, currency in price_patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        amount = parse_price_amount(match.group(1))
        if amount:
            return {
                "price_original": amount,
                "price_currency": currency,
                "price_usd": convert_to_usd(amount, currency, country),
            }
    return {}


def extract_expenses(row: dict[str, Any], text: str, country: str) -> None:
    """Extract monthly expenses or common charges when listed.

    Expenses are a strong price-prediction feature for apartments because they
    often reflect building amenities, services, and maintenance level.
    """

    currency_pattern = r"(U\$S|US\$|USD|ARS|AR\$|UYU|\$U|\$)?"
    patterns = (
        rf"(?:expensas|gastos\s+comunes)[^\d$U]{{0,30}}{currency_pattern}\s*([\d.,]+)",
        rf"{currency_pattern}\s*([\d.,]+)\s*(?:de\s*)?(?:expensas|gastos\s+comunes)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        currency = clean_text(match.group(1)) or "$"
        amount = parse_price_amount(match.group(2))
        if not amount or amount < 100:
            continue
        expenses_usd = convert_to_usd(amount, currency, country)
        if amount == row.get("price_original") or expenses_usd == row.get("price_usd"):
            continue
        row["expenses_original"] = amount
        row["expenses_currency"] = currency.upper()
        row["expenses_usd"] = expenses_usd
        return


def clean_location_part(value: Any) -> str:
    """Return a location label only when it looks like a real place name.

    This prevents long description fragments from becoming high-cardinality
    location categories in the training CSV.
    """

    text = clean_text(value).strip(" .:-")
    if not text or len(text) > 70:
        return ""
    bad_words = (
        "dormitorio",
        "dormitorios",
        "baño",
        "baños",
        "construcción",
        "descubra",
        "edificio",
        "exclusivo",
        "pregúntale",
        "ref #",
        "referencia",
        "venta en",
    )
    if any(word in text.lower() for word in bad_words):
        return ""
    return text


def set_location_from_match(row: dict[str, Any], match: re.Match[str]) -> None:
    """Set neighborhood and city from a regex match when both parts are sane.

    Listing titles can contain many commas, so location extraction must reject
    long marketing phrases.
    """

    neighborhood = clean_location_part(match.group(1))
    city = clean_location_part(match.group(2))
    if not neighborhood or not city:
        return
    if not row.get("neighborhood"):
        row["neighborhood"] = neighborhood
    if not row.get("city"):
        row["city"] = city


def title_from_slug(slug: str) -> str:
    """Convert a URL slug into a readable title-cased label.

    URL slugs are a useful fallback when page metadata does not expose clean
    neighborhood/city fields.
    """

    words = [word for word in slug.split("-") if word and not word.isdigit()]
    return " ".join(word.capitalize() for word in words)


def extract_location_from_url(row: dict[str, Any], url: str) -> None:
    """Fill location from common listing URL slugs when other methods fail.

    This is especially useful for InfoCasas URLs that embed neighborhood and
    city names in the path.
    """

    path = urlparse(url).path.strip("/")
    match = re.search(r"(?:apartamento|departamento|casa|terreno)-en-([^/]+?)(?:/\d{6,}|$)", path, re.I)
    if not match:
        return

    slug = match.group(1).strip("-")
    city_slug_map = {
        "montevideo": "Montevideo",
        "punta-del-este": "Punta del Este",
        "maldonado": "Maldonado",
        "canelones": "Canelones",
    }
    for city_slug, city_name in city_slug_map.items():
        if slug == city_slug:
            if not row.get("city"):
                row["city"] = city_name
            return
        suffix = "-" + city_slug
        if slug.endswith(suffix):
            neighborhood_slug = slug[: -len(suffix)].strip("-")
            if not row.get("city"):
                row["city"] = city_name
            if neighborhood_slug and not row.get("neighborhood"):
                row["neighborhood"] = title_from_slug(neighborhood_slug)
            return


def extract_location(row: dict[str, Any], text: str, json_blocks: list[Any]) -> None:
    """Fill location fields from structured address data and title patterns.

    Sites are inconsistent, so this combines JSON-LD address fields with common
    Spanish title formats like "en Palermo, Capital Federal".
    """

    for node in walk_json(json_blocks):
        if not isinstance(node, dict):
            continue
        address = node.get("address")
        if isinstance(address, dict):
            row.setdefault("street", clean_text(address.get("streetAddress")))
            row.setdefault("city", clean_text(address.get("addressLocality")))
            row.setdefault("province", clean_text(address.get("addressRegion")))
            row.setdefault("neighborhood", clean_text(address.get("addressSuburb")))
        if "geo" in node and isinstance(node["geo"], dict):
            row.setdefault("latitude", parse_number(node["geo"].get("latitude")))
            row.setdefault("longitude", parse_number(node["geo"].get("longitude")))

    title = row.get("listing_title", "")
    match = re.search(r"\ben\s+([^,|]+),\s*([^,|]+)", title, re.I)
    if match:
        set_location_from_match(row, match)

    if not row.get("city"):
        match = re.search(r"\ben\s+([^,|]+),\s*([^,|]+)", text[:1200], re.I)
        if match:
            set_location_from_match(row, match)


def infer_property_type(url: str, title: str, text: str) -> str:
    """Infer the normalized property type from URL, title, and page text.

    This gives the model a consistent category even when a site uses Spanish
    labels or only exposes the type in the URL.
    """

    sample = f"{url} {title} {text[:800]}"
    for pattern, label in PROPERTY_TYPE_PATTERNS:
        if pattern.search(sample):
            return label
    return ""


def infer_property_subtype(text: str) -> str:
    """Infer common apartment subtypes from listing text.

    Subtype helps separate studios, duplexes, penthouses, and standard units in
    a simple categorical feature.
    """

    lower = text.lower()
    if re.search(r"monoambiente|studio", lower):
        return "Studio"
    if re.search(r"\bd[uú]plex\b|\bduplex\b", lower):
        return "Duplex"
    if re.search(r"penthouse|[áa]tico", lower):
        return "Penthouse"
    if re.search(r"semipiso", lower):
        return "Semi-floor Apartment"
    if re.search(r"piso\s+completo", lower):
        return "Full-floor Apartment"
    return ""


def extract_area(text: str, labels: tuple[str, ...]) -> int | None:
    """Find an area measurement near one of the requested labels.

    Separate calls distinguish covered, total, semi-covered, and uncovered area
    when the page text provides those labels.
    """

    label_pattern = "|".join(labels)
    patterns = (
        rf"(?:{label_pattern})[^\d]{{0,25}}(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mts2|metros)",
        rf"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mts2|metros)[^\n.]{{0,25}}(?:{label_pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return parse_int(match.group(1))
    return None


def extract_first_count(text: str, patterns: tuple[str, ...]) -> int | None:
    """Return the first integer count matched by a group of regex patterns.

    Counts such as bedrooms and bathrooms appear in several Spanish variants.
    """

    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return parse_int(match.group(1))
    return None


def extract_counts_and_areas(row: dict[str, Any], text: str) -> None:
    """Extract numeric property features from visible text.

    These values are high-signal model inputs and are usually present on detail
    pages even when structured JSON is incomplete.
    """

    row["total_rooms"] = extract_first_count(
        text,
        (
            r"(\d+(?:[.,]\d+)?)\s*ambientes?",
            r"(\d+(?:[.,]\d+)?)\s*rooms?",
        ),
    )
    row["bedrooms"] = extract_first_count(
        text,
        (
            r"(\d+(?:[.,]\d+)?)\s*dormitorios?",
            r"(\d+(?:[.,]\d+)?)\s*dorms?\.?",
            r"(\d+(?:[.,]\d+)?)\s*bedrooms?",
        ),
    )
    row["bathrooms"] = extract_first_count(
        text,
        (
            r"(\d+(?:[.,]\d+)?)\s*ba[ñn]os?",
            r"(\d+(?:[.,]\d+)?)\s*bathrooms?",
        ),
    )
    row["parking_spaces"] = extract_first_count(
        text,
        (
            r"(\d+(?:[.,]\d+)?)\s*cocheras?",
            r"(\d+(?:[.,]\d+)?)\s*garajes?",
            r"(\d+(?:[.,]\d+)?)\s*parking",
        ),
    )
    row["toilets"] = extract_first_count(text, (r"(\d+(?:[.,]\d+)?)\s*toilettes?",))
    row["age_years"] = extract_first_count(
        text,
        (
            r"(\d+)\s*a[ñn]os?\s+de\s+antig[uü]edad",
            r"antig[uü]edad[^\d]{0,15}(\d+)",
        ),
    )
    row["floor_number"] = extract_first_count(
        text,
        (
            r"piso[^\d]{0,8}(\d+)",
            r"(\d+)[°º]?\s*piso",
        ),
    )
    row["covered_area_sqm"] = extract_area(
        text,
        ("cubierta", "construidos", "edificados", "covered"),
    )
    row["total_area_sqm"] = extract_area(
        text,
        ("total", "terreno", "superficie", "lote"),
    )
    row["semicovered_area_sqm"] = extract_area(text, ("semicubierta", "semi cubierta"))
    row["uncovered_area_sqm"] = extract_area(text, ("descubierta", "uncovered"))

    if not row.get("covered_area_sqm"):
        first_area = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²|mts2|metros)", text, re.I)
        if first_area:
            row["covered_area_sqm"] = parse_int(first_area.group(1))


def clean_numeric_outliers(row: dict[str, Any]) -> None:
    """Blank numeric values that are probably parsing mistakes.

    Real-estate pages include related listings and marketing text, so simple
    bounds protect the training data from impossible apartment features.
    """

    max_values = {
        "total_rooms": 20,
        "bedrooms": 10,
        "bathrooms": 10,
        "parking_spaces": 10,
        "toilets": 10,
        "covered_area_sqm": 2_000,
        "total_area_sqm": 5_000,
        "semicovered_area_sqm": 1_000,
        "uncovered_area_sqm": 5_000,
        "age_years": 250,
        "floor_number": 100,
        "distance_to_sea_blocks": 500,
        "expenses_usd": 20_000,
    }
    for field, maximum in max_values.items():
        value = parse_number(row.get(field))
        if value is not None and (value < 0 or value > maximum):
            row[field] = ""

    if row.get("property_subtype") == "Studio" or row.get("total_rooms") == 1:
        row["bedrooms"] = 0


def extract_orientation(row: dict[str, Any], text: str) -> None:
    """Extract apartment orientation when the page lists it.

    Orientation is categorical and can affect price in dense apartment markets.
    """

    match = re.search(r"orientaci[oó]n[^\w]{0,10}(norte|sur|este|oeste|ne|no|se|so|n|s|e|o)", text, re.I)
    if match:
        row["orientation"] = match.group(1).upper()


def extract_listing_status(row: dict[str, Any], text: str) -> None:
    """Extract construction and publication status flags.

    New construction, projects, and owner-direct listings can have different
    price behavior from ordinary resale listings.
    """

    lower = text.lower()
    if re.search(r"\ba\s+estrenar\b|brand\s+new", lower):
        row["construction_stage"] = "new"
        row["is_new_construction"] = 1
    elif re.search(r"en\s+pozo|en\s+construcci[oó]n|under\s+construction", lower):
        row["construction_stage"] = "under_construction"
        row["is_under_construction"] = 1
    elif re.search(r"reciclado|refaccionado|renovado", lower):
        row["construction_stage"] = "renovated"
    elif re.search(r"usado|antig[uü]edad", lower):
        row["construction_stage"] = "resale"

    if re.search(r"due[nñ]o\s+directo|owner\s+direct", lower):
        row["publication_type"] = "owner_direct"
        row["is_owner_direct"] = 1
    elif re.search(r"desarrollador|desarrolladora|emprendimiento|proyecto", lower):
        row["publication_type"] = "developer"
        row["is_development_project"] = 1
    elif re.search(r"inmobiliaria|broker|real\s+estate", lower):
        row["publication_type"] = "agency"


def extract_distance_features(row: dict[str, Any], text: str) -> None:
    """Extract simple distance-to-place features from listing text.

    Coastal listings commonly mention distance to the sea, which is useful for
    Uruguay and beach-market price prediction.
    """

    match = re.search(r"distancia\s+al\s+mar[^\d]{0,20}(\d+)", text, re.I)
    if match:
        row["distance_to_sea_blocks"] = parse_int(match.group(1))
        row["is_near_sea"] = 1
        return

    match = re.search(r"(\d+)\s*cuadras?\s+(?:del|al)\s+mar", text, re.I)
    if match:
        row["distance_to_sea_blocks"] = parse_int(match.group(1))
        row["is_near_sea"] = 1


def extract_amenities(row: dict[str, Any], text: str) -> None:
    """Set binary amenity columns from keyword matches.

    Binary feature flags make amenities easy to use in common ML pipelines.
    """

    for column, pattern in AMENITY_PATTERNS.items():
        if re.search(pattern, text, re.I):
            row[column] = 1


def bucket_number(value: Any, buckets: tuple[tuple[float, str], ...], empty_label: str = "") -> str:
    """Convert a numeric feature into a readable bucket label.

    Buckets give simple ML pipelines categorical signals without losing the raw
    numeric values.
    """

    number = parse_number(value)
    if number is None:
        return empty_label
    for upper_limit, label in buckets:
        if number <= upper_limit:
            return label
    return buckets[-1][1]


def safe_ratio(numerator: Any, denominator: Any, digits: int = 2) -> float | str:
    """Return a rounded ratio or blank when inputs are missing.

    Derived ratios like area per room are useful only when both values are
    present and the denominator is non-zero.
    """

    top = parse_number(numerator)
    bottom = parse_number(denominator)
    if top is None or bottom in (None, 0):
        return ""
    return round(top / bottom, digits)


def text_word_count(text: str) -> int:
    """Count word-like tokens in a title or description.

    Text length can be a rough signal for listing quality and information
    richness without requiring NLP dependencies.
    """

    return len(re.findall(r"\b[\wáéíóúñüÁÉÍÓÚÑÜ]+\b", text))


def add_derived_features(row: dict[str, Any]) -> None:
    """Add ML-friendly features derived from already extracted fields.

    These features avoid target leakage: they use property attributes, location,
    and text metadata, not price-derived calculations.
    """

    title = clean_text(row.get("listing_title"))
    description = clean_text(row.get("description"))
    property_type = clean_text(row.get("property_type")).lower()
    effective_area = row.get("covered_area_sqm") or row.get("total_area_sqm")
    bedrooms_for_rooms = parse_number(row.get("bedrooms"))
    rooms = row.get("total_rooms") or (
        bedrooms_for_rooms + 1 if bedrooms_for_rooms is not None else None
    )

    row["is_apartment"] = 1 if property_type == "apartment" else 0
    row["is_studio_apartment"] = 1 if row.get("property_subtype") == "Studio" or row.get("total_rooms") == 1 else 0
    row["is_duplex"] = 1 if row.get("property_subtype") == "Duplex" else 0
    row["is_penthouse"] = 1 if row.get("property_subtype") == "Penthouse" else 0
    row["is_new_construction"] = 1 if row.get("construction_stage") == "new" else 0
    row["is_under_construction"] = 1 if row.get("construction_stage") == "under_construction" else 0
    row["is_development_project"] = 1 if row.get("publication_type") == "developer" else 0
    row["is_owner_direct"] = 1 if row.get("publication_type") == "owner_direct" else 0
    row["is_high_floor"] = 1 if (parse_number(row.get("floor_number")) or 0) >= 9 else 0
    row["has_coordinates"] = 1 if row.get("latitude") and row.get("longitude") else 0
    row["has_exact_address"] = 1 if clean_text(row.get("street")) else 0
    row["has_expenses"] = 1 if row.get("expenses_original") else 0
    row["has_description"] = 1 if description else 0
    row["effective_area_sqm"] = effective_area or ""
    row["area_per_room_sqm"] = safe_ratio(effective_area, rooms)
    row["bedrooms_per_bathroom"] = safe_ratio(row.get("bedrooms"), row.get("bathrooms"))
    row["bathrooms_per_bedroom"] = safe_ratio(row.get("bathrooms"), row.get("bedrooms"))
    row["amenity_count"] = sum(
        1
        for column in AMENITY_COLUMNS
        if column.startswith("has_") and column not in AMENITY_COUNT_EXCLUDE and row.get(column)
    )

    core_fields = ("bedrooms", "bathrooms", "covered_area_sqm", "total_area_sqm", "parking_spaces", "age_years")
    row["core_feature_count"] = sum(1 for field in core_fields if row.get(field) not in ("", None))
    row["title_length_chars"] = len(title)
    row["description_length_chars"] = len(description)
    row["title_word_count"] = text_word_count(title)
    row["description_word_count"] = text_word_count(description)

    row["area_bucket"] = bucket_number(
        effective_area,
        (
            (30, "01_tiny"),
            (50, "02_small"),
            (80, "03_medium"),
            (120, "04_large"),
            (200, "05_extra_large"),
            (10_000, "06_luxury_scale"),
        ),
    )
    row["age_bucket"] = bucket_number(
        row.get("age_years"),
        (
            (0, "01_new"),
            (10, "02_1_to_10"),
            (30, "03_11_to_30"),
            (60, "04_31_to_60"),
            (300, "05_60_plus"),
        ),
    )
    row["floor_bucket"] = bucket_number(
        row.get("floor_number"),
        (
            (0, "01_ground"),
            (3, "02_low"),
            (8, "03_mid"),
            (200, "04_high"),
        ),
    )

    bedrooms = parse_number(row.get("bedrooms"))
    total_rooms = parse_number(row.get("total_rooms"))
    if total_rooms == 1 or row.get("is_studio_apartment"):
        row["room_bucket"] = "01_studio"
    elif bedrooms is None:
        row["room_bucket"] = ""
    elif bedrooms <= 1:
        row["room_bucket"] = "02_one_bedroom"
    elif bedrooms == 2:
        row["room_bucket"] = "03_two_bedroom"
    elif bedrooms == 3:
        row["room_bucket"] = "04_three_bedroom"
    else:
        row["room_bucket"] = "05_four_plus_bedroom"

    location_parts = [
        clean_text(row.get("country")),
        clean_text(row.get("province")),
        clean_text(row.get("city")),
        clean_text(row.get("neighborhood")),
    ]
    row["location_key"] = " | ".join(part for part in location_parts if part)


def find_links(
    soup: BeautifulSoup,
    html: str,
    current_url: str,
    base_url: str,
    config: SiteConfig,
    apartments_only: bool = False,
) -> list[str]:
    """Collect useful same-site links from one page.

    The crawler discovers pagination and detail pages by scanning anchors,
    common card attributes, and raw HTML script/image metadata.
    """

    links: list[str] = []
    candidate_values: list[Any] = []
    useful_attributes = (
        "href",
        "src",
        "data-href",
        "data-url",
        "data-link",
        "data-src",
    )

    for tag in soup.find_all(True):
        for attribute in useful_attributes:
            if tag.has_attr(attribute):
                candidate_values.append(tag.get(attribute))
    candidate_values.extend(raw_html_urls(html))

    for value in candidate_values:
        url = clean_discovered_url(value, current_url)
        if not url or not same_domain(url, base_url):
            continue
        if looks_useful_url(url, config, apartments_only):
            links.append(url)
    return sorted(dict.fromkeys(links))


def extract_property_row(
    url: str,
    config: SiteConfig,
    soup: BeautifulSoup,
    page_text: str,
    json_blocks: list[Any],
    scraped_at: str,
) -> dict[str, Any] | None:
    """Convert one property detail page into one ML-ready CSV row.

    A row is returned only when the page has a usable sale price, because price
    is the target needed for model training.
    """

    title = (
        meta_content(soup, "og:title", "twitter:title")
        or clean_text(soup.title.string if soup.title else "")
        or first_json_value(json_blocks, "name", "headline")
    )
    description = (
        meta_content(soup, "og:description", "description", "twitter:description")
        or first_json_value(json_blocks, "description")
    )

    row: dict[str, Any] = {
        "source_url": url,
        "source_domain": urlparse(url).netloc.lower(),
        "listing_id": listing_id_from_url(url),
        "listing_title": title,
        "description": clean_text(description)[:1200],
        "scraped_at": scraped_at,
        "data_source": config.name,
        "country": config.country,
        "operation_type": "Sale",
    }

    price_data = extract_price(page_text, json_blocks, config.country)
    if not price_data:
        return None
    row.update(price_data)

    row["property_type"] = infer_property_type(url, title, page_text)
    row["property_subtype"] = infer_property_subtype(f"{title} {description}")
    extract_expenses(row, page_text, config.country)
    extract_location(row, page_text, json_blocks)
    extract_location_from_url(row, url)
    extract_counts_and_areas(row, page_text)
    extract_orientation(row, page_text)
    listing_feature_text = f"{title} {description}"
    extract_listing_status(row, listing_feature_text)
    extract_distance_features(row, page_text)
    extract_amenities(row, page_text)
    clean_numeric_outliers(row)
    add_derived_features(row)

    # Very small converted local-currency prices are usually monthly rent or noise.
    if row.get("price_usd") and int(row["price_usd"]) < 20_000:
        return None
    return clean_row(row)


def row_matches_property_mode(row: dict[str, Any], apartments_only: bool) -> bool:
    """Check whether an extracted row matches the requested property mode.

    URL filtering reduces wasted requests, but row filtering is the final guard
    that keeps apartment-only datasets clean.
    """

    if not apartments_only:
        return True
    property_type = str(row.get("property_type", "")).lower()
    return property_type == "apartment" or apartment_word_in_url(str(row.get("source_url", "")))


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply final CSV defaults and column ordering to an extracted row.

    Missing numeric/categorical values become blank cells, while amenity flags
    become 0 or 1.
    """

    clean: dict[str, Any] = {}
    for column in CSV_COLUMNS:
        value = row.get(column, "")
        if column in AMENITY_COLUMNS:
            clean[column] = 1 if value not in ("", None, 0, "0") else 0
        elif column in NUMERIC_COLUMNS:
            clean[column] = "" if value in ("", None) else value
        else:
            clean[column] = clean_text(value)
    return clean


def raw_page_record(url: str, status_code: int | None, page_text: str, row: dict[str, Any] | None) -> dict[str, Any]:
    """Build a JSONL debug record for one fetched page.

    Raw page text helps inspect extraction misses without re-crawling the site.
    """

    return {
        "url": url,
        "status_code": status_code,
        "text_sha256": hashlib.sha256(page_text.encode("utf-8", errors="ignore")).hexdigest(),
        "text": page_text[:50_000],
        "extracted_row": row,
    }


def listing_id_from_url(url: str) -> str:
    """Extract a stable listing ID from common real-estate URL formats.

    Some sites expose the same property through more than one URL, so using the
    listing ID prevents duplicate rows in the model-training CSV.
    """

    patterns = (
        r"--(\d{6,})(?:$|[/?#])",
        r"-inmuebles-(\d{6,})(?:$|[/?#])",
        r"/(\d{6,})(?:$|[/?#])",
        r"-(\d{6,})\.html(?:$|[/?#])",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""


def row_dedupe_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build a duplicate-detection key for one extracted row.

    A site listing ID is preferred; otherwise the key falls back to the most
    stable property facts available in the row.
    """

    listing_id = listing_id_from_url(str(row.get("source_url", "")))
    if listing_id:
        return (row.get("source_domain"), listing_id)
    return (
        row.get("source_url"),
        row.get("price_usd"),
        row.get("covered_area_sqm"),
        row.get("bedrooms"),
        row.get("bathrooms"),
    )


def crawl_site(
    base_url: str,
    config: SiteConfig,
    args: argparse.Namespace,
    session: requests.Session,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Crawl one site and return extracted property rows plus raw page records.

    The crawl starts from configured sale pages, follows useful links, respects
    robots.txt by default, and stops at the requested limits.
    """

    seeds = build_seed_urls(base_url, config, args.apartments_only)
    queue = deque(seeds)
    seen_urls = set(seeds)
    rows: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    seen_row_keys: set[tuple[Any, ...]] = set()
    fetched_pages = 0
    blocked_pages = 0
    robot_parser = None
    if not args.ignore_robots:
        robot_parser = load_robot_parser(session, base_url, args.timeout)

    print(f"\n{config.name}: {base_url}")
    print(f"  seeds: {len(seeds)}")

    while queue and fetched_pages < args.max_pages_per_site:
        if len(rows) >= args.max_properties_per_site:
            break

        url = queue.popleft()
        if robot_parser and not robot_parser.can_fetch(USER_AGENT, url):
            print(f"  skipped by robots.txt: {url}")
            continue

        html, status_code = fetch_html(session, url, args.timeout)
        fetched_pages += 1
        if not html:
            continue

        scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        soup = BeautifulSoup(html, "html.parser")
        json_blocks = extract_json_blocks(soup)
        page_text = soup_text(soup)
        row = None

        if is_blocked_page(page_text):
            blocked_pages += 1
            print(f"  blocked page: {url} (HTTP {status_code})")
            raw_pages.append(raw_page_record(url, status_code, page_text, row))
            polite_sleep(args.delay)
            continue

        if status_code and status_code >= 400:
            print(f"  skipped HTTP {status_code}: {url}")
            raw_pages.append(raw_page_record(url, status_code, page_text, row))
            polite_sleep(args.delay)
            continue

        if looks_like_detail_url(url, config):
            row = extract_property_row(url, config, soup, page_text, json_blocks, scraped_at)
            if row:
                if not row_matches_property_mode(row, args.apartments_only):
                    print(f"  non-apartment skipped: {url}")
                    raw_pages.append(raw_page_record(url, status_code, page_text, row))
                    polite_sleep(args.delay)
                    continue
                dedupe_key = row_dedupe_key(row)
                if dedupe_key in seen_row_keys:
                    print(f"  duplicate property skipped: {url}")
                else:
                    seen_row_keys.add(dedupe_key)
                    rows.append(row)
                    print(f"  property {len(rows):>4}: {row.get('price_usd')} USD | {url}")

        raw_pages.append(raw_page_record(url, status_code, page_text, row))

        links = find_links(soup, html, url, base_url, config, args.apartments_only)
        detail_links = [link for link in links if looks_like_detail_url(link, config)]
        search_links = [link for link in links if not looks_like_detail_url(link, config)]

        new_links = 0
        for link in reversed(detail_links):
            if link in seen_urls:
                continue
            seen_urls.add(link)
            queue.appendleft(link)
            new_links += 1
        for link in search_links:
            if link in seen_urls:
                continue
            seen_urls.add(link)
            queue.append(link)
            new_links += 1
        if new_links:
            print(f"  discovered links: +{new_links} from {url}")

        polite_sleep(args.delay)

    print(f"  fetched pages: {fetched_pages}")
    if blocked_pages:
        print(f"  blocked pages: {blocked_pages}")
    print(f"  properties: {len(rows)}")
    return rows, raw_pages


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate property rows using URL and key listing facts.

    Dedupe keeps repeated pagination or tracking links from inflating the
    training data.
    """

    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = row_dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the final model-training CSV.

    UTF-8 with BOM makes the file easier to open in Excel on Windows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write raw page records as newline-delimited JSON.

    JSONL is append/debug friendly and handles variable raw text better than CSV.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the full scrape from sitelist to CSV and JSONL outputs.

    This coordinates configuration, crawling each supported site, deduping, and
    writing the files used by later ML work.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    site_urls = read_site_urls(args.sitelist)
    session = make_session()

    planned_sites = [(url, config_for_url(url)) for url in site_urls]
    planned_sites = [(url, config) for url, config in planned_sites if config is not None]
    planned_sites = [
        (url, config) for url, config in planned_sites if wanted_site(url, config, args.sites)
    ]

    if args.dry_run:
        for base_url, config in planned_sites:
            print(f"{config.name}: {base_url}")
            for seed in build_seed_urls(base_url, config, args.apartments_only):
                print(f"  {seed}")
        return

    all_rows: list[dict[str, Any]] = []
    all_raw_pages: list[dict[str, Any]] = []

    for base_url, config in planned_sites:
        rows, raw_pages = crawl_site(base_url, config, args, session)
        all_rows.extend(rows)
        all_raw_pages.extend(raw_pages)

    all_rows = dedupe_rows(all_rows)
    write_csv(args.output, all_rows)
    write_jsonl(args.raw_output, all_raw_pages)

    print("\nDone")
    print(f"CSV rows: {len(all_rows)}")
    print(f"CSV file: {args.output}")
    print(f"Raw pages: {len(all_raw_pages)}")
    print(f"Raw JSONL file: {args.raw_output}")


if __name__ == "__main__":
    main()
