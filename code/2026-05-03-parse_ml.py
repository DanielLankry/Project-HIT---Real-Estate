"""
ML-ready CSV v3 — multi-country.
Sources: Argenprop (AR), InfoCasas (UY), Gallito (UY) [optional].
Adds 'country' column. Drops 'description'.
"""
import json, re, csv, glob, pathlib, unicodedata

RAW_DIR = pathlib.Path("/sessions/practical-adoring-pascal/mnt/outputs/raw")
OUT_CSV = pathlib.Path("/sessions/practical-adoring-pascal/mnt/outputs/real_estate_ml_ready.csv")
ARS_PER_USD = 1500
UYU_PER_USD = 40
MIN_SALE_USD = 30_000

def norm_key(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def parse_int_thousands(s):
    if s is None or s == "": return None
    s = str(s).replace(" ", "")
    m = re.search(r"-?\d[\d.]*", s)
    if not m: return None
    return int(m.group(0).replace(".", ""))

def parse_decimal(s):
    if s is None or s == "": return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None

def clean_text(s):
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def floor_to_num(s):
    if s is None or str(s).strip() == "": return None
    s = str(s).strip().upper()
    if s in ("PB","PLANTA BAJA","P.B.","BAJO"): return 0
    return parse_int_thousands(s)


PROPERTY_TYPE_MAP = {
    "departamento":"Apartment", "apartamento":"Apartment", "casa":"House",
    "ph":"PH", "terreno":"Land", "local":"Commercial Space", "local comercial":"Commercial Space",
    "oficina":"Office", "quinta":"Country House", "galp\u00f3n":"Warehouse", "galpon":"Warehouse",
    "campo":"Farm", "chacra":"Farm", "chacra o campo":"Farm", "hotel":"Hotel",
    "cochera":"Parking Space", "garaje":"Parking Space",
}
CFT_MAP = {"CFT1":"total_rooms","CFT2":"bedrooms","CFT3":"bathrooms","CFT4":"toilets",
           "CFT5":"age_years","CFT6":"_expensas","CFT7":"parking_spaces",
           "CFT100":"total_area_sqm","CFT101":"covered_area_sqm"}

AMENITY_TRANSLATE = {
    "aire_acondicionado":"has_air_conditioning","alarma":"has_alarm","amoblado":"is_furnished",
    "apto_credito":"mortgage_eligible","apto_profesional":"professional_use_allowed","ascensor":"has_elevator",
    "balcon":"has_balcony","baulera":"has_storage_room","caldera":"has_boiler","calefaccion":"has_heating",
    "cancha_de_deportes":"has_sports_court","cocina_equipada":"has_equipped_kitchen","comedor":"has_dining_room",
    "comedor_diario":"has_breakfast_room","dependencia_de_servicio":"has_maid_room","dormitorio_en_suite":"has_master_suite",
    "encargado":"has_doorman","escritorio":"has_office","estudio":"has_studio_room","gimnasio":"has_gym","hall":"has_hall",
    "hidromasaje":"has_jacuzzi","iluminacion":"has_lighting","internet_wifi":"has_internet","internet":"has_internet",
    "jardin":"has_garden","lavadero":"has_laundry_room","living":"has_living_room","living_comedor":"has_living_dining_room",
    "luminoso":"is_bright","microondas":"has_microwave","parrilla":"has_grill","patio":"has_patio",
    "permite_mascotas":"pets_allowed","pileta":"has_pool","quincho":"has_bbq_house","recepcion":"has_reception",
    "ropa_de_cama":"has_bedding","sala_de_juegos":"has_game_room","solarium":"has_solarium","sum":"has_multipurpose_room",
    "termotanque":"has_water_heater","terraza":"has_terrace","toallas":"has_towels","toilette":"has_toilette_room",
    "uso_comercial":"commercial_use_allowed","vestidor":"has_walk_in_closet","vigilancia":"has_security",
    "agua_corriente":"has_running_water","electricidad":"has_electricity","gas_natural":"has_natural_gas",
    "abl":"has_property_tax_service","acceso_para_personas_con_movilidad_reducida":"is_accessible",
    "cochera":"has_garage_room","dormitorio":"has_extra_bedroom","seguridad":"has_security","suite":"has_master_suite",
    "palier":"has_private_landing","dependencia":"has_maid_room","galeria":"has_gallery",
    "jardin_fondo":"has_back_garden","jardin_frente":"has_front_garden","sotano":"has_basement","altillo":"has_attic",
    "biblioteca":"has_library","vestuario":"has_dressing_room","cocina_comedor":"has_kitchen_dining",
    "video_cable":"has_cable_tv","telefono":"has_phone_line","refrigeracion":"has_cooling","limpieza":"has_cleaning_service",
    "rentas":"has_rentas_paid","acepta_permuta":"accepts_swap","propiedad_ocupada":"is_currently_occupied",
    "circulacion":"has_double_circulation","frente":"is_front_facing","disposicion_frente":"is_front_facing",
    # InfoCasas / Uruguay extras
    "barbacoa":"has_bbq_house","piscina":"has_pool","piscina_climatizada":"has_heated_pool","sauna":"has_sauna",
    "gym":"has_gym","cocina_definida":"has_kitchen","placard":"has_closet","placards":"has_closet",
    "estufa_a_lena":"has_wood_stove","estufa":"has_heating","portero_electrico":"has_intercom",
    "portero":"has_intercom","conserje":"has_doorman","portico":"has_porch","cancha_de_tenis":"has_tennis_court",
    "cancha_de_futbol":"has_football_court","losa_radiante":"has_radiant_floor","aire":"has_air_conditioning",
    "playroom":"has_play_room","salon_de_usos_multiples":"has_multipurpose_room","salon_comunal":"has_common_room",
    "ascensores":"has_elevator","ascensor_y_escalera":"has_elevator","cable":"has_cable_tv",
    "calefon":"has_water_heater","fondo":"has_backyard","fondo_de_casa":"has_backyard",
}
DROP_AMENITIES = {"bano","cocina"}

def translate_amenity(spanish_key):
    if spanish_key in DROP_AMENITIES: return None
    if not spanish_key or len(spanish_key) > 40: return None
    if any(c.isdigit() for c in spanish_key): return None
    if any(bad in spanish_key for bad in ("gmail","hola_vi","contacten","quiero","gracias","propiedad_en")): return None
    return AMENITY_TRANSLATE.get(spanish_key, "has_" + spanish_key)

# --------------- Argenprop ---------------
ID_URL_RE = re.compile(r"--(\d+)$")
TYPE_URL_RE = re.compile(r"argenprop\.com/([a-z]+)-en-venta", re.I)
def grab_section(md, name):
    m = re.search(rf"###\s*{name}(.+?)(?:###|\Z)", md, re.DOTALL | re.I)
    return m.group(1) if m else ""
def bullets(text):
    return [b.strip() for b in re.findall(r"\*\s+([^\n*]+?)\s*(?:\n|$)", text) if b.strip()]

def parse_argenprop_md(url, title, md):
    if "alquiler" in url.lower() and "/venta" not in url.lower(): return None
    pt = TYPE_URL_RE.search(url)
    row = {"data_source":"argenprop","country":"AR","property_type": pt.group(1).capitalize() if pt else "","property_subtype": ""}
    bc = re.findall(r"\d+\.\s*\[([^\]]+)\]\(https://www\.argenprop\.com/", md)
    bc = [b.strip() for b in bc if b.strip().lower() != "argenprop" and "#" not in b and "\n" not in b]
    row["province"] = bc[2] if len(bc) > 2 else ""
    row["city"] = bc[3] if len(bc) > 3 else ""
    row["neighborhood"] = bc[4] if len(bc) > 4 else row["city"]
    op_m = re.search(r"##\s*Venta\s+en\s+([^,\n]+),\s*([^\n]+)", md)
    if op_m and not row["neighborhood"]: row["neighborhood"] = op_m.group(1).strip()
    if op_m and not row["city"]: row["city"] = op_m.group(2).strip()
    pm = re.search(r"(USD|U\$S|\$)\s*([\d.]+)", md)
    if not pm: return None
    cur = pm.group(1).upper()
    amt = parse_int_thousands(pm.group(2))
    if amt is None: return None
    price_usd = float(amt) if cur.startswith(("U","USD")) else float(amt)/ARS_PER_USD
    if price_usd < MIN_SALE_USD: return None
    row["price_usd"] = round(price_usd)
    fm = re.search(r"Piso\s+([0-9]+|PB|Pb|pb)", md)
    if fm: row["floor_number"] = floor_to_num(fm.group(1))
    car = grab_section(md, "Características")
    pairs = re.findall(r"\*\s+([^:*\n]+):\s*\*\*([^*]+)\*\*", car)
    for k,v in pairs:
        k, v = k.strip(), v.strip()
        if "Ambientes" in k:        row["total_rooms"] = parse_int_thousands(v)
        elif "Dormitorios" in k:    row["bedrooms"] = parse_int_thousands(v)
        elif "Baños" in k or "Banos" in k: row["bathrooms"] = parse_int_thousands(v)
        elif "Cocheras" in k:       row["parking_spaces"] = parse_int_thousands(v)
        elif "Antig" in k:          row["age_years"] = parse_int_thousands(v)
        elif "Tipo de Balc" in k:   row["balcony_type"] = v
        elif "Tipo de Vista" in k:  row["view_type"] = v
        elif "Tipo de Piso" in k:   row["flooring_type"] = v
        elif "Plantas" in k:        row["number_of_floors_in_unit"] = parse_int_thousands(v)
        elif "Orientaci" in k:      row["orientation"] = v
    flags = re.findall(r"\*\s+\*\*([^*]+)\*\*", car)
    for f in flags:
        en = translate_amenity(norm_key(f.strip()))
        if en: row[en] = 1
    pre = re.search(r"\\?\+?\s*\$[\d. ]+\s*expensas(.+?)###", md, re.DOTALL)
    if pre:
        for b in bullets(pre.group(1)):
            bl = b.lower()
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\W?\s*(cubierta|total|semi|descub)?", bl)
            if m:
                v = parse_int_thousands(m.group(1)); kind = (m.group(2) or "").lower()
                if "cubie" in kind: row.setdefault("covered_area_sqm", v)
                elif "total" in kind: row.setdefault("total_area_sqm", v)
                elif "semi" in kind: row.setdefault("semicovered_area_sqm", v)
                elif "descub" in kind: row.setdefault("uncovered_area_sqm", v)
                else: row.setdefault("covered_area_sqm", v)
            elif "monoambiente" in bl: row.setdefault("total_rooms", 1)
            elif b.strip() in ("Norte","Sur","Este","Oeste","NE","NO","SE","SO","N","S","E","O"):
                row.setdefault("orientation", b.strip())
    for s in bullets(grab_section(md, "Servicios")):
        en = translate_amenity(norm_key(s)); 
        if en: row[en] = 1
    for s in bullets(grab_section(md, "Ambientes")):
        en = translate_amenity(norm_key(s));
        if en: row[en] = 1
    return row

def parse_argenprop_file(p):
    with open(p) as f: d = json.load(f)
    out = []
    for it in d["items"]:
        url = it.get("url") or ""
        if not ID_URL_RE.search(url): continue
        md = it.get("markdown") or ""
        if len(md) < 500: continue
        r = parse_argenprop_md(url, it.get("metadata.title") or "", md)
        if r: out.append(r)
    return out

# --------------- InfoCasas ---------------
INFOCASAS_ID = re.compile(r"/(\d{6,})$")
TYPE_PAT = re.compile(r"\b(Casa|Apartamento|Local|Terreno|Oficina|PH|Quinta|Galp[oó]n|Campo|Hotel|Cochera|Garaje)\s+en\s+([^,\n]+),\s*([^\n*]+)", re.I)

def parse_infocasas_md(url, title, md):
    if not INFOCASAS_ID.search(url): return None
    if re.search(r"/venta(?:/|\?|$)", url): return None  # SERP page
    if "/inmobiliarias/" in url or "/proyectos/" in url: return None
    row = {"data_source":"infocasas","country":"UY","property_type":"","property_subtype":""}
    # Type + neighborhood + city via "**Apartamento en Centro, Montevideo**"
    tp = TYPE_PAT.search(md)
    if tp:
        row["property_type"] = tp.group(1).capitalize()
        row["neighborhood"] = tp.group(2).strip()
        row["city"] = tp.group(3).strip()
    # Province often = department (Montevideo, Canelones, Maldonado, etc.) — same as city in UY
    row["province"] = row.get("city","")
    # Price: "U$S 255.000"
    pm = re.search(r"U\$S\s*([\d.]+)", md) or re.search(r"USD\s*([\d.]+)", md)
    if not pm:
        pm_uyu = re.search(r"\$\s*([\d.]{6,})", md)
        if not pm_uyu: return None
        amt = parse_int_thousands(pm_uyu.group(1))
        if amt is None: return None
        price_usd = float(amt) / UYU_PER_USD
    else:
        amt = parse_int_thousands(pm.group(1))
        if amt is None: return None
        price_usd = float(amt)
    if price_usd < MIN_SALE_USD: return None
    row["price_usd"] = round(price_usd)
    # InfoCasas detail "•\n\n<Field>\n\n**<Value>**" pattern
    detail_pairs = re.findall(r"\u2022\s*\n+\s*([^\n]+?)\n+\s*\*\*([^*]+)\*\*", md)
    for k, v in detail_pairs:
        k = k.strip(); v = v.strip()
        kl = k.lower()
        if v in ("\u00a1Preg\u00fantale!", "No aplica", "No"): continue
        if "tipo de propiedad" in kl: row["property_type"] = v
        elif kl == "estado": row["estado_de_conservacion"] = v
        elif "ba\u00f1os" == kl or "banos" == kl: row.setdefault("bathrooms", parse_int_thousands(v))
        elif "antig" in kl: 
            n = parse_int_thousands(v)
            if n is not None: row["age_years"] = n
        elif "dormitorios" == kl:
            if "Mono" in v: row["total_rooms"] = 1
            else: row["bedrooms"] = parse_int_thousands(v)
        elif "garajes" == kl: row["parking_spaces"] = parse_int_thousands(v)
        elif "orientaci\u00f3n" in kl: row["orientation"] = v
        elif "m\u00b2 del terreno" in kl: row.setdefault("total_area_sqm", parse_int_thousands(v))
        elif "barrio privado" in kl: row["is_gated_community"] = 1 if v.lower() in ("si","s\u00ed","yes") else 0
        elif "zona" == kl and not row.get("neighborhood"): row["neighborhood"] = v
        elif "disposici\u00f3n" == kl: row["disposicion"] = v
        elif "acepta permuta" in kl and v.lower() in ("si","s\u00ed"): row["accepts_swap"] = 1
        elif "vivienda social" in kl and v.lower() in ("si","s\u00ed"): row["is_social_housing"] = 1
        elif "apto para oficina" in kl and v.lower() in ("si","s\u00ed"): row["professional_use_allowed"] = 1
        elif "vista al mar" in kl and v.lower() in ("si","s\u00ed"): row["has_sea_view"] = 1
        elif "distancia al mar" in kl and re.search(r"\d", v): row["distance_to_sea_blocks"] = parse_int_thousands(v)
        elif "financiaci\u00f3n" in kl and v.lower() in ("si","s\u00ed"): row["mortgage_eligible"] = 1
    # Bedrooms/baths/area: "**3 Dorms.** **2 Baños****91 m²**" or "**Mono** ..."
    if re.search(r"\bMono\b", md): row.setdefault("total_rooms", 1)
    bm = re.search(r"(\d+)\s*Dorms?\.", md)
    if bm: row["bedrooms"] = parse_int_thousands(bm.group(1))
    bam = re.search(r"(\d+)\s*Ba[ñn]os?", md)
    if bam: row["bathrooms"] = parse_int_thousands(bam.group(1))
    am = re.search(r"(\d+)\s*m\W", md)
    if am: row["covered_area_sqm"] = parse_int_thousands(am.group(1))
    # Floor: "5to piso" / "piso 5" / "PB"
    fm = re.search(r"(?:piso\s+(\d+|PB|Pb))|(\d+)to\s+piso|(\d+)°\s*piso", md, re.I)
    if fm:
        v = next((g for g in fm.groups() if g), None)
        if v: row["floor_number"] = floor_to_num(v)
    # Garage mention
    if re.search(r"\b[Gg]ara(je|ge)\b|\b[Cc]ochera\b", md):
        row["has_garage_room"] = 1
        # try to extract count
        gm = re.search(r"(\d+)\s*[Gg]araje", md)
        if gm: row["parking_spaces"] = parse_int_thousands(gm.group(1))
    # Amenity flag detection by keyword scan
    KEYWORDS = {
        "balc[oó]n":"has_balcony","terraza":"has_terrace","jardin|jard[ií]n":"has_garden",
        "piscina":"has_pool","gimnasio|gym":"has_gym","parrillero|parrilla|barbacoa":"has_grill",
        "ascensor":"has_elevator","seguridad|vigilancia|24\\s*hs":"has_security",
        "amoblado|amueblado":"is_furnished","aire\\s*acondicionado":"has_air_conditioning",
        "calefacci[oó]n":"has_heating","calef[oó]n":"has_water_heater",
        "lavadero":"has_laundry_room","living":"has_living_room","comedor":"has_dining_room",
        "vestidor":"has_walk_in_closet","sauna":"has_sauna","cancha\\s+de\\s+tenis":"has_tennis_court",
        "cancha\\s+de\\s+f[uú]tbol":"has_football_court","barbacoa":"has_bbq_house",
        "playroom":"has_play_room","sum\\b":"has_multipurpose_room","conserjeria|conserje|porter[ií]a":"has_doorman",
        "estufa\\s+a\\s+le[nñ]a|estufa\\s+a\\s+lena":"has_wood_stove","losa\\s+radiante":"has_radiant_floor",
        "video\\s+cable|cable\\s+tv":"has_cable_tv","internet|wifi":"has_internet",
        "agua\\s+caliente":"has_hot_water","gas\\s+natural":"has_natural_gas",
        "apto\\s+cr[ée]dito|acepta\\s+banco":"mortgage_eligible","placard":"has_closet",
        "lumino[s]?o":"is_bright","frente":"is_front_facing","contrafrente":"is_back_facing",
        "altillo":"has_attic","baulera":"has_storage_room","mascotas":"pets_allowed",
        "negociable":"is_negotiable","oportunidad":"is_opportunity",
    }
    md_low = md.lower()
    for pat, col in KEYWORDS.items():
        if re.search(pat, md_low): row[col] = 1
    return row

def parse_infocasas_file(p):
    with open(p) as f: d = json.load(f)
    out = []
    for it in d["items"]:
        url = it.get("url") or ""
        md = it.get("markdown") or ""
        if len(md) < 300: continue
        r = parse_infocasas_md(url, it.get("metadata.title") or "", md)
        if r: out.append(r)
    return out

# --------------- Gallito (placeholder, depends on what we get) ---------------
def parse_gallito_file(p):
    with open(p) as f: d = json.load(f)
    out = []
    for it in d["items"]:
        url = it.get("url") or ""
        md = it.get("markdown") or ""
        if len(md) < 300: continue
        if "/inmuebles" not in url: continue
        # Gallito uses similar UYU/USD mix and Spanish keywords; reuse InfoCasas-style parser
        row = parse_infocasas_md(url, it.get("metadata.title") or "", md)
        if row:
            row["data_source"] = "gallito"
            row["country"] = "UY"
            out.append(row)
    return out

# --------------- Build ---------------
def main():
    rows = []
    for p in sorted(RAW_DIR.glob("argenprop_run*.json")):
        r = parse_argenprop_file(p); print(f"Argenprop {p.name}: {len(r)}"); rows.extend(r)
    for p in sorted(RAW_DIR.glob("infocasas_run*.json")):
        r = parse_infocasas_file(p); print(f"InfoCasas {p.name}: {len(r)}"); rows.extend(r)
    for p in sorted(RAW_DIR.glob("gallito_run*.json")):
        r = parse_gallito_file(p); print(f"Gallito {p.name}: {len(r)}"); rows.extend(r)

    # Dedupe by (data_source, neighborhood, price_usd, covered_area_sqm)
    seen, uniq = set(), []
    for r in rows:
        k = (r["data_source"], r.get("neighborhood"), r.get("price_usd"), r.get("covered_area_sqm"))
        if k in seen: continue
        seen.add(k); uniq.append(r)
    print(f"Unique: {len(uniq)}")

    all_cols = set()
    for r in uniq: all_cols.update(r.keys())
    amenity_cols = sorted([c for c in all_cols if c.startswith(("has_","is_","pets_","mortgage_","commercial_","professional_","accepts_"))])
    counts = {c: sum(1 for r in uniq if r.get(c) not in (None,"",0,"0")) for c in amenity_cols}
    amenity_cols = [c for c in amenity_cols if counts[c] >= 2]

    NUMERIC = ["price_usd","total_rooms","bedrooms","bathrooms","parking_spaces","toilets",
               "covered_area_sqm","total_area_sqm","semicovered_area_sqm","uncovered_area_sqm",
               "lot_frontage_sqm","lot_length_sqm","age_years","number_of_floors_in_unit",
               "floors_in_building","units_per_floor","floor_number","latitude","longitude"]
    CATEGORICAL = ["data_source","country","property_type","property_subtype","province","city","neighborhood",
                   "balcony_type","view_type","flooring_type","orientation"]
    columns = CATEGORICAL + NUMERIC + amenity_cols

    for r in uniq:
        pt = (r.get("property_type") or "").strip().lower()
        if pt in PROPERTY_TYPE_MAP: r["property_type"] = PROPERTY_TYPE_MAP[pt]
        for c in NUMERIC:
            v = r.get(c)
            if v is None or v == "": r[c] = ""
            elif isinstance(v,(int,float)): r[c] = v
            else: r[c] = parse_int_thousands(v) if c not in ("latitude","longitude") else parse_decimal(v)
            if r[c] is None: r[c] = ""
        for c in CATEGORICAL:
            r[c] = "" if r.get(c) is None else clean_text(r.get(c))
        for c in amenity_cols:
            v = r.get(c)
            r[c] = 1 if v not in (None, "", 0, "0") else 0

    with open(OUT_CSV,"w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader(); w.writerows(uniq)
    print(f"Wrote {OUT_CSV} shape=({len(uniq)}, {len(columns)})")

if __name__ == "__main__":
    main()
