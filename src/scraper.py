#!/usr/bin/env python3
"""
Complete sand mining data aggregator — 15 sources.
Sources:
  1.  OpenStreetMap Overpass API  — sand quarries + all mining tags
  2.  PANGAEA/GEE global mining   — 44,000+ polygons, India filtered
  3.  Wikidata SPARQL             — structured mine locations
  4.  USGS MRDS                   — US Geological Survey mineral DB
  5.  India Sand Watch (Veditum)  — 2000+ India-specific reports
  6.  SANDRP                      — South Asia rivers network
  7.  India Water Portal           — aggregated water/mining news
  8.  Down to Earth               — environment magazine
  9.  Google News RSS             — all Indian outlets, 10 queries
  10. The Hindu                   — environment section
  11. NDTV                        — environment section
  12. Times of India              — mining topic page
  13. NGT India                   — National Green Tribunal orders
  14. data.gov.in                 — Government of India open data
  15. PRS India                   — parliamentary questions

All results geocoded → deduplicated → single CSV.
Integrates with existing pipeline via run_scraper().
"""

import os, re, time, json, math, hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from urllib.parse import quote_plus, urljoin

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False
    print("Warning: pip install beautifulsoup4")

try:
    import ee
    EE_OK = True
except ImportError:
    EE_OK = False

from src import config

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRAPED_FILE = os.path.join(config.OUTPUT_DIR, 'scraped_mining_locations.csv')
CACHE_DIR    = os.path.join(config.OUTPUT_DIR, 'scraper_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ── India bounding box  S, W, N, E ───────────────────────────────────────────
INDIA_BBOX = (6.5, 68.0, 37.5, 98.0)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

INDIAN_RIVERS = [
    'Ganga','Yamuna','Brahmaputra','Godavari','Krishna','Kaveri','Narmada',
    'Mahanadi','Damodar','Hooghly','Bhagirathi','Chambal','Betwa','Ken',
    'Son','Gandak','Kosi','Ghaghra','Sabarmati','Mahi','Tapi','Tungabhadra',
    'Cauvery','Periyar','Vaigai','Banas','Luni','Gomti','Rapti','Sone',
    'Subarnarekha','Kangsabati','Ajay','Mayurakshi','Rupnarayan','Silabati',
    'Kasai','Kopai','Dwarka','Falgu','Punpun','Gaula','Ramganga','Sharda',
    'Kali','Tons','Alaknanda','Mandakini','Pindar','Teesta','Torsa',
]

INDIAN_STATES = [
    'Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chhattisgarh',
    'Goa','Gujarat','Haryana','Himachal Pradesh','Jharkhand','Karnataka',
    'Kerala','Madhya Pradesh','Maharashtra','Manipur','Meghalaya','Mizoram',
    'Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu',
    'Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
    'Delhi','Jammu','Kashmir','UP','MP','WB','TN','AP','J&K',
]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(url):
    key = hashlib.md5(url.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f'{key}.json')

def _cached_get(url, ttl_hours=72):
    path = _cache_path(url)
    if os.path.exists(path):
        if (time.time() - os.path.getmtime(path)) / 3600 < ttl_hours:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            data = {'text': r.text, 'url': url}
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            return data
    except Exception as ex:
        pass
    return None

def _in_india(lat, lon):
    s, w, n, e = INDIA_BBOX
    return s < lat < n and w < lon < e

def _rec(lat, lon, label=1, source='', place='', river='',
         state='', url='', confidence=0.8):
    return dict(lat=round(float(lat), 6), lon=round(float(lon), 6),
                label=int(label), source=source, place=str(place),
                river=str(river), state=str(state),
                url=str(url), confidence=float(confidence))

def geocode(place, river=None, state=None):
    queries = []
    if river and state:
        queries.append(f"{place} near {river} river {state} India")
    if state:
        queries.append(f"{place} {state} India")
    if river:
        queries.append(f"{place} near {river} India")
    queries.append(f"{place} India")
    for q in queries:
        try:
            url  = (f"https://nominatim.openstreetmap.org/search"
                    f"?q={quote_plus(q)}&format=json&limit=1&countrycodes=in")
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    lat, lon = float(data[0]['lat']), float(data[0]['lon'])
                    if _in_india(lat, lon):
                        return lat, lon
            time.sleep(1.2)
        except Exception:
            time.sleep(2)
    return None, None

def extract_locs(text):
    river = next((r for r in INDIAN_RIVERS
                  if re.search(rf'\b{r}\b', text, re.I)), None)
    state = next((s for s in INDIAN_STATES
                  if re.search(rf'\b{s}\b', text, re.I)), None)
    patterns = [
        r'(?:sand mining|sand mine|illegal mining|riverbed mining|river sand)'
        r'.{0,150}?([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)'
        r'\s+(?:district|ghat|village|block|tehsil|taluk|mandal)',
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)'
        r'\s+(?:district|ghat|village|block|tehsil|taluk)'
        r'.{0,100}?(?:sand mining|sand mine|illegal mining)',
        r'mining\s+(?:in|at|near|along|from)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+(?:river|nadi)'
        r'.{0,50}?(?:mining|excavat|dredg)',
    ]
    seen, locs = set(), []
    for pat in patterns:
        for m in re.findall(pat, text, re.I):
            place = m.strip().title()
            if len(place) > 2 and place.lower() not in seen \
                    and place not in INDIAN_STATES:
                seen.add(place.lower())
                locs.append({'place': place, 'river': river,
                             'state': state, 'lat': None, 'lon': None,
                             'label': 1, 'confidence': 0.7})
    return locs

def _text_rec(loc, source, url=''):
    return {**loc, 'source': source, 'url': url}


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — OpenStreetMap (ALL mining/quarry/dredging tags)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_osm():
    print("  [1/15] OpenStreetMap Overpass API...")
    results = []
    s, w, n, e = INDIA_BBOX
    bbox = f"{s},{w},{n},{e}"
    overpass = 'https://overpass-api.de/api/interpreter'

    osm_queries = [
        # Sand quarries — all tag variants
        f'[out:json][timeout:60];(node["man_made"="quarry"]["resource"="sand"]({bbox});'
        f'way["man_made"="quarry"]["resource"="sand"]({bbox});'
        f'node["landuse"="quarry"]["resource"="sand"]({bbox});'
        f'way["landuse"="quarry"]["resource"="sand"]({bbox}););out center;',
        # Sand quarries by name
        f'[out:json][timeout:60];(node["man_made"="quarry"]["name"~"sand",i]({bbox});'
        f'way["man_made"="quarry"]["name"~"sand",i]({bbox}););out center;',
        # Sand extraction / mining tags
        f'[out:json][timeout:60];(node["industrial"="sand_mining"]({bbox});'
        f'way["industrial"="sand_mining"]({bbox});'
        f'node["extraction"="sand"]({bbox});'
        f'way["extraction"="sand"]({bbox});'
        f'node["natural"="sand"]["man_made"~"quarry|mine"]({bbox}););out center;',
        # Dredging operations
        f'[out:json][timeout:60];(node["man_made"="dredger"]({bbox});'
        f'way["man_made"="dredger"]({bbox});'
        f'node["waterway"="dredging"]({bbox});'
        f'way["waterway"="dredging"]({bbox}););out center;',
        # General quarries near rivers (by name pattern)
        f'[out:json][timeout:60];(node["landuse"="quarry"]["name"~"river|nadi|sand",i]({bbox});'
        f'way["landuse"="quarry"]["name"~"river|nadi|sand",i]({bbox}););out center;',
    ]

    for q in osm_queries:
        try:
            r = requests.post(overpass, data={'data': q},
                              headers=HEADERS, timeout=70)
            if r.status_code == 200:
                for el in r.json().get('elements', []):
                    lat = el.get('lat') or el.get('center', {}).get('lat')
                    lon = el.get('lon') or el.get('center', {}).get('lon')
                    if lat and lon and _in_india(float(lat), float(lon)):
                        tags = el.get('tags', {})
                        results.append(_rec(lat, lon, source='OpenStreetMap',
                            place=tags.get('name', 'OSM quarry'),
                            confidence=0.9))
            time.sleep(3)
        except Exception as ex:
            print(f"      OSM error: {ex}")
            time.sleep(5)

    print(f"      -> {len(results)} locations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — PANGAEA / GEE global mining polygons
# ══════════════════════════════════════════════════════════════════════════════

def scrape_pangaea_gee():
    if not EE_OK:
        print("  [2/15] PANGAEA/GEE — skipped (ee not available)")
        return []
    print("  [2/15] PANGAEA Global Mining via GEE...")
    results = []
    try:
        india = ee.Geometry.BBox(68, 6.5, 98, 37.5)
        fc    = (ee.FeatureCollection(
                    'projects/sat-io/open-datasets/global-mining/global_mining_polygons')
                 .filterBounds(india).limit(2000))
        for feat in fc.getInfo().get('features', []):
            geom  = feat.get('geometry', {})
            props = feat.get('properties', {})
            gtype = geom.get('type', '')
            coords = geom.get('coordinates', [])
            lat, lon = None, None
            if gtype == 'Point':
                lon, lat = coords[0], coords[1]
            elif gtype == 'Polygon' and coords:
                ring = coords[0]
                lon  = sum(c[0] for c in ring) / len(ring)
                lat  = sum(c[1] for c in ring) / len(ring)
            elif gtype == 'MultiPolygon' and coords:
                ring = coords[0][0]
                lon  = sum(c[0] for c in ring) / len(ring)
                lat  = sum(c[1] for c in ring) / len(ring)
            if lat and lon and _in_india(lat, lon):
                results.append(_rec(lat, lon, source='PANGAEA_GEE',
                    place=props.get('SITE_NAME', props.get('site_name', '')),
                    confidence=0.85))
    except Exception as ex:
        print(f"      GEE error: {ex}")
    print(f"      -> {len(results)} locations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — Wikidata SPARQL
# ══════════════════════════════════════════════════════════════════════════════

def scrape_wikidata():
    print("  [3/15] Wikidata SPARQL...")
    results = []
    sparql = """
    SELECT ?place ?placeLabel ?lat ?lon WHERE {
      ?place wdt:P31/wdt:P279* wd:Q820477.
      ?place wdt:P17 wd:Q668.
      ?place wdt:P625 ?coord.
      BIND(geof:latitude(?coord) AS ?lat)
      BIND(geof:longitude(?coord) AS ?lon)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 500
    """
    try:
        r = requests.get('https://query.wikidata.org/sparql',
                         params={'query': sparql, 'format': 'json'},
                         headers={**HEADERS,
                                  'Accept': 'application/sparql-results+json'},
                         timeout=30)
        if r.status_code == 200:
            for b in r.json().get('results', {}).get('bindings', []):
                try:
                    lat = float(b['lat']['value'])
                    lon = float(b['lon']['value'])
                    if _in_india(lat, lon):
                        results.append(_rec(lat, lon, source='Wikidata',
                            place=b.get('placeLabel', {}).get('value', ''),
                            confidence=0.8))
                except Exception:
                    pass
        time.sleep(2)
    except Exception as ex:
        print(f"      Wikidata error: {ex}")
    print(f"      -> {len(results)} locations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — USGS MRDS
# ══════════════════════════════════════════════════════════════════════════════

def scrape_usgs_mrds():
    print("  [4/15] USGS Mineral Resources Data System...")
    results = []
    url = ('https://mrdata.usgs.gov/services/wfs/mrds?service=WFS&version=1.0.0'
           '&request=GetFeature&typeName=mrds&outputFormat=application/json'
           "&CQL_FILTER=COUNTRY_NAME='India' AND COMMOD1 LIKE '%25sand%25'"
           '&maxFeatures=500')
    data = _cached_get(url)
    if data:
        try:
            gj = json.loads(data['text'])
            for feat in gj.get('features', []):
                coords = feat.get('geometry', {}).get('coordinates', [])
                props  = feat.get('properties', {})
                if len(coords) >= 2:
                    lon, lat = float(coords[0]), float(coords[1])
                    if _in_india(lat, lon):
                        results.append(_rec(lat, lon, source='USGS_MRDS',
                            place=props.get('DEP_NAME', ''), confidence=0.85))
        except Exception:
            pass
    print(f"      -> {len(results)} locations")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5 — India Sand Watch
# ══════════════════════════════════════════════════════════════════════════════

def scrape_india_sand_watch():
    if not BS4_OK:
        return []
    print("  [5/15] India Sand Watch (Veditum)...")
    results = []
    gps_pat = r'(\d{1,2}\.\d{3,})\s*[,/]\s*(\d{2,3}\.\d{3,})'
    base_urls = [
        'https://sandwatch.envmonitoring.in/',
        'https://veditum.org/india-sand-watch/',
        'https://veditum.org/?s=sand+mining',
        'https://veditum.org/category/india-sand-watch/',
        'https://veditum.org/2025/06/12/isw-pr-mpdata/',
    ]
    for base in base_urls:
        data = _cached_get(base)
        if not data:
            continue
        soup = BeautifulSoup(data['text'], 'html.parser')
        text = soup.get_text(' ')
        for m in re.findall(gps_pat, text):
            try:
                lat, lon = float(m[0]), float(m[1])
                if _in_india(lat, lon):
                    results.append(_rec(lat, lon, source='IndiaSandWatch',
                                        url=base, confidence=0.95))
            except Exception:
                pass
        for loc in extract_locs(text):
            results.append(_text_rec(loc, 'IndiaSandWatch', base))
        # Follow article links
        links = [a['href'] for a in soup.find_all('a', href=True)
                 if 'veditum.org' in a.get('href','')]
        for link in links[:20]:
            art = _cached_get(link)
            if not art:
                continue
            at = BeautifulSoup(art['text'], 'html.parser').get_text(' ')
            for m in re.findall(gps_pat, at):
                try:
                    lat, lon = float(m[0]), float(m[1])
                    if _in_india(lat, lon):
                        results.append(_rec(lat, lon, source='IndiaSandWatch',
                                            url=link, confidence=0.95))
                except Exception:
                    pass
            if 'sand mining' in at.lower():
                for loc in extract_locs(at):
                    results.append(_text_rec(loc, 'IndiaSandWatch', link))
            time.sleep(0.4)
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 6 — SANDRP
# ══════════════════════════════════════════════════════════════════════════════

def scrape_sandrp():
    if not BS4_OK:
        return []
    print("  [6/15] SANDRP...")
    results = []
    for url in ['https://sandrp.in/?s=sand+mining',
                'https://sandrp.in/category/sand-mining/',
                'https://sandrp.in/?s=illegal+sand+mining']:
        data = _cached_get(url)
        if not data:
            continue
        soup = BeautifulSoup(data['text'], 'html.parser')
        links = list(set(a['href'] for a in soup.find_all('a', href=True)
                         if 'sandrp.in' in a.get('href','') and
                         any(k in a['href'].lower()
                             for k in ['sand','mining','river'])))
        for link in links[:15]:
            art = _cached_get(link)
            if not art:
                continue
            text = BeautifulSoup(art['text'], 'html.parser').get_text(' ')
            if 'sand' in text.lower():
                for loc in extract_locs(text):
                    results.append(_text_rec(loc, 'SANDRP', link))
            time.sleep(0.5)
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 7 — India Water Portal
# ══════════════════════════════════════════════════════════════════════════════

def scrape_indiawaterportal():
    if not BS4_OK:
        return []
    print("  [7/15] India Water Portal...")
    results = []
    for url in ['https://www.indiawaterportal.org/topics/sand-mining',
                'https://www.indiawaterportal.org/search?query=sand+mining']:
        data = _cached_get(url)
        if not data:
            continue
        text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
        for loc in extract_locs(text):
            results.append(_text_rec(loc, 'IndiaWaterPortal', url))
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 8 — Down to Earth
# ══════════════════════════════════════════════════════════════════════════════

def scrape_downtoearth():
    if not BS4_OK:
        return []
    print("  [8/15] Down to Earth...")
    results = []
    for base_url in ['https://www.downtoearth.org.in/topic/sand-mining',
                     'https://www.downtoearth.org.in/search?q=riverbed+mining+india']:
        data = _cached_get(base_url)
        if not data:
            continue
        soup = BeautifulSoup(data['text'], 'html.parser')
        links = list(set(
            urljoin('https://www.downtoearth.org.in', a['href'])
            for a in soup.find_all('a', href=True)
            if any(k in a.get('href','') for k in ['/news/','/blog/','/mining/'])
        ))
        for link in links[:15]:
            art = _cached_get(link)
            if not art:
                continue
            text = BeautifulSoup(art['text'], 'html.parser').get_text(' ')
            if 'sand mining' in text.lower():
                for loc in extract_locs(text):
                    results.append(_text_rec(loc, 'DownToEarth', link))
            time.sleep(0.5)
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 9 — Google News RSS (10 queries)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_google_news():
    if not BS4_OK:
        return []
    print("  [9/15] Google News RSS...")
    results = []
    queries = [
        'illegal sand mining India river',
        'sand mining river district India NGT',
        'sand mafia India river arrested',
        'riverbed mining India High Court',
        'sand mining West Bengal Jharkhand',
        'sand mining Odisha Madhya Pradesh',
        'sand mining Karnataka Tamil Nadu',
        'sand mining Ganga Yamuna Narmada',
        'illegal sand mining FIR India 2024',
        'sand mining ban India Supreme Court',
    ]
    for q in queries:
        rss = (f'https://news.google.com/rss/search'
               f'?q={quote_plus(q)}&hl=en-IN&gl=IN&ceid=IN:en')
        data = _cached_get(rss)
        if not data:
            continue
        try:
            soup = BeautifulSoup(data['text'], 'xml')
            for item in soup.find_all('item')[:15]:
                title = item.find('title')
                desc  = item.find('description')
                link  = item.find('link')
                text  = (title.get_text() if title else '') + \
                        ' ' + (desc.get_text() if desc else '')
                url   = link.get_text() if link else ''
                if 'sand mining' in text.lower() or 'riverbed mining' in text.lower():
                    for loc in extract_locs(text):
                        results.append(_text_rec(loc, 'GoogleNews', url))
        except Exception:
            pass
        time.sleep(1.5)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 10 — The Hindu
# ══════════════════════════════════════════════════════════════════════════════

def scrape_the_hindu():
    if not BS4_OK:
        return []
    print("  [10/15] The Hindu...")
    results = []
    for url in ['https://www.thehindu.com/tag/sand-mining/',
                'https://www.thehindu.com/search/?q=illegal+sand+mining+river']:
        data = _cached_get(url)
        if data:
            text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
            for loc in extract_locs(text):
                results.append(_text_rec(loc, 'TheHindu', url))
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 11 — NDTV
# ══════════════════════════════════════════════════════════════════════════════

def scrape_ndtv():
    if not BS4_OK:
        return []
    print("  [11/15] NDTV...")
    results = []
    for url in ['https://www.ndtv.com/topic/sand-mining',
                'https://www.ndtv.com/search?searchtext=sand+mining+river+india']:
        data = _cached_get(url)
        if data:
            text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
            if 'sand mining' in text.lower():
                for loc in extract_locs(text):
                    results.append(_text_rec(loc, 'NDTV', url))
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 12 — Times of India
# ══════════════════════════════════════════════════════════════════════════════

def scrape_toi():
    if not BS4_OK:
        return []
    print("  [12/15] Times of India...")
    results = []
    for url in ['https://timesofindia.indiatimes.com/topic/sand-mining',
                'https://timesofindia.indiatimes.com/topic/illegal-sand-mining']:
        data = _cached_get(url)
        if data:
            text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
            for loc in extract_locs(text):
                results.append(_text_rec(loc, 'TimesOfIndia', url))
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 13 — NGT India
# ══════════════════════════════════════════════════════════════════════════════

def scrape_ngt():
    if not BS4_OK:
        return []
    print("  [13/15] NGT India...")
    results = []
    for url in ['https://www.greentribunal.gov.in/orderlist.aspx',
                'https://www.greentribunal.gov.in/Search.aspx']:
        data = _cached_get(url)
        if data:
            text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
            if 'sand mining' in text.lower():
                for loc in extract_locs(text):
                    loc['confidence'] = 0.9
                    results.append(_text_rec(loc, 'NGT', url))
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 14 — data.gov.in
# ══════════════════════════════════════════════════════════════════════════════

def scrape_data_gov_in():
    print("  [14/15] data.gov.in...")
    results = []
    gps_pat = r'(\d{1,2}\.\d{4,})\s*[,]\s*(\d{2,3}\.\d{4,})'
    urls = [
        'https://api.data.gov.in/resource/search?q=sand+mining&format=json&limit=10',
        'https://www.data.gov.in/sector/Mining',
    ]
    for url in urls:
        data = _cached_get(url)
        if data:
            text = data['text']
            for m in re.findall(gps_pat, text):
                try:
                    lat, lon = float(m[0]), float(m[1])
                    if _in_india(lat, lon):
                        results.append(_rec(lat, lon, source='data.gov.in',
                                            confidence=0.85))
                except Exception:
                    pass
        time.sleep(1)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 15 — PRS India
# ══════════════════════════════════════════════════════════════════════════════

def scrape_prs_india():
    if not BS4_OK:
        return []
    print("  [15/15] PRS India...")
    results = []
    data = _cached_get('https://prsindia.org/policy/analytical-reports/sand-mining')
    if data:
        text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
        for loc in extract_locs(text):
            loc['confidence'] = 0.8
            results.append(_text_rec(loc, 'PRSIndia',
                'https://prsindia.org/policy/analytical-reports/sand-mining'))
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 16 — Nitter (Twitter/X mirrors)  — sand-mining tweets w/ place mentions
# ══════════════════════════════════════════════════════════════════════════════

NITTER_INSTANCES = [
    'https://nitter.net',
    'https://nitter.poast.org',
    'https://nitter.privacydev.net',
    'https://nitter.lucabased.xyz',
    'https://nitter.kavin.rocks',
]

NITTER_QUERIES = [
    'illegal sand mining india',
    'sand mafia river',
    'riverbed sand mining',
    'sand mining ngt',
    'sand mining arrested river',
]

def scrape_nitter():
    """
    Scrape Twitter/X via Nitter mirrors (no API key needed).
    Tries multiple instances until one responds, then mines tweet text for
    Indian river/place mentions using the shared extract_locs() patterns.
    """
    if not BS4_OK:
        return []
    print("  [16] Nitter (Twitter/X mirrors)...")
    results = []
    for q in NITTER_QUERIES:
        got = False
        for inst in NITTER_INSTANCES:
            if got:
                break
            url = f"{inst}/search?f=tweets&q={quote_plus(q)}"
            data = _cached_get(url, ttl_hours=24)
            if not data:
                continue
            try:
                soup = BeautifulSoup(data['text'], 'html.parser')
                tweets = soup.select('.tweet-content, .timeline-item')
                if not tweets:
                    continue
                got = True
                for tw in tweets[:25]:
                    text = tw.get_text(' ', strip=True)
                    if not text:
                        continue
                    low = text.lower()
                    if 'sand' in low and ('min' in low or 'mafia' in low or 'dredg' in low):
                        for loc in extract_locs(text):
                            loc['confidence'] = 0.6   # social media = lower trust
                            results.append(_text_rec(loc, 'Nitter', url))
            except Exception:
                continue
        time.sleep(1.0)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 17 — GDELT  (global news event DB, geolocated)  — strong external source
# ══════════════════════════════════════════════════════════════════════════════

def scrape_gdelt():
    """
    GDELT DOC 2.0 API — worldwide news monitoring, returns matching articles as
    JSON. We pull sand-mining articles mentioning India and mine their titles for
    river/place names. No key required.
    """
    print("  [17] GDELT global news DB...")
    results = []
    queries = [
        '"sand mining" india river',
        '"illegal sand mining" india',
        '"sand mafia" india',
        'riverbed mining india',
    ]
    for q in queries:
        url = ('https://api.gdeltproject.org/api/v2/doc/doc'
               f'?query={quote_plus(q)}&mode=ArtList&format=json'
               '&maxrecords=50&sort=DateDesc')
        data = _cached_get(url, ttl_hours=24)
        if not data:
            continue
        try:
            payload = json.loads(data['text'])
            for art in payload.get('articles', []):
                text = (art.get('title', '') or '')
                link = art.get('url', '')
                if 'sand' in text.lower():
                    for loc in extract_locs(text):
                        loc['confidence'] = 0.75
                        results.append(_text_rec(loc, 'GDELT', link))
        except Exception:
            pass
        time.sleep(1.0)
    print(f"      -> {len(results)} records")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 18 — Extra Indian news outlets (Mongabay, The Wire, Scroll,
#             Indian Express, Hindustan Times, New Indian Express)
# ══════════════════════════════════════════════════════════════════════════════

def scrape_extra_news():
    """Additional environment/news outlets that frequently cover river sand mining."""
    if not BS4_OK:
        return []
    print("  [18] Extra news outlets (Mongabay/TheWire/Scroll/IE/HT)...")
    results = []
    outlets = [
        ('Mongabay',       'https://india.mongabay.com/?s=sand+mining'),
        ('Mongabay',       'https://india.mongabay.com/list/environment/sand-mining/'),
        ('TheWire',        'https://thewire.in/?s=sand+mining'),
        ('Scroll',         'https://scroll.in/search?q=sand%20mining'),
        ('IndianExpress',  'https://indianexpress.com/?s=illegal+sand+mining'),
        ('HindustanTimes', 'https://www.hindustantimes.com/topic/sand-mining'),
        ('NewIndianExpress','https://www.newindianexpress.com/topic/Sand_mining'),
    ]
    for name, url in outlets:
        data = _cached_get(url)
        if not data:
            continue
        try:
            text = BeautifulSoup(data['text'], 'html.parser').get_text(' ')
            if 'sand mining' in text.lower() or 'riverbed' in text.lower():
                for loc in extract_locs(text):
                    loc['confidence'] = 0.7
                    results.append(_text_rec(loc, name, url))
        except Exception:
            pass
        time.sleep(1.0)
    print(f"      -> {len(results)} records")
    return results



    needs = [r for r in records if not r.get('lat')]
    has   = [r for r in records if r.get('lat')]
    print(f"\n  Geocoding {len(needs)} text mentions (~{len(needs)} sec)...")
    geocoded = list(has)
    seen = {}
    for i, rec in enumerate(needs):
        place = (rec.get('place') or '').strip()
        if not place:
            continue
        key = place.lower()
        if key in seen:
            lat, lon = seen[key]
        else:
            lat, lon = geocode(place, river=rec.get('river'),
                               state=rec.get('state'))
            seen[key] = (lat, lon)
            time.sleep(1.1)
        if lat and lon:
            rec['lat'] = round(lat, 6)
            rec['lon'] = round(lon, 6)
            geocoded.append(rec)
        if (i+1) % 20 == 0:
            print(f"    {i+1}/{len(needs)} geocoded ({len(geocoded)} with coords)")
    return geocoded


# ══════════════════════════════════════════════════════════════════════════════
# Spatial deduplication
# ══════════════════════════════════════════════════════════════════════════════

def _deduplicate(df, radius_km=0.5):
    if len(df) == 0:
        return df
    df   = df.sort_values('confidence', ascending=False).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    lats = np.radians(df['lat'].values)
    lons = np.radians(df['lon'].values)
    for i in range(len(df)):
        if not keep[i]:
            continue
        for j in range(i+1, len(df)):
            if not keep[j]:
                continue
            dlat = lats[j] - lats[i]
            dlon = lons[j] - lons[i]
            a    = (math.sin(dlat/2)**2
                    + math.cos(lats[i]) * math.cos(lats[j]) * math.sin(dlon/2)**2)
            if 6371 * 2 * math.asin(math.sqrt(max(0, min(1, a)))) < radius_km:
                keep[j] = False
    return df[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_scraper(river_name=None, state=None, use_cache=True):
    """
    Run all 15 scrapers → geocode → deduplicate → save CSV.
    Integrates with existing pipeline.
    """
    if use_cache and os.path.exists(SCRAPED_FILE):
        print(f"\n[Scraper] Loading cache: {SCRAPED_FILE}")
        df = pd.read_csv(SCRAPED_FILE)
        print(f"  {len(df)} cached locations")
        return df

    print("\n" + "="*60)
    print("  SAND MINING DATA AGGREGATION — 18 SOURCES")
    print("="*60)
    t0 = time.time()

    all_recs = []
    all_recs += scrape_osm()
    all_recs += scrape_pangaea_gee()
    all_recs += scrape_wikidata()
    all_recs += scrape_usgs_mrds()
    all_recs += scrape_india_sand_watch()
    all_recs += scrape_sandrp()
    all_recs += scrape_indiawaterportal()
    all_recs += scrape_downtoearth()
    all_recs += scrape_google_news()
    all_recs += scrape_the_hindu()
    all_recs += scrape_ndtv()
    all_recs += scrape_toi()
    all_recs += scrape_ngt()
    all_recs += scrape_data_gov_in()
    all_recs += scrape_prs_india()
    all_recs += scrape_nitter()
    all_recs += scrape_gdelt()
    all_recs += scrape_extra_news()

    print(f"\n  Raw records: {len(all_recs)}")
    all_recs = geocode_all(all_recs)

    df = pd.DataFrame(all_recs)
    df = df.dropna(subset=['lat','lon'])
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    df = df.dropna(subset=['lat','lon'])
    df = df[df.apply(lambda r: _in_india(r['lat'], r['lon']), axis=1)]

    for col, default in [('label',1),('confidence',0.7),
                         ('source',''),('place',''),
                         ('river',''),('state',''),('url','')]:
        if col not in df.columns:
            df[col] = default

    df['label']      = df['label'].fillna(1).astype(int)
    df['confidence'] = df['confidence'].fillna(0.7).astype(float)
    df = _deduplicate(df, radius_km=0.5)

    if river_name and 'river' in df.columns:
        mask = df['river'].str.contains(river_name, case=False, na=False)
        if mask.any():
            df = df[mask]
    if state and 'state' in df.columns:
        mask = df['state'].str.contains(state, case=False, na=False)
        if mask.any():
            df = df[mask]

    df = df.reset_index(drop=True)
    os.makedirs(os.path.dirname(SCRAPED_FILE), exist_ok=True)
    df.to_csv(SCRAPED_FILE, index=False)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  DONE: {len(df)} unique mining locations ({elapsed:.1f} min)")
    if 'source' in df.columns:
        for src, cnt in df['source'].value_counts().items():
            print(f"    {src:<28} {cnt:>4}")
    print(f"  Saved: {SCRAPED_FILE}")
    print(f"{'='*60}")
    return df