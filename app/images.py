import base64
import unicodedata
from pathlib import Path
import pandas as pd

# Utilities for image lookup (drivers & racetracks)

def _normalize_name(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    for ch in ['_', '-', '.', '(', ')']:
        s = s.replace(ch, ' ')
    s = ' '.join(part for part in s.split() if not part.isdigit())
    return s


def build_driver_image_index() -> dict:
    img_dir = Path('ressources') / 'drivers_images'
    index = {}
    if not img_dir.exists():
        return index
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue
        key = _normalize_name(p.stem)
        if key not in index:
            index[key] = p
        join_key = ''.join(key.split())
        if join_key and join_key not in index:
            index[join_key] = p
        token_key = ' '.join([t for t in key.split() if len(t) > 2])
        if token_key and token_key not in index:
            index[token_key] = p
        parts = [t for t in key.split() if t]
        if parts:
            last = parts[-1]
            if last and last not in index:
                index[last] = p
    return index


def find_driver_image_path(driver_name: str):
    idx = build_driver_image_index()
    if not idx:
        return None
    key = _normalize_name(driver_name)
    if key in idx:
        return idx[key]
    for k, p in idx.items():
        if key == k or key in k or k in key:
            return p
    parts = [t for t in key.split() if t]
    if parts:
        surname = parts[-1]
        first_initial = parts[0][0] if parts[0] else ''
        candidates = [(k, p) for k, p in idx.items() if surname in k]
        if candidates:
            for ck, cp in candidates:
                ck_parts = [t for t in ck.split() if t]
                if ck_parts and first_initial and ck_parts[0].startswith(first_initial):
                    return cp
            return candidates[0][1]
    try:
        import difflib
        keys = list(idx.keys())
        best = difflib.get_close_matches(key, keys, n=1, cutoff=0.8)
        if best:
            return idx[best[0]]
        if parts:
            surname = parts[-1]
            best2 = difflib.get_close_matches(surname, keys, n=1, cutoff=0.8)
            if best2:
                return idx[best2[0]]
    except Exception:
        pass
    return None


def driver_image_b64(driver_name: str):
    p = find_driver_image_path(driver_name)
    if not p:
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode('ascii')
    except Exception:
        return None


# Racetrack image utilities

def build_racetrack_image_index() -> dict:
    img_dir = Path('ressources') / 'racetracks'
    index = {}
    if not img_dir.exists():
        return index
    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.webp', '.svg']:
            continue
        key = _normalize_name(p.stem)
        if key and key not in index:
            index[key] = p
        join_key = ''.join(key.split())
        if join_key and join_key not in index:
            index[join_key] = p
        token_key = ' '.join([t for t in key.split() if len(t) > 2])
        if token_key and token_key not in index:
            index[token_key] = p
        parts = [t for t in key.split() if t]
        if parts:
            last = parts[-1]
            if last and last not in index:
                index[last] = p
    return index


def _normalize_track_tokens(s: str) -> str:
    s = _normalize_name(s)
    generic = {
        'circuit', 'grand', 'prix', 'gp', 'international', 'autodromo', 'autódromo',
        'autodrome', 'street', 'ring', 'park', 'corniche', 'national', 'internazionale',
        'de', 'del', 'dos', 'do', 'di', 'la', 'le', 'los', 'las', 'the'
    }
    tokens = [t for t in s.split() if t not in generic]
    return ' '.join(tokens) if tokens else s


def find_racetrack_image_path(*circuit_candidates):
    idx = build_racetrack_image_index()
    if not idx:
        return None
    raw_inputs = [str(c) for c in circuit_candidates if c]
    if not raw_inputs:
        return None
    candidates = []
    for raw in raw_inputs:
        n1 = _normalize_name(raw)
        n2 = _normalize_track_tokens(raw)
        for n in (n1, n2):
            if n and n not in candidates:
                candidates.append(n)
    for k in candidates:
        if k in idx:
            return idx[k]
    for k in candidates:
        for j, p in idx.items():
            if k == j or k in j or j in k:
                return p
    cand_tokens = set(t for k in candidates for t in (k or '').split() if t)
    if cand_tokens:
        for j, p in idx.items():
            j_tokens = set(j.split())
            if cand_tokens & j_tokens:
                return p
    try:
        import difflib
        keys = list(idx.keys())
        best_score = 0.0
        best_key = None
        for v in candidates:
            best = difflib.get_close_matches(v, keys, n=1, cutoff=0.78)
            if best:
                return idx[best[0]]
            for k in keys:
                ratio = difflib.SequenceMatcher(None, v, k).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_key = k
        if best_score >= 0.70 and best_key:
            return idx[best_key]
    except Exception:
        pass
    return None


def racetrack_image_b64(circuit_ref: str = None, circuit_name: str = None):
    p = find_racetrack_image_path(circuit_ref, circuit_name)
    if not p:
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode('ascii')
    except Exception:
        return None
