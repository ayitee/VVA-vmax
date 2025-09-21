import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from functools import lru_cache
import time
import unicodedata
from images import (
    _normalize_name,
    build_driver_image_index,
    find_driver_image_path,
    driver_image_b64,
    build_racetrack_image_index,
    _normalize_track_tokens,
    find_racetrack_image_path,
    racetrack_image_b64,
    f1_logo_b64,
)

# Paths
DATA_DIR = Path('data')
MODEL_PATH = Path('model_f1_rf.joblib')
 

@st.cache_data(show_spinner=True)
def load_csvs() -> dict:
    files = {
        'circuits': 'circuits.csv',
        'constructors': 'constructors.csv',
        'constructor_results': 'constructor_results.csv',
        'constructor_standings': 'constructor_standings.csv',
        'drivers': 'drivers.csv',
        'driver_standings': 'driver_standings.csv',
        'lap_times': 'lap_times.csv',
        'pit_stops': 'pit_stops.csv',
        'qualifying': 'qualifying.csv',
        'races': 'races.csv',
        'results': 'results.csv',
        'seasons': 'seasons.csv',
        'sprint_results': 'sprint_results.csv',
        'status': 'status.csv',
    }

    out = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            out[key] = pd.DataFrame()

    # Basic dtype adjustments
    if not out['races'].empty:
        out['races']['date'] = pd.to_datetime(out['races']['date'], errors='coerce')
    if not out['drivers'].empty and 'dob' in out['drivers'].columns:
        out['drivers']['dob'] = pd.to_datetime(out['drivers']['dob'], errors='coerce')
    if not out['qualifying'].empty and 'position' in out['qualifying'].columns:
        out['qualifying']['position'] = pd.to_numeric(out['qualifying']['position'], errors='coerce')
    if not out['results'].empty:
        for c in ['grid', 'position', 'points', 'rank']:
            if c in out['results'].columns:
                out['results'][c] = pd.to_numeric(out['results'][c], errors='coerce')

    return out

@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

# ----------------------------
# Feature engineering (mirrors notebook logic)
# ----------------------------
@st.cache_data(show_spinner=True)
def build_feature_table(d):
    circuits = d['circuits']
    constructors = d['constructors']
    drivers = d['drivers']
    qualifying = d['qualifying']
    races = d['races']
    results = d['results']
    sprint_results = d['sprint_results']
    lap_times = d['lap_times']
    pit_stops = d['pit_stops']
    status = d['status']
    driver_standings = d['driver_standings']
    constructor_standings = d['constructor_standings']

    # Base assembly
    base = results.merge(
        races[['raceId','year','round','circuitId','date','name']].rename(columns={'name':'race_name'}),
        on='raceId', how='left'
    )
    base = base.merge(
        drivers[['driverId','driverRef','code','dob','nationality']].rename(columns={'code':'driver_code'}),
        on='driverId', how='left'
    )
    base = base.merge(
        constructors[['constructorId','name','nationality']].rename(columns={'name':'constructor_name','nationality':'constructor_nat'}),
        on='constructorId', how='left'
    )
    base = base.merge(
        circuits[['circuitId','name','country','location']].rename(columns={'name':'circuit_name','country':'circuit_country'}),
        on='circuitId', how='left'
    )

    # Qualif min position per race/driver
    qpos = qualifying.groupby(['raceId','driverId'], as_index=False)['position'].min().rename(columns={'position':'q_position'})
    base = base.merge(qpos, on=['raceId','driverId'], how='left')

    # Sprint points before race
    sprint_pts = sprint_results.groupby(['raceId','driverId'], as_index=False)['points'].sum().rename(columns={'points':'sprint_points'})
    base = base.merge(sprint_pts, on=['raceId','driverId'], how='left')
    base['sprint_points'] = base['sprint_points'].fillna(0.0)

    # Target (unused in prediction but kept)
    base['target_points_scored'] = (base['points'] > 0).astype(int)

    # Driver cumulative features
    base = base.sort_values(['driverId','date'])
    group = base.groupby('driverId', group_keys=False)
    base['driver_prev_races'] = group.cumcount()
    base['driver_cum_points'] = group['points'].apply(lambda s: s.shift(1).cumsum())
    base['driver_cum_wins'] = group['position'].apply(lambda s: (s.eq(1).shift(1)).cumsum())
    base['driver_last_grid'] = group['grid'].shift(1)
    base['driver_avg_grid_5'] = group['grid'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    base['driver_avg_points_5'] = group['points'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Team cumulative
    base = base.sort_values(['constructorId','date'])
    cgroup = base.groupby('constructorId', group_keys=False)
    base['team_prev_races'] = cgroup.cumcount()
    base['team_cum_points'] = cgroup['points'].apply(lambda s: s.shift(1).cumsum())
    base['team_avg_points_5'] = cgroup['points'].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    # Driver age
    base['driver_age'] = (base['date'] - base['dob']).dt.days / 365.25

    # Driver history at circuit
    base = base.sort_values(['driverId','circuitId','date'])
    dcg = base.groupby(['driverId','circuitId'], group_keys=False)
    base['driver_circuit_prev'] = dcg.cumcount()
    base['driver_circuit_best_pos'] = dcg['position'].apply(lambda s: s.shift(1).cummin())

    # Fill NaNs created by shifts
    for col in ['driver_cum_points','driver_cum_wins','driver_last_grid','driver_avg_grid_5','driver_avg_points_5',
                'team_cum_points','team_avg_points_5','driver_age','driver_circuit_prev','driver_circuit_best_pos']:
        base[col] = base[col].fillna(0)

    # Grid feature preference
    base['grid_feature'] = base['q_position'].fillna(base['grid'])
    base['grid_feature'] = base['grid_feature'].fillna(base['grid_feature'].median())

    # Categoricals
    base['season'] = base['year'].astype(int)
    base['round'] = base['round'].astype(int)
    base['driver_nat'] = base['nationality']

    # Lap times features
    for col in ['driver_prev_best_lap_ms','driver_roll_bestlap5']:
        if col in base.columns:
            base = base.drop(columns=[col])
    lap_best = lap_times.groupby(['raceId', 'driverId'], as_index=False)['milliseconds'].min()
    lap_best = lap_best.rename(columns={'milliseconds': 'best_lap_ms'})
    lap_best = lap_best.merge(races[['raceId', 'date']], on='raceId', how='left')
    lap_best = lap_best.sort_values(['driverId', 'date'])
    lg = lap_best.groupby('driverId', group_keys=False)
    lap_best['driver_prev_best_lap_ms'] = lg['best_lap_ms'].shift(1)
    lap_best['driver_roll_bestlap5'] = lg['best_lap_ms'].shift(1).rolling(5, min_periods=1).mean()
    base = base.merge(lap_best[['raceId','driverId','driver_prev_best_lap_ms','driver_roll_bestlap5']],
                      on=['raceId','driverId'], how='left')

    # Pit stops
    for col in ['driver_prev_pit_count','driver_roll_pit5']:
        if col in base.columns:
            base = base.drop(columns=[col])
    pit_counts = pit_stops.groupby(['raceId','driverId']).size().reset_index(name='pit_count')
    pit_counts = pit_counts.merge(races[['raceId','date']], on='raceId', how='left')
    pit_counts = pit_counts.sort_values(['driverId','date'])
    pg = pit_counts.groupby('driverId', group_keys=False)
    pit_counts['driver_prev_pit_count'] = pg['pit_count'].shift(1)
    pit_counts['driver_roll_pit5'] = pg['pit_count'].shift(1).rolling(5, min_periods=1).mean()
    base = base.merge(pit_counts[['raceId','driverId','driver_prev_pit_count','driver_roll_pit5']],
                      on=['raceId','driverId'], how='left')

    # DNF ratio
    if 'driver_prev_dnf_rate' in base.columns:
        base = base.drop(columns=['driver_prev_dnf_rate'])
    res_status = results.merge(status, on='statusId', how='left')
    res_status = res_status.merge(races[['raceId','date']], on='raceId', how='left')
    finished_mask = res_status['status'].astype(str).str.contains('Finished', case=False, na=False) | \
                    res_status['status'].astype(str).str.contains('Lap', case=False, na=False)
    res_status['dnf'] = (~finished_mask).astype(int)
    res_status = res_status.sort_values(['driverId','date'])
    dg = res_status.groupby('driverId', group_keys=False)
    res_status['driver_prev_dnf_rate'] = dg['dnf'].shift(1).rolling(10, min_periods=1).mean()
    base = base.merge(res_status[['raceId','driverId','driver_prev_dnf_rate']].drop_duplicates(['raceId','driverId']),
                      on=['raceId','driverId'], how='left')

    # Standings (driver)
    for col in ['driver_prev_stand_pos','driver_prev_stand_points']:
        if col in base.columns:
            base = base.drop(columns=[col])
    _ds = driver_standings.merge(races[['raceId','date']], on='raceId', how='left')
    _ds = _ds.sort_values(['driverId','date'])
    dsg = _ds.groupby('driverId', group_keys=False)
    _ds['driver_prev_stand_pos'] = dsg['position'].shift(1)
    _ds['driver_prev_stand_points'] = dsg['points'].shift(1)
    base = base.merge(_ds[['raceId','driverId','driver_prev_stand_pos','driver_prev_stand_points']].drop_duplicates(['raceId','driverId']),
                      on=['raceId','driverId'], how='left')

    # Standings (team)
    for col in ['team_prev_stand_pos','team_prev_stand_points']:
        if col in base.columns:
            base = base.drop(columns=[col])
    _cs = constructor_standings.merge(races[['raceId','date']], on='raceId', how='left')
    _cs = _cs.sort_values(['constructorId','date'])
    csg = _cs.groupby('constructorId', group_keys=False)
    _cs['team_prev_stand_pos'] = csg['position'].shift(1)
    _cs['team_prev_stand_points'] = csg['points'].shift(1)
    base = base.merge(_cs[['raceId','constructorId','team_prev_stand_pos','team_prev_stand_points']].drop_duplicates(['raceId','constructorId']),
                      on=['raceId','constructorId'], how='left')

    # Season features
    season_rounds = races.groupby('year', as_index=False)['round'].max().rename(columns={'round':'season_rounds'})
    if 'season_rounds' in base.columns:
        base = base.drop(columns=['season_rounds'])
    base = base.merge(season_rounds, left_on='year', right_on='year', how='left')
    base['round_ratio'] = base['round'] / base['season_rounds']

    # Fill NaNs for engineered features
    for col in ['driver_prev_best_lap_ms','driver_roll_bestlap5','driver_prev_pit_count','driver_roll_pit5',
                'driver_prev_dnf_rate','driver_prev_stand_pos','driver_prev_stand_points',
                'team_prev_stand_pos','team_prev_stand_points','season_rounds','round_ratio']:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    # Weather placeholders (user can override via UI; zeros by default)
    for col in ['wx_temp','wx_humidity','wx_wind','wx_precip','wx_rain_flag','wx_heavy_rain']:
        if col not in base.columns:
            base[col] = 0
        base[col] = base[col].fillna(0)

    # Cat columns
    cat_cols = ['constructor_name','constructor_nat','circuit_name','circuit_country','driver_nat']
    feature_cols = [
        'grid_feature','sprint_points','driver_prev_races','driver_cum_points','driver_cum_wins',
        'driver_last_grid','driver_avg_grid_5','driver_avg_points_5','team_prev_races','team_cum_points',
        'team_avg_points_5','driver_age','driver_circuit_prev','driver_circuit_best_pos','season','round',
        'driver_prev_best_lap_ms','driver_roll_bestlap5','driver_prev_pit_count','driver_roll_pit5',
        'driver_prev_dnf_rate','driver_prev_stand_pos','driver_prev_stand_points',
        'team_prev_stand_pos','team_prev_stand_points','season_rounds','round_ratio',
        'wx_temp','wx_humidity','wx_wind','wx_precip','wx_rain_flag','wx_heavy_rain'
    ] + cat_cols

    # Keep only needed columns + identifiers
    keep_cols = feature_cols + ['raceId','driverId','constructorId','date']
    df_feat = base[keep_cols].copy()
    df_feat['date'] = pd.to_datetime(df_feat['date'], errors='coerce')

    # Also handy lookup fields
    df_feat = df_feat.merge(drivers[['driverId','forename','surname']], on='driverId', how='left')
    df_feat = df_feat.merge(races[['raceId','year','name']].rename(columns={'name':'race_name'}), on='raceId', how='left')
    return df_feat, feature_cols, cat_cols


# ----------------------------
# Driver images utilities
# ----------------------------
def _normalize_name(s: str) -> str:
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    # remove common punctuation and dashes/underscores and numeric suffixes like "-1"
    for ch in ['_', '-', '.', '(', ')']:
        s = s.replace(ch, ' ')
    s = ' '.join(part for part in s.split() if not part.isdigit())
    return s

@st.cache_data(show_spinner=False)
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
        # primary key
        if key not in index:
            index[key] = p
        # also add common variants to improve matching robustness
        #  - joined (no spaces)
        join_key = ''.join(key.split())
        if join_key and join_key not in index:
            index[join_key] = p
        #  - tokens longer than 2 chars joined with spaces (drops small words)
        token_key = ' '.join([t for t in key.split() if len(t) > 2])
        if token_key and token_key not in index:
            index[token_key] = p
        #  - last token (often unique place name)
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
    # 1) Exact key
    if key in idx:
        return idx[key]
    # 2) Simple containment in either direction
    for k, p in idx.items():
        if key == k or key in k or k in key:
            return p
    # 3) Surname + initial heuristic
    parts = [t for t in key.split() if t]
    if parts:
        surname = parts[-1]
        first_initial = parts[0][0] if parts[0] else ''
        candidates = [(k, p) for k, p in idx.items() if surname in k]
        if candidates:
            # Prefer candidates whose first token starts with same initial
            for ck, cp in candidates:
                ck_parts = [t for t in ck.split() if t]
                if ck_parts and first_initial and ck_parts[0].startswith(first_initial):
                    return cp
            # Else fallback to the first candidate
            return candidates[0][1]
    # 4) Fuzzy match on full name
    try:
        import difflib
        keys = list(idx.keys())
        best = difflib.get_close_matches(key, keys, n=1, cutoff=0.8)
        if best:
            return idx[best[0]]
        # 5) Fuzzy match on surname only
        if parts:
            surname = parts[-1]
            best2 = difflib.get_close_matches(surname, keys, n=1, cutoff=0.8)
            if best2:
                return idx[best2[0]]
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def estimate_race_params(data: dict, race_id: int):
    # Defaults if we can't estimate
    defaults = {
        'alpha_pace': 0.01,        # scale of pace differences per-lap relative to baseline
        'lap_noise': 0.30,         # seconds of random lap-to-lap variation
        'pit_loss': 23.0,          # seconds added on a pit stop
        'incident_rate_lap': 0.5,  # percent per lap per driver
    }
    try:
        laps = data.get('lap_times', pd.DataFrame())
        pits = data.get('pit_stops', pd.DataFrame())
        if laps is None or laps.empty:
            return defaults
        lr = laps[laps['raceId'] == race_id].copy()
        if lr.empty:
            return defaults
        lr['lap'] = pd.to_numeric(lr['lap'], errors='coerce')
        lr['milliseconds'] = pd.to_numeric(lr['milliseconds'], errors='coerce')
        lr = lr.dropna(subset=['lap', 'milliseconds'])
        lr['sec'] = lr['milliseconds'] / 1000.0
        # Basic sanity bounds
        lr = lr[(lr['sec'] >= 40.0) & (lr['sec'] <= 200.0)]

        # Identify pit laps for this race
        pit_set = set()
        if pits is not None and not pits.empty:
            pr = pits[pits['raceId'] == race_id].copy()
            if not pr.empty:
                pr['lap'] = pd.to_numeric(pr['lap'], errors='coerce')
                pr = pr.dropna(subset=['lap'])
                for _, r in pr.iterrows():
                    pit_set.add((int(r['driverId']), int(r['lap'])))

        # Compute driver medians/std on non-pit laps
        lr['is_pit_lap'] = lr.apply(lambda r: (int(r['driverId']), int(r['lap'])) in pit_set, axis=1)
        non_pit = lr[~lr['is_pit_lap']].copy()
        if non_pit.empty:
            non_pit = lr.copy()
        stats = non_pit.groupby('driverId')['sec'].agg(['median','std']).reset_index()
        medians = stats['median'].dropna().values
        stds = stats['std'].dropna().values
        # Lap noise: typical std within a driver
        lap_noise = float(np.nanmedian(stds)) if len(stds) else defaults['lap_noise']
        lap_noise = float(np.clip(lap_noise, 0.05, 1.0))
        # Alpha pace: spread between drivers relative to baseline pace
        if len(medians) >= 2:
            alpha_raw = float(np.nanstd(medians)) / float(np.nanmedian(medians))
            alpha_pace = float(np.clip(alpha_raw, 0.003, 0.02))
        else:
            alpha_pace = defaults['alpha_pace']

        # Pit loss: extra time on pit laps vs driver's non-pit median
        pit_loss = defaults['pit_loss']
        if pit_set:
            # Merge pit laps times
            pit_laps_df = lr[lr['is_pit_lap']][['driverId','lap','sec']].copy()
            if not pit_laps_df.empty and not non_pit.empty:
                med_map = non_pit.groupby('driverId')['sec'].median().to_dict()
                pit_deltas = []
                for _, r in pit_laps_df.iterrows():
                    m = med_map.get(int(r['driverId']))
                    if m and np.isfinite(m):
                        d = float(r['sec']) - float(m)
                        if 5.0 <= d <= 40.0:
                            pit_deltas.append(d)
                if pit_deltas:
                    pit_loss = float(np.median(pit_deltas))
                    pit_loss = float(np.clip(pit_loss, 8.0, 35.0))

        # Incident rate per lap: laps moderately slower than median (excluding pits)
        incident_rate_lap = defaults['incident_rate_lap']
        if not non_pit.empty:
            med_map = non_pit.groupby('driverId')['sec'].median().to_dict()
            deltas = []
            for _, r in non_pit.iterrows():
                m = med_map.get(int(r['driverId']))
                if m and np.isfinite(m):
                    deltas.append(float(r['sec']) - float(m))
            if deltas:
                deltas = np.array(deltas, dtype=float)
                # Consider "minor incidents" as +1s to +4s outliers
                minors = np.logical_and(deltas >= 1.0, deltas <= 4.0).sum()
                rate = 100.0 * (minors / max(1, len(deltas)))
                incident_rate_lap = float(np.clip(rate, 0.0, 5.0))

        return {
            'alpha_pace': alpha_pace,
            'lap_noise': lap_noise,
            'pit_loss': pit_loss,
            'incident_rate_lap': incident_rate_lap,
        }
    except Exception:
        return defaults


def racetrack_image_b64(circuit_ref: str = None, circuit_name: str = None):
    p = find_racetrack_image_path(circuit_ref, circuit_name)
    if not p:
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode('ascii')
    except Exception:
        return None


def get_expected_input_columns(model):
    """Extract the expected input column names for the model's ColumnTransformer."""
    prep = model.named_steps.get('prep', None)
    if prep is None:
        return [], []
    num_cols_expected = []
    cat_cols_expected = []
    for name, transformer, columns in prep.transformers:
        if name == 'num':
            num_cols_expected = list(columns)
        elif name == 'cat':
            cat_cols_expected = list(columns)
    return num_cols_expected, cat_cols_expected


def make_single_row(df_feat, race_id, driver_id, overrides, num_expected, cat_expected):
    row = df_feat[(df_feat['raceId'] == race_id) & (df_feat['driverId'] == driver_id)].copy()
    if row.empty:
        return None
    row = row.iloc[0:1].copy()  # keep dataframe shape

    # Apply overrides (grid, sprint, weather)
    for k, v in overrides.items():
        if v is not None and k in row.columns:
            row.loc[:, k] = v

    # Build input with expected columns only
    X = pd.DataFrame()
    for c in num_expected:
        if c in row.columns:
            X[c] = pd.to_numeric(row[c], errors='coerce').fillna(0)
        else:
            X[c] = 0
    for c in cat_expected:
        if c in row.columns:
            X[c] = row[c].astype(str).fillna("")
        else:
            X[c] = ""
    return X, row


def build_batch_X(df_r: pd.DataFrame, overrides: dict, num_expected, cat_expected) -> pd.DataFrame:
    """Build a batch input matrix X for model inference from df_r rows.
    Applies weather overrides to all rows if those columns exist in the model input.
    """
    Xb = pd.DataFrame(index=df_r.index)
    for c in num_expected:
        Xb[c] = pd.to_numeric(df_r[c], errors='coerce').fillna(0) if c in df_r.columns else 0
    for c in cat_expected:
        Xb[c] = df_r[c].astype(str).fillna("") if c in df_r.columns else ""
    # Apply weather overrides globally if provided
    for k in ['wx_temp','wx_humidity','wx_wind','wx_precip']:
        if k in Xb.columns and k in overrides and overrides[k] is not None:
            Xb[k] = overrides[k]
    for k in ['wx_rain_flag','wx_heavy_rain']:
        if k in Xb.columns and k in overrides and overrides[k] is not None:
            Xb[k] = int(overrides[k])
    return Xb

# ----------------------------
# UI
# ----------------------------
st.title("🏁 F1 Race Simulator")

if not MODEL_PATH.exists():
    st.warning("Model file not found (model_f1_rf.joblib). Train and save the model in the notebook.")
else:
    try:
        model = load_model()
        num_expected, cat_expected = get_expected_input_columns(model)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    data = load_csvs()

    with st.sidebar:
        # Top brand logo
        try:
            _logo = f1_logo_b64()
            if _logo:
                st.markdown(
                    f"""
                    <div class=\"vva-brand\"> 
                      <img src=\"data:image/png;base64,{_logo}\" alt=\"F1\" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        except Exception:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Type of prediction
        st.markdown("<div class='vva-label'>Type of prediction</div>", unsafe_allow_html=True)
        # Provide an explicit but hidden label to satisfy accessibility checks
        pred_type = st.radio("prediction_type", ["Driver", "Team"], index=0, horizontal=False, label_visibility="collapsed", key='prediction_type_radio')

        # Races: default to latest season only (screenshot shows no season control)
        races = data['races'].copy()
        circuits_df = data['circuits'][['circuitId','name']].rename(columns={'name':'circuit_name'})
        races['date'] = pd.to_datetime(races['date'], errors='coerce')
        # Latest available season
        latest_year = int(races['year'].dropna().max())
        races_y = races[races['year'] == latest_year].sort_values('date')
        # Note: keep full calendar visible, but annotate events that lack event-level rows
        results_df = data.get('results') if 'results' in data else load_csvs()['results']
        valid_race_ids = set(results_df['raceId'].unique().tolist())

        # Grand Prix selector: show both GP name and circuit name so users (and the racetrack image index)
        # can match images named after track names rather than GP event names.
        st.markdown("<div class='vva-label'>Grand Prix</div>", unsafe_allow_html=True)
        # Build display labels combining GP name and circuit name and keep mapping to raceId
        circuits_map = circuits_df.set_index('circuitId')['circuit_name'].to_dict()
        gp_display = []
        gp_map = {}
        for _, rrow in races_y.iterrows():
            cid = int(rrow['circuitId']) if not pd.isna(rrow['circuitId']) else None
            cval = circuits_map.get(cid, '') if cid is not None else ''
            base_label = f"{rrow['name']} — {cval}" if cval else f"{rrow['name']}"
            # Annotate races that have no event rows in the repo
            rid = int(rrow['raceId'])
            if rid not in valid_race_ids:
                label = f"{base_label} (no event data in repo)"
            else:
                label = base_label
            gp_display.append(label)
            gp_map[label] = int(rrow['raceId'])

        # Preselect British GP if present; else last race
        pre_idx_label = next((lbl for lbl in gp_display if 'british grand prix' in lbl.lower() or 'silverstone' in lbl.lower()), gp_display[-1])
        sel_display = st.selectbox("Grand Prix", gp_display, index=max(0, gp_display.index(pre_idx_label)))
        # Map selected display back to raceId and race_row
        race_id = int(gp_map[sel_display])
        race_row = races_y[races_y['raceId'] == race_id].iloc[0]

        # Track card: render selected circuit image from ressources/racetracks if available; otherwise show placeholder SVG
        try:
            _circuits = data['circuits'] if 'circuits' in data else load_csvs()['circuits']
            _cinfo = _circuits[['circuitId','circuitRef','name','location']].rename(columns={'name':'circuit_name'})
            _crow = _cinfo[_cinfo['circuitId'] == int(race_row['circuitId'])].iloc[0]
            c_ref = str(_crow.get('circuitRef', '') or '')
            c_name = str(_crow.get('circuit_name', '') or '')
            c_loc = str(_crow.get('location', '') or '')

            # Try multiple candidates: circuitRef, circuit name, race name, location
            b64_track = None
            b64_track = racetrack_image_b64(c_ref, c_name)
            if not b64_track:
                b64_track = racetrack_image_b64(c_ref, race_row.get('name'))
            if not b64_track and c_loc:
                b64_track = racetrack_image_b64(c_ref, c_loc)

            # If still none, try to get the matched path and load bytes directly
            if not b64_track:
                matched_path = find_racetrack_image_path(c_ref, c_name)
                if matched_path is not None and matched_path.exists():
                    try:
                        b64_track = base64.b64encode(matched_path.read_bytes()).decode('ascii')
                    except Exception:
                        b64_track = None

            # Generic token-overlap fallback: pick the indexed image whose
            # normalized key shares the most tokens with our candidates.
            if not b64_track:
                try:
                    idx = build_racetrack_image_index()
                    if idx:
                        # build candidate tokens from available fields
                        cand_src = ' '.join([str(x) for x in (c_ref, c_name, race_row.get('name'), c_loc) if x])
                        cand_norm = _normalize_name(cand_src)
                        cand_tokens = set(t for t in cand_norm.split() if t)
                        best_key = None
                        best_score = 0
                        for k, p in idx.items():
                            k_tokens = set(t for t in k.split() if t)
                            score = len(cand_tokens & k_tokens)
                            if score > best_score:
                                best_score = score
                                best_key = k
                        if best_score > 0 and best_key:
                            p = idx.get(best_key)
                            if p and p.exists():
                                try:
                                    b64_track = base64.b64encode(p.read_bytes()).decode('ascii')
                                    matched_path = p
                                except Exception:
                                    b64_track = None
                except Exception:
                    pass

            # Fallback: special-case British GP demo image if available
            if not b64_track:
                if any(k in c_name.lower() for k in ['silverstone']) or c_ref.lower() == 'silverstone' or 'silverstone' in race_row.get('name','').lower():
                    track_img_path = Path('ressources') / 'racetrack_testbritish.png'
                    if track_img_path.exists():
                        b64_track = base64.b64encode(track_img_path.read_bytes()).decode('ascii')

            if b64_track:
                matched_path = find_racetrack_image_path(c_ref, c_name)
                matched_name = matched_path.name if matched_path is not None else 'no match'
                st.markdown(
                    f"""
                    <div class=\"vva-card-lg vva-track\" style=\"padding:6px;\">\n                        <div style=\"width:100%; height:auto; display:flex; align-items:center; justify-content:center;\">\n                            <img src=\"data:image/png;base64,{b64_track}\" style=\"width:100%; height:auto; object-fit:contain; border-radius:16px; display:block;\" />\n                        </div>
                        <div style=\"margin-top:6px; font-size:0.75rem; color:#9ea4b3; text-align:center;\">Matched: {matched_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="vva-card-lg vva-track" style="min-height:260px;"> 
                      <svg viewBox="0 0 300 220" width="92%" height="92%" fill="none" xmlns="http://www.w3.org/2000/svg"> 
                        <path d="M40 160 C 60 120, 90 110, 120 130 C 150 150, 200 150, 220 120 C 240 90, 210 70, 180 80 C 150 90, 120 70, 100 60 C 80 50, 60 60, 55 85 C 50 110, 30 130, 40 160 Z" stroke="#ff3a3a" stroke-width="4" fill="none"/> 
                      </svg> 
                    </div> 
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            st.markdown(
                """
                <div class=\"vva-card-lg vva-track\" style=\"min-height:260px;\"> 
                  <svg viewBox=\"0 0 300 220\" width=\"92%\" height=\"92%\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"> 
                    <path d=\"M40 160 C 60 120, 90 110, 120 130 C 150 150, 200 150, 220 120 C 240 90, 210 70, 180 80 C 150 90, 120 70, 100 60 C 80 50, 60 60, 55 85 C 50 110, 30 130, 40 160 Z\" stroke=\"#ff3a3a\" stroke-width=\"4\" fill=\"none\"/> 
                  </svg> 
                </div> 
                """,
                unsafe_allow_html=True,
            )

                    # --- end racetrack card ---

        # Driver or Team selector depending on pred_type

                    # --- end racetrack card ---
        results_y = data['results']
        if pred_type == "Team":
            st.markdown("<div class='vva-label'>Team</div>", unsafe_allow_html=True)
            cons_df = data['constructors'][['constructorId','name']].rename(columns={'name':'constructor_name'})
            cids = results_y[results_y['raceId'] == race_id]['constructorId'].unique().tolist()
            cons_race = cons_df[cons_df['constructorId'].isin(cids)].sort_values('constructor_name')
            constructor_name = st.selectbox("Team", cons_race['constructor_name'].tolist(), index=0)
            constructor_id = int(cons_race.loc[cons_race['constructor_name'] == constructor_name, 'constructorId'].iloc[0])
            driver_id = None  # not used in Team focus; will handle later in app sections

            # Team two-driver image card
            try:
                # get both drivers for this team in this race
                dids_team = results_y[(results_y['raceId'] == race_id) & (results_y['constructorId'] == constructor_id)]['driverId'].unique().tolist()
                drv = data['drivers'][data['drivers']['driverId'].isin(dids_team)].copy()
                drv['driver_label'] = drv['forename'].fillna('') + ' ' + drv['surname'].fillna('')
                labels = drv['driver_label'].tolist()
                imgs = [driver_image_b64(n) for n in labels]
                # Build HTML
                def initials(n):
                    parts = [p for p in str(n).split() if p]
                    return (''.join(p[0] for p in parts[:2]).upper() or '?')
                item_html = []
                for name, b64 in zip(labels, imgs):
                    if b64:
                        item_html.append(f"<div class='vva-team-img-item'><img src='data:image/png;base64,{b64}' style='width:100%; height:100%; object-fit:cover; border-radius:12px;' /></div>")
                    else:
                        item_html.append(f"<div class='vva-team-img-item'>{initials(name)}</div>")
                html = f"<div class='vva-team-img'><div class='vva-team-img-inner'>{''.join(item_html)}</div></div>"
                st.markdown(html, unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.markdown("<div class='vva-label'>Driver</div>", unsafe_allow_html=True)
            # Drivers in this race; label only shows name to mimic screenshot
            dids = results_y[results_y['raceId'] == race_id]['driverId'].unique().tolist()
            drivers_df = data['drivers'][['driverId','forename','surname']].copy()
            drivers_df['driver_label'] = drivers_df['forename'] + ' ' + drivers_df['surname']
            drivers_race = drivers_df[drivers_df['driverId'].isin(dids)].sort_values('driver_label')
            # Guard: if no drivers are present for this GP, try a conservative fallback:
            # use drivers from the most recent previous race at the same circuit (if any).
            if drivers_race.empty:
                try:
                    # find previous races at same circuit with results
                    circ_id = int(race_row['circuitId']) if not pd.isna(race_row['circuitId']) else None
                    if circ_id is not None:
                        # search races earlier than current date for same circuit
                        prev_races = races_y[(races_y['circuitId'] == circ_id) & (races_y['date'] < race_row['date'])].sort_values('date', ascending=False)
                        found = False
                        for _, pr in prev_races.iterrows():
                            pr_id = int(pr['raceId'])
                            pr_dids = results_y[results_y['raceId'] == pr_id]['driverId'].unique().tolist()
                            if pr_dids:
                                drivers_race = drivers_df[drivers_df['driverId'].isin(pr_dids)].sort_values('driver_label')
                                found = True
                                st.warning(f"No event results for this race; using drivers from previous {pr['name']} ({pr['date'].date()}) as a fallback.")
                                break
                        if not found:
                            st.error("No drivers found for this Grand Prix in the dataset.")
                            st.stop()
                    else:
                        st.error("No drivers found for this Grand Prix in the dataset.")
                        st.stop()
                except Exception:
                    st.error("No drivers found for this Grand Prix in the dataset.")
                    st.stop()
            # Preselect Pierre Gasly if present to mimic screenshot
            default_idx = next((i for i, v in enumerate(drivers_race['driver_label'].tolist()) if 'Pierre Gasly' == v), 0)
            driver_label = st.selectbox("Driver", drivers_race['driver_label'].tolist(), index=default_idx)
            driver_id = int(drivers_race.loc[drivers_race['driver_label'] == driver_label, 'driverId'].iloc[0])
            # Infer constructor_id from results for this driver+race
            try:
                constructor_id = int(results_y[(results_y['raceId'] == race_id) & (results_y['driverId'] == driver_id)]['constructorId'].iloc[0])
            except Exception:
                # Fallback to the driver's most recent constructor this season
                recent_rows = results_y[(results_y['driverId'] == driver_id) & (results_y['raceId'].isin(races_y['raceId']))]
                constructor_id = int(recent_rows.sort_values('raceId', ascending=False)['constructorId'].iloc[0]) if not recent_rows.empty else int(results_y['constructorId'].mode().iloc[0])

            # Driver image card (with real image if available)
            _b64 = driver_image_b64(driver_label)
            if _b64:
                st.markdown(
                    f"""
                    <div class=\"vva-driver-img\" style=\"padding:0; background:none;\"> 
                      <img src=\"data:image/png;base64,{_b64}\" style=\"width:100%; height:auto; object-fit:cover; border-radius:16px; display:block;\" />
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class=\"vva-driver-img\" style=\"min-height:280px;\">{driver_label}</div>
                    """,
                    unsafe_allow_html=True,
                )

    # Build features (cached)
    with st.spinner("Preparing data…"):
        df_feat, feature_cols, cat_cols = build_feature_table(data)

    # Validate selection exists in data
    if pred_type == "Driver":
        row_ids = data['results'][(data['results']['raceId'] == race_id) & (data['results']['driverId'] == driver_id) & (data['results']['constructorId'] == constructor_id)]
        if row_ids.empty:
            st.error("No data found for this driver/team in this Grand Prix.")
            st.stop()
    else:  # Team
        row_ids_c = data['results'][(data['results']['raceId'] == race_id) & (data['results']['constructorId'] == constructor_id)]
        if row_ids_c.empty:
            st.error("No data found for this team in this Grand Prix.")
            st.stop()

    # Defaults from computed features (depends on mode)
    if pred_type == "Driver":
        row_default = df_feat[(df_feat['raceId'] == race_id) & (df_feat['driverId'] == driver_id)]
    else:  # Team: take a representative row for the selected constructor in this GP
        row_default = df_feat[(df_feat['raceId'] == race_id) & (df_feat['constructorId'] == constructor_id)]
    if row_default.empty:
        st.error("Unable to build features for this selection. Try another choice.")
        st.stop()
    row_default = row_default.iloc[0]

    st.subheader("Adjustments (optional)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        grid_default = int(float(row_default.get('grid_feature', 1) or 1))
        grid_default = max(1, min(40, grid_default))
        grid_feature = st.number_input(
            "Starting position",
            min_value=1,
            max_value=40,
            value=grid_default,
            step=1,
            format="%d",
            help="Grid position at race start (1 = pole)."
        )
        sp_default = float(row_default.get('sprint_points', 0.0) or 0.0)
        sp_default = max(0.0, min(10.0, sp_default))
        sprint_points = st.number_input(
            "Sprint points",
            min_value=0.0,
            max_value=10.0,
            value=sp_default,
            step=0.5,
            help="Points scored in the sprint (if applicable)."
        )
    with c2:
        wx_temp = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=float(row_default.get('wx_temp', 0.0)),
            step=0.5,
            help="Approximate ambient temperature on race day."
        )
        wx_humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(row_default.get('wx_humidity', 0.0)),
            step=1.0,
            help="Relative humidity."
        )
    with c3:
        wx_wind = st.number_input(
            "Wind (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=float(row_default.get('wx_wind', 0.0)),
            step=1.0,
            help="Average wind speed in km/h."
        )
        wx_precip = st.number_input(
            "Precipitation (mm)",
            min_value=0.0,
            max_value=100.0,
            value=float(row_default.get('wx_precip', 0.0)),
            step=0.1,
            help="Estimated rainfall total."
        )
    with c4:
        rain_default = int(float(row_default.get('wx_rain_flag', 0) or 0))
        rain_default = 0 if rain_default not in (0,1) else rain_default
        heavy_default = int(float(row_default.get('wx_heavy_rain', 0) or 0))
        heavy_default = 0 if heavy_default not in (0,1) else heavy_default
        wx_rain_choice = st.radio(
            "Rain?",
            ["No","Yes"],
            index=rain_default,
            horizontal=True,
            help="Rain expected during the race.",
            key='wx_rain_radio'
        )
        wx_heavy_choice = st.radio(
            "Heavy rain?",
            ["No","Yes"],
            index=heavy_default,
            horizontal=True,
            help="Heavy rain conditions.",
            key='wx_heavy_radio'
        )
        wx_rain_flag = 1 if wx_rain_choice == "Yes" else 0
        wx_heavy_rain = 1 if wx_heavy_choice == "Yes" else 0

    overrides = {
        'grid_feature': grid_feature,
        'sprint_points': sprint_points,
        'wx_temp': wx_temp,
        'wx_humidity': wx_humidity,
        'wx_wind': wx_wind,
        'wx_precip': wx_precip,
        'wx_rain_flag': int(wx_rain_flag),
        'wx_heavy_rain': int(wx_heavy_rain),
    }
    
    # --- Course simulée (alpha) ---
    st.subheader("Race settings (alpha)")
    # Estimate parameters from data and display as read-only values
    _params = estimate_race_params(data, race_id)
    csa1, csa2, csa3, csa4 = st.columns(4)
    with csa1:
        st.metric("Pace influence (α)", f"{_params['alpha_pace']:.3f}")
        st.caption("Derived from inter-driver median lap time spread")
    with csa2:
        st.metric("Lap noise (s)", f"{_params['lap_noise']:.2f}")
        st.caption("Median within-driver lap std (non-pit laps)")
    with csa3:
        st.metric("Pit loss (s)", f"{_params['pit_loss']:.1f}")
        st.caption("Median pit lap delta vs driver's non-pit median")
    with csa4:
        st.metric("Incidents per lap (%)", f"{_params['incident_rate_lap']:.2f}")
        st.caption("Share of +1s to +4s laps (non-pit)")

    # Fixed configuration flags
    use_hist_pits = True
    auto_seed_race = True
    seed_race = 2025
    # Unpack for simulation
    alpha_pace = float(_params['alpha_pace'])
    lap_noise = float(_params['lap_noise'])
    pit_loss = float(_params['pit_loss'])
    incident_rate_lap = float(_params['incident_rate_lap'])

    if st.button("Run race (alpha)", type="primary"):
        try:
                df_r = df_feat[df_feat['raceId'] == race_id].copy()
                if df_r.empty:
                    st.warning("No feature data for this Grand Prix.")
                else:
                    # Entrées modèle pour pace
                    Xb = build_batch_X(df_r, overrides, num_expected, cat_expected)
                    if pred_type == "Driver":
                        mask_sel = (df_r['driverId'] == driver_id)
                        if 'grid_feature' in Xb.columns and overrides.get('grid_feature') is not None:
                            Xb.loc[mask_sel, 'grid_feature'] = overrides['grid_feature']
                        if 'sprint_points' in Xb.columns and overrides.get('sprint_points') is not None:
                            Xb.loc[mask_sel, 'sprint_points'] = overrides['sprint_points']
                    else:
                        # Apply driver-specific overrides to both team drivers in Team mode
                        mask_team = (df_r['constructorId'] == constructor_id)
                        if 'grid_feature' in Xb.columns and overrides.get('grid_feature') is not None:
                            Xb.loc[mask_team, 'grid_feature'] = overrides['grid_feature']
                        if 'sprint_points' in Xb.columns and overrides.get('sprint_points') is not None:
                            Xb.loc[mask_team, 'sprint_points'] = overrides['sprint_points']

                    proba = model.predict_proba(Xb)[:, 1]
                    p = np.clip(proba, 1e-6, 1 - 1e-6)
                    base_logit = np.log(p / (1 - p))
                    # Normaliser le logit autour de 0
                    norm_logit = (base_logit - np.mean(base_logit)) / (np.std(base_logit) + 1e-8)

                    # Contexte
                    d_ids = df_r['driverId'].values
                    names = (df_r['forename'].fillna('') + ' ' + df_r['surname'].fillna('')).values
                    teams = (df_r['constructor_name'] if 'constructor_name' in df_r.columns else 
                             df_r['constructorId'].map(data['constructors'].set_index('constructorId')['name'])).astype(str).values

                    # Determine lap count and baseline time
                    laps_gp = data['lap_times'][data['lap_times']['raceId'] == race_id]
                    if laps_gp.empty:
                        n_laps = 60
                        baseline_seconds = 90.0
                    else:
                        n_laps = int(pd.to_numeric(laps_gp['lap'], errors='coerce').max() or 60)
                        try:
                            total_ms_by_driver = laps_gp.groupby('driverId')['milliseconds'].sum()
                            baseline_ms = float(total_ms_by_driver.min())
                            baseline_seconds = baseline_ms / max(1, n_laps) / 1000.0  # approx best avg lap
                        except Exception:
                            baseline_seconds = 90.0

                    # Pit plan
                    pit_plan = {int(d): [] for d in d_ids}
                    if use_hist_pits and not data['pit_stops'].empty:
                        ps = data['pit_stops'][data['pit_stops']['raceId'] == race_id]
                        if not ps.empty:
                            for d in d_ids:
                                laps_d = pd.to_numeric(ps[ps['driverId'] == d]['lap'], errors='coerce').dropna().astype(int).tolist()
                                pit_plan[int(d)] = [l for l in laps_d if 1 <= l <= n_laps]
                    # If no history: simple heuristic (1 stop mid-race if race > 45 laps)
                    if all(len(v) == 0 for v in pit_plan.values()):
                        if n_laps > 45:
                            mid = max(15, int(n_laps * 0.5))
                            for d in d_ids:
                                pit_plan[int(d)] = [mid]

                    # Per-lap DNF (based on historical rate/length)
                    if 'driver_prev_dnf_rate' in df_r.columns:
                        dnf_base = pd.to_numeric(df_r['driver_prev_dnf_rate'], errors='coerce').fillna(0).clip(0, 0.9).values
                    else:
                        dnf_base = np.zeros(len(df_r))
                    dnf_per_lap = dnf_base / max(1, n_laps)

                    # Simulation
                    eff_seed = int(time.time() * 1000) % (2**32 - 1) if auto_seed_race else int(seed_race)
                    rng = np.random.default_rng(eff_seed)
                    n = len(d_ids)
                    cum_time = np.zeros(n, dtype=float)
                    laps_done = np.zeros(n, dtype=int)
                    dnf_flag = np.zeros(n, dtype=bool)
                    n_pits = np.zeros(n, dtype=int)
                    events = []

                    for lap in range(1, n_laps + 1):
                        # Tour pour chaque pilote actif
                        for i in range(n):
                            if dnf_flag[i]:
                                continue
                            # base pace
                            lap_time = baseline_seconds * (1.0 - float(alpha_pace) * norm_logit[i])
                            lap_time += rng.normal(0.0, float(lap_noise))
                            # Minor incident with randomized time penalty
                            if rng.random() < (incident_rate_lap / 100.0):
                                penalty_s = float(rng.uniform(1.0, 4.0))
                                lap_time += penalty_s
                                events.append((lap, int(d_ids[i]), 'incident', penalty_s))
                            # arrêt au stand
                            if lap in pit_plan[int(d_ids[i])]:
                                lap_time += float(pit_loss)
                                n_pits[i] += 1
                                events.append((lap, int(d_ids[i]), 'pit', float(pit_loss)))
                            cum_time[i] += max(0.5, lap_time)
                            laps_done[i] = lap
                            # DNF probabiliste
                            if rng.random() < dnf_per_lap[i]:
                                dnf_flag[i] = True
                                events.append((lap, int(d_ids[i]), 'dnf', 0.0))

                    # Final classification
                    # Non-DNF sorted by total time, then DNF sorted by laps completed desc
                    idx_fin = np.where(~dnf_flag)[0]
                    idx_dnf = np.where(dnf_flag)[0]
                    order_fin = idx_fin[np.argsort(cum_time[idx_fin])]
                    order_dnf = idx_dnf[np.argsort(-laps_done[idx_dnf])]
                    order_all = np.concatenate([order_fin, order_dnf])

                    # Gaps vs winner (use precise seconds to avoid scaling issues; format with decimals when small)
                    if len(order_fin) > 0:
                        t0 = cum_time[order_fin[0]]
                    else:
                        t0 = np.nan

                    def fmt_total(sec):
                        if not np.isfinite(sec):
                            return 'n/d'
                        sec = float(sec)
                        h = int(sec // 3600)
                        rem = sec - h * 3600
                        m = int(rem // 60)
                        s = rem - m * 60  # fractional seconds
                        if h > 0:
                            return f"{h:d}:{m:02d}:{s:06.3f}"
                        else:
                            return f"{m:d}:{s:06.3f}"

                    def fmt_gap(sec):
                        if not np.isfinite(sec):
                            return '+n/d'
                        sec = max(0.0, float(sec))
                        if sec < 1.0:
                            return f"+{sec:.2f}s"
                        if sec < 10.0:
                            return f"+{sec:.2f}s"
                        m = int(sec // 60)
                        if m > 0:
                            s = int(sec - m * 60)
                            return f"+{m:d}:{s:02d}"
                        else:
                            return f"+{int(sec):d}s"

                    rows = []
                    for rank, i in enumerate(order_all, start=1):
                        is_dnf = bool(dnf_flag[i])
                        total = cum_time[i] if not is_dnf else np.nan
                        gap_precise = (total - t0) if (not is_dnf and np.isfinite(t0)) else np.nan
                        rows.append({
                            'Rank': rank,
                            'driverId': int(d_ids[i]),
                            'Driver': names[i],
                            'Team': teams[i],
                            'Simulated time': fmt_total(total),
                            'Simulated gap': fmt_gap(gap_precise) if np.isfinite(gap_precise) else ('DNF' if is_dnf else '+n/d'),
                            'Laps': int(laps_done[i]),
                            'Pits': int(n_pits[i]),
                            'DNF': 'Yes' if is_dnf else 'No',
                        })
                    res_df = pd.DataFrame(rows)
                    # Prepare team name for matching if in Team mode
                    if pred_type == "Team":
                        team_name_sel = constructor_name if 'constructor_name' in locals() else None
                        if not team_name_sel:
                            cmap = data['constructors'].set_index('constructorId')['name'].to_dict()
                            team_name_sel = cmap.get(constructor_id, 'Team')
                    # Add focus star(s)
                    if pred_type == "Driver":
                        sel_mask = res_df['driverId'] == int(driver_id)
                    else:
                        # Team: star both team drivers by name label if possible; else by constructorId membership
                        try:
                            sel_mask = res_df['Team'].astype(str) == str(team_name_sel)
                        except Exception:
                            sel_mask = res_df['driverId'].isin(df_r[df_r['constructorId'] == constructor_id]['driverId'].unique())
                    res_df.insert(0, '★', np.where(sel_mask, '⭐', ''))
                    st.subheader("Race result (alpha)")
                    st.dataframe(res_df[['★','Rank','Driver','Team','Simulated time','Simulated gap','Laps','Pits','DNF']], width='stretch')

                    # Compact Podium section
                    try:
                        _finishers = res_df[res_df['DNF'] == 'No'].copy().sort_values('Rank').head(3)
                        if _finishers.empty:
                            _finishers = res_df.copy().sort_values('Rank').head(3)
                        medals = [("🥇", "#ffd700"), ("🥈", "#c0c0c0"), ("🥉", "#cd7f32")]
                        st.markdown(
                            """
                            <style>
                            .vva-podium-card { background: #2f3440; border: 1px solid #444a57; border-radius: 14px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.25); display:flex; align-items:center; gap:16px; }
                            .vva-podium-media { flex: 0 0 150px; max-width: 150px; }
                            .vva-podium-media img { width: 100%; height: auto; display:block; border-radius: 12px; background:#1f2230; }
                            .vva-podium-photo { width: 150px; height: 150px; border-radius: 12px;
                                background: linear-gradient(180deg, #4b2f4b 0%, #3a2a3f 100%);
                                display:flex; align-items:center; justify-content:center; color:#e5e1ea;
                                font-weight:700; letter-spacing:0.02em; }
                            .vva-podium-info { flex: 1 1 auto; min-width: 0; }
                            .vva-podium-medal { font-size: 22px; margin-right: 10px; }
                            .vva-podium-name { font-weight: 700; color: #e7eaf3; font-size: 1.1rem; }
                            .vva-podium-team { color: #c8ccd6; font-size: 0.95rem; }
                            .vva-podium-meta { color: #9ea4b3; font-size: 0.9rem; margin-top: 6px; }
                            @media (max-width: 900px) {
                              .vva-podium-card { flex-direction: column; align-items: stretch; }
                              .vva-podium-media { flex-basis: auto; max-width: none; }
                              .vva-podium-photo { width: 100%; height: auto; aspect-ratio: 1 / 1; }
                            }

                            /* New insight card style: compact, low-padding card for Driver/Race insights */
                            .vva-insight-card { background: #22262d; border: 1px solid #3b424d; border-radius: 12px; padding: 14px; margin-top: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.25); }
                            .vva-insight-card h2, .vva-insight-card h3 { margin: 4px 0 8px 0; }
                            .vva-insight-compact { font-size: 0.95rem; color: #d7dbe3; }
                                                        /* Reduce main container padding slightly to give more width to cards/podium */
                                                        .main > div[role="main"] { padding-top: 8px !important; padding-left: 12px !important; padding-right: 12px !important; }
                                                        /* Also target Streamlit block container to reduce side gutters and allow wider content for podium/cards */
                                                        div.block-container { padding-left: 12px !important; padding-right: 12px !important; max-width: 1400px !important; }
                                                        @media (max-width: 900px) {
                                                            div.block-container { padding-left: 16px !important; padding-right: 16px !important; }
                                                        }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("### Podium 🏆")
                        pc1, pc2, pc3 = st.columns(3)
                        for idx, col in enumerate([pc1, pc2, pc3]):
                            if idx < len(_finishers):
                                r = _finishers.iloc[idx]
                                medal, color = medals[idx]
                                # Show time for P1, gap for P2/P3
                                meta_label = "Time" if idx == 0 else "Gap"
                                meta_value = (
                                    r.get('Simulated time', 'n/d') if idx == 0 else r.get('Simulated gap', '+n/d')
                                )
                                # Build placeholder initials for the photo area
                                _name = str(r['Driver']) if pd.notna(r['Driver']) else "?"
                                # Photo area: in Driver mode, avoid duplicating the selected driver's photo on the podium
                                _is_selected_driver = False
                                try:
                                    _is_selected_driver = (pred_type == "Driver" and int(r.get('driverId')) == int(driver_id))
                                except Exception:
                                    _is_selected_driver = False
                                _b64p = None if _is_selected_driver else driver_image_b64(_name)
                                if _b64p:
                                    photo_html = f"<div class=\"vva-podium-media\"><img src='data:image/png;base64,{_b64p}' alt='driver photo' /></div>"
                                else:
                                    _parts = [p for p in _name.strip().split() if p]
                                    _initials = ("".join([p[0] for p in _parts[:2]])).upper() if _parts else "?"
                                    photo_html = f"<div class=\"vva-podium-media\"><div class=\"vva-podium-photo\">{_initials}</div></div>"
                                html = f"""
                                <div class=\"vva-podium-card\">\n  {photo_html}\n  <div class=\"vva-podium-info\">\n    <div><span class=\"vva-podium-medal\" style=\"color:{color}\">{medal}</span>\n         <span class=\"vva-podium-name\">{r['Driver']}</span></div>\n    <div class=\"vva-podium-team\">{r['Team']}</div>\n    <div class=\"vva-podium-meta\">{meta_label}: {meta_value}</div>\n  </div>\n</div>\n"""
                                with col:
                                    st.markdown(html, unsafe_allow_html=True)
                    except Exception:
                        pass

                    with st.expander("Event log"):
                        if events:
                            ev_df = pd.DataFrame(events, columns=['Lap','driverId','Type','Value (s)'])
                            name_map = {int(d): n for d, n in zip(d_ids, names)}
                            ev_df['Driver'] = ev_df['driverId'].map(name_map)
                            st.dataframe(ev_df[['Lap','Driver','Type','Value (s)']])
                        else:
                            st.caption("No events.")

                    # Driver insights panel (Driver mode only)
                    if pred_type == "Driver":
                        try:
                            drv_row = res_df.loc[res_df['driverId'] == int(driver_id)].iloc[0]
                            # Points mapping (no FL bonus)
                            pts_map = [25,18,15,12,10,8,6,4,2,1]
                            rank_val = drv_row['Rank']
                            points_val = pts_map[int(rank_val)-1] if pd.notna(rank_val) and 1 <= int(rank_val) <= 10 else 0
                            # DNF lap if any
                            dnf_lap = None
                            if events:
                                _ev = pd.DataFrame(events, columns=['Lap','driverId','Type','Value (s)'])
                                _ev_sel = _ev[(_ev['driverId'] == int(driver_id))]
                                dnf_rows = _ev_sel[_ev_sel['Type'] == 'dnf']
                                if not dnf_rows.empty:
                                    dnf_lap = int(dnf_rows.iloc[0]['Lap'])
                                pit_laps = _ev_sel[_ev_sel['Type'] == 'pit']['Lap'].astype(int).tolist()
                                inc_laps = _ev_sel[_ev_sel['Type'] == 'incident']['Lap'].astype(int).tolist()
                            else:
                                pit_laps, inc_laps = [], []

                            finish_label = f"P{int(rank_val)}" if pd.notna(rank_val) else "-"
                            if drv_row['DNF'] == 'Yes' and dnf_lap is not None:
                                finish_label = f"DNF (Lap {dnf_lap})"
                            elif drv_row['DNF'] == 'Yes':
                                finish_label = "DNF"

                            pit_txt = ", ".join(map(str, pit_laps)) if pit_laps else "—"
                            inc_txt = ", ".join(map(str, inc_laps)) if inc_laps else "—"

                            # Build a full HTML card so the content renders inside the .vva-insight-card
                            drv_html = f"""
                            <div class='vva-insight-card'>
                              <h2 style='margin-top:0;margin-bottom:8px;color:#ffffff;'>Driver insights</h2>
                              <div style='display:flex;gap:32px;align-items:flex-start;flex-wrap:wrap;'>
                                <div style='flex:1;min-width:220px;'>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Driver</div>
                                  <div style='font-weight:800;font-size:1.5rem;color:#e7eaf3;margin-bottom:6px;'>{drv_row['Driver']}</div>
                                  <div style='color:#9ea4b3;font-weight:700;margin-top:6px;'>Finish</div>
                                  <div style='font-weight:700;font-size:1.15rem;margin-bottom:6px;color:#ffffff;'>{finish_label}</div>
                                  <div style='color:#9ea4b3;margin-top:8px;font-size:0.95rem;'>Pit laps: {pit_txt} · Incident laps: {inc_txt}</div>
                                </div>
                                <div style='flex:1;min-width:160px;'>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Gap</div>
                                  <div style='font-weight:800;font-size:1.5rem;color:#ffffff;margin-bottom:10px;'>{drv_row['Simulated gap']}</div>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Laps</div>
                                  <div style='font-weight:700;font-size:1.25rem;color:#ffffff;'>{int(drv_row['Laps'])}</div>
                                </div>
                                <div style='flex:1;min-width:160px;'>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Pits</div>
                                  <div style='font-weight:800;font-size:1.5rem;color:#ffffff;margin-bottom:10px;'>{int(drv_row['Pits'])}</div>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Incidents</div>
                                  <div style='font-weight:700;font-size:1.25rem;color:#ffffff;'>{len(inc_laps)}</div>
                                </div>
                                <div style='width:140px;min-width:120px;'>
                                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Points</div>
                                  <div style='font-weight:800;font-size:1.5rem;color:#ffffff;'>{points_val}</div>
                                </div>
                              </div>
                            </div>
                            """
                            st.markdown(drv_html, unsafe_allow_html=True)
                        except Exception:
                            pass

                    # Team summary panel (Team mode only)
                    if pred_type == "Team":
                        # Compute points from final rank (no FL point)
                        points_table = [25,18,15,12,10,8,6,4,2,1]
                        def points_for_rank(r):
                            try:
                                r = int(r)
                            except Exception:
                                return 0
                            return points_table[r-1] if 1 <= r <= 10 else 0

                        team_name_sel = constructor_name if 'constructor_name' in locals() else None
                        # If we don't have the label, infer from constructorId map
                        if not team_name_sel:
                            cmap = data['constructors'].set_index('constructorId')['name'].to_dict()
                            team_name_sel = cmap.get(constructor_id, 'Team')

                        res_team = res_df[res_df['Team'].astype(str) == str(team_name_sel)].copy()
                        res_team['Points'] = res_team['Rank'].apply(points_for_rank)
                        team_points_sum = int(res_team['Points'].sum()) if not res_team.empty else 0

                        st.subheader("Team summary")
                        cA, cB, cC = st.columns([1,1,1])
                        if len(res_team) >= 1:
                            r0 = res_team.iloc[0]
                            with cA:
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Driver A</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{r0['Driver']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Finish</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">P{int(r0['Rank']) if pd.notna(r0['Rank']) else '-'}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Gap</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{r0['Simulated gap']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Points</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{int(r0['Points'])}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        if len(res_team) >= 2:
                            r1 = res_team.iloc[1]
                            with cB:
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Driver B</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{r1['Driver']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Finish</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">P{int(r1['Rank']) if pd.notna(r1['Rank']) else '-'}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Gap</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{r1['Simulated gap']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f"""
                                    <div>
                                        <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.9rem; margin-bottom:2px;\">Points</div>
                                        <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.15rem;\">{int(r1['Points'])}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        with cC:
                            st.markdown(
                                f"""
                                <div>
                                    <div class=\"vva-f1-regular\" style=\"color:#e5e9f0; font-weight:700; font-size:0.95rem; margin-bottom:2px;\">Team points</div>
                                    <div class=\"vva-f1\" style=\"font-weight:700; font-size:1.25rem;\">{team_points_sum}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
        except Exception as e:
            st.error(f"La course simulée a échoué : {e}")


        # -------------------------
        # Race insights (main area)
        # -------------------------
        try:
            # basic context
            circ_id = int(race_row['circuitId']) if not pd.isna(race_row['circuitId']) else None
            race_year = int(race_row['year']) if not pd.isna(race_row['year']) else None

            # historical races at this circuit
            hist_races = races[races['circuitId'] == circ_id].sort_values('date') if circ_id is not None else pd.DataFrame()
            hist_race_ids = hist_races['raceId'].unique().tolist() if not hist_races.empty else []

            # quick aggregates from results/lap_times/pit_stops
            res_all = data['results'] if 'results' in data else pd.DataFrame()
            laps_all = data['lap_times'] if 'lap_times' in data else pd.DataFrame()
            pits_all = data['pit_stops'] if 'pit_stops' in data else pd.DataFrame()

            # historical event counts
            hist_events = len(hist_race_ids)
            # average number of drivers per historical event (unique driverIds per race)
            avg_drivers = 0.0
            med_drivers = 0.0
            if hist_race_ids:
                drv_counts = [int(res_all[res_all['raceId'] == rid]['driverId'].nunique()) for rid in hist_race_ids]
                avg_drivers = float(np.mean(drv_counts)) if drv_counts else 0.0
                med_drivers = float(np.median(drv_counts)) if drv_counts else 0.0

            # median laps for historical races (from lap_times)
            med_laps = 0
            if hist_race_ids:
                lap_counts = []
                for rid in hist_race_ids:
                    lmax = pd.to_numeric(laps_all[laps_all['raceId'] == rid]['lap'], errors='coerce').max()
                    if np.isfinite(lmax):
                        lap_counts.append(int(lmax))
                med_laps = int(np.median(lap_counts)) if lap_counts else 0

            # avg pit stops per driver (historical)
            avg_pits = 0.0
            if hist_race_ids and not pits_all.empty:
                pit_avgs = []
                for rid in hist_race_ids:
                    p = pits_all[pits_all['raceId'] == rid]
                    if not p.empty:
                        pit_avgs.append(float(p.groupby('driverId').size().mean()))
                avg_pits = float(np.mean(pit_avgs)) if pit_avgs else 0.0

            # DNF rate historical (approx from results.statusId via status mapping)
            dnf_rate = 0.0
            if hist_race_ids and not res_all.empty:
                status_df = data.get('status', pd.DataFrame())
                merged = (res_all[res_all['raceId'].isin(hist_race_ids)].merge(status_df, on='statusId', how='left')
                          if not status_df.empty else res_all[res_all['raceId'].isin(hist_race_ids)])
                if 'status' in merged.columns:
                    finished_mask = (
                        merged['status'].astype(str).str.contains('Finished', case=False, na=False)
                        | merged['status'].astype(str).str.contains('Lap', case=False, na=False)
                    )
                else:
                    finished_mask = merged['positionText'].astype(str).str.isnumeric()
                dnf_rate = 100.0 * (1.0 - (finished_mask.sum() / max(1, len(merged))))

            # Whether this particular 2024 race has event rows
            has_2024_event = int(race_id) in set(res_all['raceId'].unique().tolist())

            # Render Race insights as a styled card with HTML so everything is contained in the card
            race_html = f"""
            <div class='vva-insight-card'>
              <h2 style='margin-top:0;margin-bottom:8px;color:#ffffff;'>Race insights</h2>
              <div style='display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;'>
                <div style='flex:1;min-width:220px;'>
                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Historical events</div>
                  <div style='font-weight:800;font-size:2.25rem;color:#ffffff;margin-bottom:6px;'>{hist_events}</div>
                  <div style='color:#9ea4b3;font-size:0.95rem;'>Total races at this circuit (historical)</div>
                </div>
                <div style='flex:1;min-width:220px;'>
                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Avg drivers</div>
                  <div style='font-weight:800;font-size:2.25rem;color:#ffffff;margin-bottom:6px;'>{avg_drivers:.1f}</div>
                  <div style='color:#9ea4b3;font-size:0.95rem;'>Mean unique drivers per event</div>
                </div>
                <div style='flex:1;min-width:220px;'>
                  <div style='color:#9ea4b3;font-weight:700;margin-bottom:6px;'>Median laps</div>
                  <div style='font-weight:800;font-size:2.25rem;color:#ffffff;margin-bottom:6px;'>{med_laps:d}</div>
                  <div style='color:#9ea4b3;font-size:0.95rem;'>Median lap count (historical)</div>
                </div>
              </div>
              <div style='display:flex;gap:24px;margin-top:14px;align-items:center;flex-wrap:wrap;'>
                <div style='flex:2;min-width:240px;color:#d7dbe3;'>
                  <div style='font-weight:700;margin-bottom:6px;'>Avg pits per driver: <span style="font-weight:800;color:#ffffff;">{avg_pits:.2f}</span></div>
                  <div style='font-weight:700;margin-bottom:6px;'>DNF rate (historical): <span style="font-weight:800;color:#ffffff;">{dnf_rate:.1f}%</span></div>
                </div>
                <div style='flex:1;min-width:220px;'>
                  {'<div style="background:#1b6b3a;padding:12px;border-radius:8px;color:#fff;font-weight:700;">2024 event data: present</div>' if has_2024_event else '<div style="background:#6b4b1b;padding:12px;border-radius:8px;color:#fff;font-weight:700;">2024 event data: missing in repo</div>'}
                </div>
              </div>
            </div>
            """
            st.markdown(race_html, unsafe_allow_html=True)
        except Exception:
            pass

