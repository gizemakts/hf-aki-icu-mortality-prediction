#!/usr/bin/env python3
"""04 Urine Output With Stents.

Add ureteral-stent sources and complete UO variant construction.

This program is part of the command-line research workflow.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT
from helpers._runtime import display_table


def main() -> None:
    """Execute this workflow step."""
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))
    os.environ['HF_AKI_PROJECT_ROOT'] = str(_REPOSITORY_ROOT)
    from pathlib import Path
    import json
    import numpy as np
    import pandas as pd
    import duckdb
    CONFIRMED_MIMIC_ROOT = Path('data/raw/mimic-iv-3.1')
    CONFIRMED_PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT)))
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', CONFIRMED_MIMIC_ROOT))
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', CONFIRMED_PROJECT_ROOT))
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    PHENOTYPE_QC_DIR = PROJECT_ROOT / 'quality_control' / 'phenotype'
    for d in [COHORT_DIR, PHENOTYPE_DIR, URINE_OUTPUT_DIR, PHENOTYPE_QC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not MIMIC_ROOT.exists():
        raise FileNotFoundError(f'MIMIC_ROOT not found: {MIMIC_ROOT}')

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def require_file(p: Path):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def require_cols(df, cols, label='dataframe'):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f'{label} missing columns: {miss}')

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: null {key}')
        ndup = int(df[key].duplicated().sum())
        if ndup:
            raise AssertionError(f'{label}: duplicate {key} rows={ndup}')
    OUTCOME_TOKENS = ('expire', 'mortality', 'death', 'outcome')

    def assert_outcome_blind(df, label):
        bad = [c for c in df.columns if any((tok in c.lower() for tok in OUTCOME_TOKENS))]
        if bad:
            raise AssertionError(f'{label}: outcome-like fields prohibited: {bad}')
    print('MIMIC_ROOT     :', MIMIC_ROOT)
    print('PROJECT_ROOT   :', PROJECT_ROOT)
    print('URINE_OUTPUT_DIR :', URINE_OUTPUT_DIR)
    print('duckdb         :', duckdb.__version__)
    con = duckdb.connect(database=':memory:')
    LANDMARK_PATH = require_file(COHORT_DIR / 'landmark_population.parquet')
    BASE_OLD_PATH = require_file(URINE_OUTPUT_DIR / 'urine_output_base.parquet')
    OE_PATH = require_file(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    CE_PATH = require_file(MIMIC_ROOT / 'icu' / 'chartevents.csv.gz')
    PAT_PATH = require_file(MIMIC_ROOT / 'hosp' / 'patients.csv.gz')
    OMR_PATH = MIMIC_ROOT / 'hosp' / 'omr.csv.gz'
    con.execute(f"\nCREATE OR REPLACE VIEW landmark AS\nSELECT subject_id, hadm_id, stay_id, admittime, intime, landmark_time\nFROM read_parquet('{qpath(LANDMARK_PATH)}')\n")
    con.execute(f"\nCREATE OR REPLACE VIEW outputevents AS\nSELECT *\nFROM read_csv_auto('{qpath(OE_PATH)}', header=true, sample_size=200000)\n")
    con.execute(f"\nCREATE OR REPLACE VIEW chartevents AS\nSELECT *\nFROM read_csv_auto('{qpath(CE_PATH)}', header=true, sample_size=200000)\n")
    con.execute(f"\nCREATE OR REPLACE VIEW patients AS\nSELECT *\nFROM read_csv_auto('{qpath(PAT_PATH)}', header=true, sample_size=200000)\n")
    HAS_OMR = OMR_PATH.exists()
    if HAS_OMR:
        con.execute(f"\n    CREATE OR REPLACE VIEW omr AS\n    SELECT *\n    FROM read_csv_auto('{qpath(OMR_PATH)}', header=true, sample_size=200000)\n    ")
    else:
        print('WARNING: omr.csv.gz not found.')
    base_old = pd.read_parquet(BASE_OLD_PATH)
    assert_unique(base_old, 'stay_id', '04_uo_v6_base')
    assert_outcome_blind(base_old, '04_uo_v6_base')
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    FULL_EPS = 1e-06
    COMPARTMENT_CASE = "\nCASE\n  WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n  WHEN oe.itemid = 226584 THEN 'ILEAL_CONDUIT'\n  WHEN oe.itemid = 226564 THEN 'NEPHROSTOMY_R'\n  WHEN oe.itemid = 226565 THEN 'NEPHROSTOMY_L'\n  WHEN oe.itemid = 226557 THEN 'URETERAL_STENT_R'\n  WHEN oe.itemid = 226558 THEN 'URETERAL_STENT_L'\n  ELSE NULL\nEND\n"
    VARIANTS = [{'variant_id': 'event_nocap_full', 'variant_order': 0, 'max_gap_hours': None}, {'variant_id': 'event_gap2_full', 'variant_order': 1, 'max_gap_hours': 2.0}, {'variant_id': 'event_gap4_full', 'variant_order': 2, 'max_gap_hours': 4.0}, {'variant_id': 'event_gap6_full', 'variant_order': 3, 'max_gap_hours': 6.0}]
    con.execute('\nCREATE OR REPLACE TEMP TABLE all_icu_height_events AS\nSELECT\n    subject_id,\n    stay_id,\n    charttime,\n    storetime,\n    CASE\n      WHEN itemid=226730 THEN valuenum\n      WHEN itemid=226707 THEN valuenum*2.54\n    END AS height_cm\nFROM chartevents\nWHERE itemid IN (226730,226707)\n  AND valuenum IS NOT NULL\n  AND CASE\n        WHEN itemid=226730 THEN valuenum\n        WHEN itemid=226707 THEN valuenum*2.54\n      END > 120\n  AND CASE\n        WHEN itemid=226730 THEN valuenum\n        WHEN itemid=226707 THEN valuenum*2.54\n      END < 230\n  AND storetime IS NOT NULL\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE current_icu_height AS\nSELECT\n    l.stay_id,\n    h.height_cm,\n    h.charttime AS height_time_used,\n    h.storetime AS height_storetime_used,\n    'CURRENT_ICU_CHARTEVENTS' AS height_source\nFROM landmark l\nJOIN all_icu_height_events h\n  ON h.stay_id=l.stay_id\nWHERE h.charttime <= l.landmark_time\n  AND h.storetime <= l.landmark_time\nQUALIFY ROW_NUMBER() OVER(\n    PARTITION BY l.stay_id\n    ORDER BY h.charttime ASC, h.storetime ASC\n)=1\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE prior_icu_height AS\nSELECT\n    l.stay_id,\n    h.height_cm,\n    h.charttime AS historical_height_time,\n    h.storetime AS historical_storetime,\n    'PRIOR_ICU_CHARTEVENTS' AS height_source\nFROM landmark l\nJOIN all_icu_height_events h\n  ON h.subject_id=l.subject_id\nWHERE h.charttime < l.admittime\n  AND h.storetime <= l.landmark_time\nQUALIFY ROW_NUMBER() OVER(\n    PARTITION BY l.stay_id\n    ORDER BY h.charttime DESC, h.storetime DESC\n)=1\n")
    if HAS_OMR:
        con.execute("\n    CREATE OR REPLACE TEMP TABLE prior_omr_height AS\n    SELECT\n        l.stay_id,\n        TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 AS height_cm,\n        CAST(o.chartdate AS TIMESTAMP) AS historical_height_time,\n        CAST(NULL AS TIMESTAMP) AS historical_storetime,\n        'PRIOR_OMR' AS height_source\n    FROM landmark l\n    JOIN omr o USING(subject_id)\n    WHERE o.result_name IN ('Height (Inches)','Height')\n      AND o.chartdate < CAST(l.admittime AS DATE)\n      AND TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 > 120\n      AND TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 < 230\n    QUALIFY ROW_NUMBER() OVER(\n        PARTITION BY l.stay_id\n        ORDER BY o.chartdate DESC, o.seq_num DESC\n    )=1\n    ")
    else:
        con.execute('\n    CREATE OR REPLACE TEMP TABLE prior_omr_height AS\n    SELECT\n        CAST(NULL AS BIGINT) AS stay_id,\n        CAST(NULL AS DOUBLE) AS height_cm,\n        CAST(NULL AS TIMESTAMP) AS historical_height_time,\n        CAST(NULL AS TIMESTAMP) AS historical_storetime,\n        CAST(NULL AS VARCHAR) AS height_source\n    WHERE FALSE\n    ')
    con.execute('\nCREATE OR REPLACE TEMP TABLE historical_height_options AS\nSELECT * FROM prior_icu_height\nUNION ALL\nSELECT * FROM prior_omr_height\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE best_historical_height AS\nSELECT *\nFROM historical_height_options\nQUALIFY ROW_NUMBER() OVER(\n    PARTITION BY stay_id\n    ORDER BY historical_height_time DESC NULLS LAST,\n             CASE WHEN height_source='PRIOR_ICU_CHARTEVENTS' THEN 0 ELSE 1 END\n)=1\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE selected_height AS\nSELECT\n    l.stay_id,\n    COALESCE(c.height_cm,h.height_cm) AS height_cm,\n    CASE\n      WHEN c.height_cm IS NOT NULL THEN c.height_source\n      ELSE h.height_source\n    END AS height_source,\n    CASE\n      WHEN c.height_cm IS NOT NULL THEN c.height_time_used\n      ELSE h.historical_height_time\n    END AS height_time_used,\n    c.height_storetime_used,\n    p.gender,\n    CASE\n      WHEN COALESCE(c.height_cm,h.height_cm) IS NULL THEN NULL\n      WHEN UPPER(p.gender)='M'\n        THEN 50.0 + 2.3*((COALESCE(c.height_cm,h.height_cm)/2.54)-60.0)\n      WHEN UPPER(p.gender)='F'\n        THEN 45.5 + 2.3*((COALESCE(c.height_cm,h.height_cm)/2.54)-60.0)\n      ELSE NULL\n    END AS ibw_kg\nFROM landmark l\nLEFT JOIN current_icu_height c USING(stay_id)\nLEFT JOIN best_historical_height h USING(stay_id)\nLEFT JOIN patients p USING(subject_id)\n")
    selected_height = con.execute('SELECT * FROM selected_height').df()
    assert_unique(selected_height, 'stay_id', 'selected_height')
    height_qc = pd.DataFrame([{'n_landmark': len(selected_height), 'height_available': int(selected_height.height_cm.notna().sum()), 'ibw_available_positive': int((selected_height.ibw_kg > 0).sum()), 'current_icu_height_n': int(selected_height.height_source.eq('CURRENT_ICU_CHARTEVENTS').sum()), 'prior_icu_height_n': int(selected_height.height_source.eq('PRIOR_ICU_CHARTEVENTS').sum()), 'prior_omr_height_n': int(selected_height.height_source.eq('PRIOR_OMR').sum()), 'height_lt_140cm_n': int((selected_height.height_cm < 140).sum()), 'ibw_lt_40kg_n': int((selected_height.ibw_kg < 40).sum()), 'height_min_cm': float(selected_height.height_cm.min()), 'height_max_cm': float(selected_height.height_cm.max()), 'ibw_min_kg': float(selected_height.ibw_kg.min()), 'ibw_max_kg': float(selected_height.ibw_kg.max())}])
    display_table(height_qc)
    drop_cols = [c for c in ['height_cm', 'height_source', 'height_time_used', 'height_storetime_used', 'height_chartdate', 'gender', 'ibw_kg'] if c in base_old.columns]
    base = base_old.drop(columns=drop_cols).merge(selected_height, on='stay_id', how='left', validate='1:1')
    assert_unique(base, 'stay_id', 'validated base')
    assert_outcome_blind(base, 'validated base')
    con.execute(f'\nCREATE OR REPLACE TEMP TABLE uo_events AS\nSELECT\n    l.stay_id,\n    {COMPARTMENT_CASE} AS compartment,\n    oe.charttime,\n    MAX(oe.storetime) AS storetime,\n    SUM(oe.value) AS volume_ml\nFROM landmark l\nJOIN outputevents oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL\n  AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime >= l.intime\n  AND oe.charttime <= l.landmark_time\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime <= l.landmark_time\nGROUP BY l.stay_id, compartment, oe.charttime\nHAVING compartment IS NOT NULL\n')
    con.execute('\nCREATE OR REPLACE TEMP TABLE uo_interval_raw AS\nSELECT\n    stay_id,\n    compartment,\n    charttime AS interval_end,\n    storetime,\n    volume_ml,\n    LAG(charttime) OVER(\n        PARTITION BY stay_id, compartment\n        ORDER BY charttime\n    ) AS interval_start\nFROM uo_events\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE uo_intervals AS\nSELECT\n    r.stay_id,\n    r.compartment,\n    DATE_DIFF('second', l.intime, r.interval_start)/3600.0 AS start_hour,\n    DATE_DIFF('second', l.intime, r.interval_end)/3600.0 AS end_hour,\n    r.volume_ml,\n    DATE_DIFF('second', r.interval_start, r.interval_end)/3600.0 AS duration_hours\nFROM uo_interval_raw r\nJOIN landmark l USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end > r.interval_start\n  AND r.interval_start >= l.intime\n  AND r.interval_end <= l.landmark_time\n")
    intervals = con.execute('\nSELECT stay_id, compartment, start_hour, end_hour, volume_ml, duration_hours\nFROM uo_intervals\nORDER BY stay_id, start_hour, end_hour\n').df()
    source_qc = con.execute('\nSELECT\n    (SELECT COUNT(*) FROM uo_events) AS uo_event_rows,\n    (SELECT COUNT(DISTINCT stay_id) FROM uo_events) AS stays_with_uo_event,\n    (SELECT COUNT(*) FROM uo_intervals) AS reconstructed_intervals,\n    (SELECT COUNT(DISTINCT stay_id) FROM uo_intervals) AS stays_with_reconstructed_interval,\n    (SELECT MAX(duration_hours) FROM uo_intervals) AS max_interval_hours\n').df()
    source_item_qc = con.execute(f'\nSELECT oe.itemid, COUNT(*) AS rows, COUNT(DISTINCT oe.stay_id) AS stays,\n       SUM(CASE WHEN oe.value < 0 THEN 1 ELSE 0 END) AS negative_rows,\n       SUM(CASE WHEN oe.value > 5000 THEN 1 ELSE 0 END) AS over_5000_rows\nFROM landmark l JOIN outputevents oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.charttime >= l.intime AND oe.charttime <= l.landmark_time\n  AND oe.storetime IS NOT NULL AND oe.storetime <= l.landmark_time\nGROUP BY oe.itemid ORDER BY oe.itemid\n').df()
    uo_timing = con.execute('\nSELECT stay_id, MAX(charttime) AS max_uo_charttime, MAX(storetime) AS max_uo_storetime\nFROM uo_events GROUP BY stay_id\n').df()
    base = base.merge(uo_timing, on='stay_id', how='left', validate='1:1')
    multi_compartment_qc = intervals.groupby('stay_id')['compartment'].nunique().value_counts().sort_index().rename_axis('n_compartments').reset_index(name='n_stays')
    display_table(source_qc)
    display_table(source_item_qc)
    display_table(multi_compartment_qc)
    def _segments_from_intervals(g: pd.DataFrame, max_gap_hours=None):
        starts = g['start_hour'].to_numpy(dtype=float)
        ends = g['end_hour'].to_numpy(dtype=float)
        vols = g['volume_ml'].to_numpy(dtype=float)
        dur = ends - starts
        keep = np.isfinite(starts) & np.isfinite(ends) & np.isfinite(vols) & (dur > 0)
        if max_gap_hours is not None:
            keep &= dur <= float(max_gap_hours) + 1e-12
        starts, ends, vols, dur = (starts[keep], ends[keep], vols[keep], dur[keep])
        if len(starts) == 0:
            return None
        rates = vols / dur
        event_times = np.concatenate([starts, ends])
        rate_delta = np.concatenate([rates, -rates])
        count_delta = np.concatenate([np.ones(len(starts), dtype=float), -np.ones(len(starts), dtype=float)])
        uniq, inv = np.unique(event_times, return_inverse=True)
        d_rate = np.zeros(len(uniq), dtype=float)
        d_count = np.zeros(len(uniq), dtype=float)
        np.add.at(d_rate, inv, rate_delta)
        np.add.at(d_count, inv, count_delta)
        if len(uniq) < 2:
            return None
        seg_start = uniq[:-1]
        seg_end = uniq[1:]
        seg_rate = np.cumsum(d_rate)[:-1]
        seg_count = np.cumsum(d_count)[:-1]
        seg_cov = (seg_count > 0).astype(float)
        good = seg_end > seg_start
        seg_start = seg_start[good]
        seg_end = seg_end[good]
        seg_rate = seg_rate[good]
        seg_cov = seg_cov[good]
        if len(seg_start) == 0:
            return None
        boundaries = np.unique(np.concatenate([seg_start, seg_end]))
        return (seg_start, seg_end, seg_rate, seg_cov, boundaries, len(starts))

    def _cumulative_at(t, seg_start, seg_end, seg_rate, seg_cov):
        t = np.asarray(t, dtype=float)
        durations = seg_end - seg_start
        prefix_vol = np.concatenate([[0.0], np.cumsum(seg_rate * durations)])
        prefix_cov = np.concatenate([[0.0], np.cumsum(seg_cov * durations)])
        idx = np.searchsorted(seg_end, t, side='right')
        vol = prefix_vol[idx].astype(float, copy=True)
        cov = prefix_cov[idx].astype(float, copy=True)
        valid = idx < len(seg_start)
        j = np.where(valid, idx, 0)
        inside = valid & (t > seg_start[j]) & (t < seg_end[j])
        if inside.any():
            jj = j[inside]
            delta = t[inside] - seg_start[jj]
            vol[inside] += seg_rate[jj] * delta
            cov[inside] += seg_cov[jj] * delta
        return (vol, cov)

    def _critical_times(window_h, boundaries):
        ts = np.concatenate([boundaries, boundaries + window_h, np.asarray([window_h, 24.0], dtype=float)])
        ts = np.unique(np.round(ts, 12))
        ts = ts[(ts >= window_h - 1e-09) & (ts <= 24.0 + 1e-09)]
        return ts

    def _window_values(window_h, segments):
        seg_start, seg_end, seg_rate, seg_cov, boundaries, _ = segments
        ts = _critical_times(window_h, boundaries)
        if len(ts) == 0:
            return (ts, np.array([]), np.array([]))
        v1, c1 = _cumulative_at(ts, seg_start, seg_end, seg_rate, seg_cov)
        v0, c0 = _cumulative_at(ts - window_h, seg_start, seg_end, seg_rate, seg_cov)
        return (ts, v1 - v0, c1 - c0)

    def _stage_for_weight(segments, weight_kg):
        if segments is None or weight_kg is None or (not np.isfinite(weight_kg)) or (weight_kg <= 0):
            return np.nan
        positive_stage = 0
        _, vol24, cov24 = _window_values(24.0, segments)
        if len(vol24):
            full24 = cov24 >= 24.0 - FULL_EPS
            if np.any(full24 & (vol24 / (weight_kg * 24.0) < 0.3)):
                positive_stage = max(positive_stage, 3)
        _, vol12, cov12 = _window_values(12.0, segments)
        if len(vol12):
            full12 = cov12 >= 12.0 - FULL_EPS
            if np.any(full12 & (np.abs(vol12) <= 1e-09)):
                positive_stage = max(positive_stage, 3)
            if np.any(full12 & (vol12 / (weight_kg * 12.0) < 0.5)):
                positive_stage = max(positive_stage, 2)
        _, vol6, cov6 = _window_values(6.0, segments)
        if len(vol6):
            full6 = cov6 >= 6.0 - FULL_EPS
            if np.any(full6 & (vol6 / (weight_kg * 6.0) < 0.5)):
                positive_stage = max(positive_stage, 1)
        return float(positive_stage) if positive_stage >= 1 else np.nan

    def evaluate_stay_eventtime(g, admission_weight_kg, ibw_kg, max_gap_hours=None):
        segments = _segments_from_intervals(g, max_gap_hours=max_gap_hours)
        if segments is None:
            return {'positive_stage_abw': np.nan, 'positive_stage_ibw': np.nan, 'uo_positive_abw': 0, 'uo_positive_ibw': 0, 'absence_evaluable_abw': 0, 'absence_evaluable_ibw': 0, 'any_full6_window': 0, 'any_full12_window': 0, 'any_full24_window': 0, 'max_coverage6_hours': 0.0, 'max_coverage12_hours': 0.0, 'max_coverage24_hours': 0.0, 'landmark_coverage_hours': 0.0, 'n_valid_intervals': 0}
        seg_start, seg_end, seg_rate, seg_cov, boundaries, n_valid = segments
        coverage_max = {}
        for W in (6.0, 12.0, 24.0):
            _, _, cov = _window_values(W, segments)
            coverage_max[W] = float(np.max(cov)) if len(cov) else 0.0
        _, cov_lm = _cumulative_at(np.asarray([0.0, 24.0]), seg_start, seg_end, seg_rate, seg_cov)
        landmark_cov = float(cov_lm[1] - cov_lm[0])
        stage_abw = _stage_for_weight(segments, admission_weight_kg)
        stage_ibw = _stage_for_weight(segments, ibw_kg)
        abw_ok = admission_weight_kg is not None and np.isfinite(admission_weight_kg) and (admission_weight_kg > 0)
        ibw_ok = ibw_kg is not None and np.isfinite(ibw_kg) and (ibw_kg > 0)
        return {'positive_stage_abw': stage_abw, 'positive_stage_ibw': stage_ibw, 'uo_positive_abw': int(np.isfinite(stage_abw) and stage_abw >= 1), 'uo_positive_ibw': int(np.isfinite(stage_ibw) and stage_ibw >= 1), 'absence_evaluable_abw': int(abw_ok and landmark_cov >= 24.0 - FULL_EPS), 'absence_evaluable_ibw': int(ibw_ok and landmark_cov >= 24.0 - FULL_EPS), 'any_full6_window': int(coverage_max[6.0] >= 6.0 - FULL_EPS), 'any_full12_window': int(coverage_max[12.0] >= 12.0 - FULL_EPS), 'any_full24_window': int(coverage_max[24.0] >= 24.0 - FULL_EPS), 'max_coverage6_hours': coverage_max[6.0], 'max_coverage12_hours': coverage_max[12.0], 'max_coverage24_hours': coverage_max[24.0], 'landmark_coverage_hours': landmark_cov, 'n_valid_intervals': int(n_valid)}

    def _synthetic(rows):
        return pd.DataFrame(rows, columns=['start_hour', 'end_hour', 'volume_ml'])
    assert int(_stage_for_weight(_segments_from_intervals(_synthetic([(0, 6, 240)])), 100.0)) == 1
    assert int(_stage_for_weight(_segments_from_intervals(_synthetic([(0, 12, 240)])), 100.0)) == 2
    assert int(_stage_for_weight(_segments_from_intervals(_synthetic([(0, 12, 0)])), 100.0)) == 3
    assert int(_stage_for_weight(_segments_from_intervals(_synthetic([(0, 24, 480)])), 100.0)) == 3
    assert np.isnan(_stage_for_weight(_segments_from_intervals(_synthetic([(0, 6, 300)])), 100.0))
    critical_test = _synthetic([(0, 4, 0), (4, 10, 600)])
    assert int(_stage_for_weight(_segments_from_intervals(critical_test), 100.0)) == 1
    seven_hour = _synthetic([(0, 7, 280)])
    assert int(_stage_for_weight(_segments_from_intervals(seven_hour), 100.0)) == 1
    assert _segments_from_intervals(seven_hour, max_gap_hours=6.0) is None
    print('STATUS: exhaustive event-time engine unit tests — PASS')
    weights = base[['stay_id', 'admission_weight_kg', 'ibw_kg']].set_index('stay_id')
    grouped = {int(sid): g[['start_hour', 'end_hour', 'volume_ml']].copy() for sid, g in intervals.groupby('stay_id', sort=False)}
    records = []
    stays_with_intervals = list(grouped.keys())
    for idx, sid in enumerate(stays_with_intervals, start=1):
        g = grouped[sid]
        w_abw = weights.at[sid, 'admission_weight_kg'] if sid in weights.index else np.nan
        w_ibw = weights.at[sid, 'ibw_kg'] if sid in weights.index else np.nan
        for meta in VARIANTS:
            res = evaluate_stay_eventtime(g, w_abw, w_ibw, max_gap_hours=meta['max_gap_hours'])
            records.append({'stay_id': sid, 'variant_id': meta['variant_id'], 'variant_order': meta['variant_order'], 'max_gap_hours': meta['max_gap_hours'], **res})
        if idx % 5000 == 0 or idx == len(stays_with_intervals):
            print(f'Processed {idx:,}/{len(stays_with_intervals):,}')
    result = pd.DataFrame.from_records(records)
    skeleton = base[['stay_id']].assign(_k=1).merge(pd.DataFrame(VARIANTS).assign(_k=1), on='_k').drop(columns='_k')
    cand = skeleton.merge(result, on=['stay_id', 'variant_id', 'variant_order', 'max_gap_hours'], how='left', validate='1:1')
    zero_cols = ['uo_positive_abw', 'uo_positive_ibw', 'absence_evaluable_abw', 'absence_evaluable_ibw', 'any_full6_window', 'any_full12_window', 'any_full24_window', 'max_coverage6_hours', 'max_coverage12_hours', 'max_coverage24_hours', 'landmark_coverage_hours', 'n_valid_intervals']
    for c in zero_cols:
        cand[c] = cand[c].fillna(0)
    assert len(cand) == len(base) * len(VARIANTS)
    assert not cand.duplicated(['stay_id', 'variant_id']).any()
    summary = cand.groupby(['variant_order', 'variant_id'], as_index=False).agg(n=('stay_id', 'size'), abw_positive=('uo_positive_abw', 'sum'), ibw_positive=('uo_positive_ibw', 'sum'), abw_kdigo_stage_1_n=('positive_stage_abw', lambda x: int((x == 1).sum())), abw_kdigo_stage_2_n=('positive_stage_abw', lambda x: int((x == 2).sum())), abw_kdigo_stage_3_n=('positive_stage_abw', lambda x: int((x == 3).sum())), ibw_kdigo_stage_1_n=('positive_stage_ibw', lambda x: int((x == 1).sum())), ibw_kdigo_stage_2_n=('positive_stage_ibw', lambda x: int((x == 2).sum())), ibw_kdigo_stage_3_n=('positive_stage_ibw', lambda x: int((x == 3).sum())), any_full6_window=('any_full6_window', 'sum'), any_full12_window=('any_full12_window', 'sum'), any_full24_window=('any_full24_window', 'sum'), abw_absence_evaluable=('absence_evaluable_abw', 'sum'), ibw_absence_evaluable=('absence_evaluable_ibw', 'sum'), median_landmark_coverage_hours=('landmark_coverage_hours', 'median')).sort_values('variant_order').reset_index(drop=True)
    display_table(summary)
    P_BASE = URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet'
    P_CAND = URINE_OUTPUT_DIR / 'urine_output_event_time_variants.parquet'
    base.to_parquet(P_BASE, index=False)
    cand.to_parquet(P_CAND, index=False)
    height_qc.to_csv(PHENOTYPE_QC_DIR / 'height_hierarchy_summary.csv', index=False)
    source_qc.to_csv(PHENOTYPE_QC_DIR / 'urine_output_source_summary.csv', index=False)
    source_item_qc.to_csv(PHENOTYPE_QC_DIR / 'urine_output_source_by_item.csv', index=False)
    multi_compartment_qc.to_csv(PHENOTYPE_QC_DIR / 'urine_output_multi_compartment_summary.csv', index=False)
    summary.to_csv(PHENOTYPE_QC_DIR / 'urine_output_event_time_summary.csv', index=False)
    source_definition = {'primary_direct_itemids': list(UO_PRIMARY), 'primary_direct_item_labels': {'226559': 'Foley', '226560': 'Void', '226561': 'Condom Cath', '226584': 'Ileoconduit', '226563': 'Suprapubic', '226564': 'R Nephrostomy', '226565': 'L Nephrostomy', '226567': 'Straight Cath', '226557': 'R Ureteral Stent', '226558': 'L Ureteral Stent'}, 'gu_irrigation_itemids': [227488, 227489], 'gu_irrigation_in_primary': False, 'gu_irrigation_role': 'MIT-LCP-like net comparator / sensitivity only', 'engine': 'event-time exhaustive critical-time integration', 'first_observation': 'residual-only within compartment', 'primary_variant': 'event_gap6_full', 'primary_weight_selected_later': 'IBW'}
    (PHENOTYPE_QC_DIR / 'urine_output_source_definition.json').write_text(json.dumps(source_definition, indent=2), encoding='utf-8')
    print('Saved:', P_BASE)
    print('Saved:', P_CAND)
    print('URINE OUTPUT EVENT-TIME RECONSTRUCTION: PASS')


if __name__ == "__main__":
    main()
