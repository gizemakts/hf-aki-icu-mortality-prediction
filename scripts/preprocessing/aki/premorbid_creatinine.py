#!/usr/bin/env python3
"""07 Premorbid Creatinine.

Add premorbid outpatient creatinine.

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
    from datetime import datetime, timezone
    import json, hashlib
    import numpy as np
    import pandas as pd
    import duckdb
    EXPECTED_REFERENCE_N = 3251
    PRIMARY_UO = 'event_gap6_full'
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT))).resolve()
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', str(_RAW_DATA_ROOT))).resolve()
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f'MIMIC root not found: {MIMIC_ROOT}')
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    CREATININE_DIR = PHENOTYPE_DIR / 'creatinine'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    RRT_DIR = PHENOTYPE_DIR / 'rrt'
    COMBINED_DIR = PHENOTYPE_DIR / 'combined'
    HISTORICAL_DIR = PHENOTYPE_DIR / 'historical_creatinine'
    OUT = HISTORICAL_DIR / 'premorbid_proxy'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'historical_creatinine' / 'premorbid_proxy'
    for directory in (OUT, QC):
        directory.mkdir(parents=True, exist_ok=True)

    def req(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def sha256_file(p, block=1024 * 1024):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while True:
                b = f.read(block)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: null {key}')
        if df[key].duplicated().any():
            raise AssertionError(f'{label}: duplicate {key}')

    def outcome_like_columns(df):
        toks = ('expire', 'mortality', 'death', 'outcome')
        return [c for c in df.columns if any((t in c.lower() for t in toks))]

    def assert_outcome_blind(df, label):
        bad = outcome_like_columns(df)
        if bad:
            raise AssertionError(f'{label} contains prohibited outcome-like fields: {bad}')

    def wcsv(df, name):
        p = QC / name
        df.to_csv(p, index=False)
        return p

    def wjson(obj, name):
        p = QC / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p

    def membership_compare(reference, alternative, label):
        c = reference[['subject_id', 'stay_id']].copy()
        a = alternative[['subject_id', 'stay_id']].copy()
        assert_unique(c, 'subject_id', 'reference subject index')
        assert_unique(a, 'subject_id', label)
        cm = dict(zip(c.subject_id.astype(int), c.stay_id.astype(int)))
        am = dict(zip(a.subject_id.astype(int), a.stay_id.astype(int)))
        cs, aset = (set(cm), set(am))
        common = cs & aset
        replaced = sum((cm[s] != am[s] for s in common))
        return {'variant': label, 'subjects': len(a), 'subject_added_vs_reference': len(aset - cs), 'subject_dropped_vs_reference': len(cs - aset), 'common_subjects': len(common), 'same_subject_replaced_stay': int(replaced), 'same_subject_same_stay': int(len(common) - replaced), 'stay_symmetric_difference': len(set(cm.values()) ^ set(am.values()))}
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT  :', MIMIC_ROOT)
    print('OUTPUT      :', OUT)
    print('pandas      :', pd.__version__)
    print('duckdb      :', duckdb.__version__)
    P_LM = req(COHORT_DIR / 'landmark_population.parquet')
    P_PH = req(COHORT_DIR / 'heart_failure_krt_phenotypes.parquet')
    P_CR_REFERENCE = req(CREATININE_DIR / 'kdigo_creatinine.parquet')
    P_RRT = req(RRT_DIR / 'active_rrt.parquet')
    P_BASE = req(URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet')
    P_COMB = req(COMBINED_DIR / 'combined_kdigo_variants.parquet')
    P_REFERENCE_INDEX = req(COHORT_DIR / 'reference_cohort_preoutcome.parquet')
    reference_sha = sha256_file(P_REFERENCE_INDEX)
    lm = pd.read_parquet(P_LM, columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time'])
    ph = pd.read_parquet(P_PH)
    rolling_cr = pd.read_parquet(P_CR_REFERENCE)
    rrt = pd.read_parquet(P_RRT)
    base = pd.read_parquet(P_BASE)
    comb = pd.read_parquet(P_COMB)
    reference_idx = pd.read_parquet(P_REFERENCE_INDEX)
    for label, df in [('landmark', lm), ('phenotype', ph), ('creatinine component', rolling_cr), ('RRT', rrt), ('UO base', base), ('combined', comb), ('reference cohort', reference_idx)]:
        assert_outcome_blind(df, label)
    for df, label in [(lm, 'landmark'), (ph, 'phenotype'), (rolling_cr, 'creatinine'), (rrt, 'RRT'), (base, 'UO base'), (reference_idx, 'reference cohort')]:
        assert_unique(df, 'stay_id', label)
    if comb.duplicated(['stay_id', 'variant_id']).any():
        raise AssertionError('Combined table duplicates stay_id x variant_id.')
    if len(reference_idx) != EXPECTED_REFERENCE_N or reference_idx.subject_id.nunique() != EXPECTED_REFERENCE_N:
        raise AssertionError('Reference cohort N mismatch.')
    krt_col = 'chronic_krt_definite'
    need_ph = ['stay_id', 'prior_documented_hf', krt_col]
    miss = [c for c in need_ph if c not in ph.columns]
    if miss:
        raise KeyError(f'Phenotype missing: {miss}')
    primary = comb.loc[comb.variant_id.eq(PRIMARY_UO)].copy()
    assert_unique(primary, 'stay_id', 'primary combined alternative')
    if len(primary) != len(lm):
        raise AssertionError('Primary urine-output alternative does not span all landmark stays.')
    target = lm.merge(ph[need_ph], on='stay_id', how='left', validate='1:1')
    target = target[target.prior_documented_hf.eq(1) & target[krt_col].fillna(0).eq(0)].copy()
    assert_unique(target, 'stay_id', 'HF/non-chronic-KRT target')
    reference_reconstruction = target.merge(primary[['stay_id', 'combined_status_ibw_anyrrt']], on='stay_id', how='left', validate='1:1')
    reference_qualifying = reference_reconstruction.loc[reference_reconstruction.combined_status_ibw_anyrrt.eq('positive')].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
    reference_selection = reference_qualifying.drop_duplicates('subject_id', keep='first').copy()
    if len(reference_selection) != EXPECTED_REFERENCE_N:
        raise AssertionError(f'reference reconstruction N={len(reference_selection)}')
    if set(reference_selection.stay_id.astype(int)) != set(reference_idx.stay_id.astype(int)):
        raise AssertionError('reference reconstruction membership mismatch.')
    guards = pd.DataFrame([{'reference_cohort_sha256': reference_sha, 'reference_cohort_n': len(reference_idx), 'landmark_stays': len(lm), 'target_stays': len(target), 'reference_qualifying_stays': len(reference_qualifying), 'reference_exact_membership_reconstruction': 1, 'outcome_blind': 1}])
    display_table(guards)
    wcsv(guards, 'premorbid_proxy_hard_guards.csv')
    print('REFERENCE COHORT RECONSTRUCTION: PASS')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    DLAB = req(MIMIC_ROOT / 'hosp' / 'd_labitems.csv.gz')
    ED = MIMIC_ROOT / 'ed' / 'edstays.csv.gz'
    dlab = pd.read_csv(DLAB, compression='gzip', usecols=['itemid', 'label'])
    lab_match = dlab.loc[dlab.itemid.eq(50912), 'label']
    if len(lab_match) != 1 or 'creatin' not in str(lab_match.iloc[0]).lower():
        raise AssertionError(f'Unexpected itemid 50912 mapping: {lab_match.tolist()}')
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('target_df', target[['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time']].copy())
    con.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM target_df')
    con.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT\n    subject_id,\n    hadm_id,\n    charttime,\n    storetime,\n    valuenum AS creatinine\nFROM read_csv_auto(\n    '{qpath(LAB)}',\n    header=true,\n    sample_size=200000\n)\nWHERE itemid=50912\n  AND valuenum>0\n  AND valuenum<=150\n  AND charttime IS NOT NULL\n  AND storetime IS NOT NULL\n")
    con.execute(f"\nCREATE VIEW admissions AS\nSELECT subject_id,hadm_id,admittime,dischtime\nFROM read_csv_auto(\n    '{qpath(ADM)}',\n    header=true,\n    sample_size=200000\n)\n")
    HAS_ED = ED.is_file()
    if HAS_ED:
        con.execute(f"\n    CREATE VIEW edstays AS\n    SELECT subject_id,intime,outtime\n    FROM read_csv_auto(\n        '{qpath(ED)}',\n        header=true,\n        sample_size=200000\n    )\n    ")
    else:
        con.execute('\n    CREATE TEMP TABLE edstays AS\n    SELECT\n        CAST(NULL AS BIGINT) AS subject_id,\n        CAST(NULL AS TIMESTAMP) AS intime,\n        CAST(NULL AS TIMESTAMP) AS outtime\n    WHERE FALSE\n    ')
    con.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT\n    t.stay_id,\n    t.subject_id,\n    t.admittime,\n    t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS storetime,\n    AVG(c.creatinine) AS creatinine\nFROM target_lm t\nJOIN creat_all c\n  ON c.subject_id=t.subject_id\nWHERE c.charttime >= t.admittime - INTERVAL '365 days'\n  AND c.charttime <  t.admittime - INTERVAL '7 days'\n  AND c.storetime <= t.landmark_time\nGROUP BY\n    t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    ed_clause = '\nEXISTS (\n    SELECT 1\n    FROM edstays e\n    WHERE e.subject_id=h.subject_id\n      AND h.charttime>=e.intime\n      AND h.charttime<=e.outtime\n)\n' if HAS_ED else 'FALSE'
    con.execute(f'\nCREATE TEMP TABLE hist_classified AS\nSELECT\n    h.*,\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM admissions a\n        WHERE a.subject_id=h.subject_id\n          AND a.admittime IS NOT NULL\n          AND a.dischtime IS NOT NULL\n          AND h.charttime>=a.admittime\n          AND h.charttime<=a.dischtime\n    ) THEN 1 ELSE 0 END AS overlaps_known_inpatient,\n    CASE WHEN {ed_clause} THEN 1 ELSE 0 END AS overlaps_known_ed\nFROM hist_same_time h\n')
    con.execute('\nCREATE TEMP TABLE hist_outpt AS\nSELECT *\nFROM hist_classified\nWHERE overlaps_known_inpatient=0\n  AND overlaps_known_ed=0\n')
    hist = con.execute('\nSELECT\n    t.stay_id,\n    COUNT(h.charttime) AS hist_outpt_n,\n    MEDIAN(h.creatinine) AS hist_outpt_median,\n    AVG(h.creatinine) AS hist_outpt_mean,\n    ARG_MAX(h.creatinine,h.charttime) AS hist_outpt_most_recent,\n    MIN(h.charttime) AS hist_outpt_first_time,\n    MAX(h.charttime) AS hist_outpt_last_time\nFROM target_lm t\nLEFT JOIN hist_outpt h USING(stay_id)\nGROUP BY t.stay_id\n').df()
    assert_unique(hist, 'stay_id', 'historical outpatient proxy')
    hist_avail = pd.DataFrame([{'target_stays': len(hist), 'with_hist_outpatient_proxy': int(hist.hist_outpt_median.notna().sum()), 'availability_pct': float(100 * hist.hist_outpt_median.notna().mean()), 'single_measurement': int(hist.hist_outpt_n.eq(1).sum()), 'n_ge_2': int(hist.hist_outpt_n.ge(2).sum()), 'ed_module_used': int(HAS_ED)}])
    display_table(hist_avail)
    wcsv(hist_avail, '02_historical_baseline_availability.csv')
    wcsv(hist, '02_historical_baseline_by_target_stay.csv')
    con.execute("\nCREATE TEMP TABLE cr_same_time AS\nSELECT\n    t.stay_id,t.subject_id,t.hadm_id,t.admittime,t.intime,t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS cr_storetime,\n    AVG(c.creatinine) AS creatinine\nFROM target_lm t\nJOIN creat_all c\n  ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY\n    t.stay_id,t.subject_id,t.hadm_id,t.admittime,t.intime,t.landmark_time,c.charttime\n")
    con.execute("\nCREATE TEMP TABLE cr_refs AS\nSELECT\n    *,\n    MIN(creatinine) OVER(\n        PARTITION BY stay_id ORDER BY charttime\n        RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n        AND INTERVAL '1 microsecond' PRECEDING\n    ) AS low48,\n    MIN(creatinine) OVER(\n        PARTITION BY stay_id ORDER BY charttime\n        RANGE BETWEEN INTERVAL '7 days' PRECEDING\n        AND INTERVAL '1 microsecond' PRECEDING\n    ) AS low7d\nFROM cr_same_time\n")
    events = con.execute('\nSELECT\n    r.stay_id,\n    r.charttime,\n    r.cr_storetime,\n    r.creatinine,\n    r.low48,\n    r.low7d,\n    h.hist_outpt_n,\n    h.hist_outpt_median,\n    h.hist_outpt_mean,\n    h.hist_outpt_most_recent\nFROM cr_refs r\nLEFT JOIN (\n    SELECT\n        stay_id,\n        COUNT(*) AS hist_outpt_n,\n        MEDIAN(creatinine) AS hist_outpt_median,\n        AVG(creatinine) AS hist_outpt_mean,\n        ARG_MAX(creatinine,charttime) AS hist_outpt_most_recent\n    FROM hist_outpt\n    GROUP BY stay_id\n) h USING(stay_id)\nWHERE r.charttime>=r.intime\n  AND r.charttime<=r.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()

    def rolling_stage(df):
        cr = pd.to_numeric(df['creatinine'], errors='coerce')
        low48 = pd.to_numeric(df['low48'], errors='coerce')
        low7 = pd.to_numeric(df['low7d'], errors='coerce')
        has_ref = low48.notna() | low7.notna()
        delta03 = low48.notna() & cr.ge(low48 + 0.3)
        ratio15 = low7.notna() & cr.ge(1.5 * low7)
        ratio20 = low7.notna() & cr.ge(2.0 * low7)
        ratio30 = low7.notna() & cr.ge(3.0 * low7)
        out = np.select([ratio30, cr.ge(4.0) & (delta03 | ratio15), ratio20, delta03 | ratio15], [3.0, 3.0, 2.0, 1.0], default=0.0)
        out = pd.Series(out, index=df.index, dtype='float64')
        out.loc[~has_ref] = np.nan
        return out

    def historical_ratio_stage(df, baseline_col):
        cr = pd.to_numeric(df['creatinine'], errors='coerce')
        b = pd.to_numeric(df[baseline_col], errors='coerce')
        low48 = pd.to_numeric(df['low48'], errors='coerce')
        has_ref = b.notna()
        ratio15 = has_ref & cr.ge(1.5 * b)
        ratio20 = has_ref & cr.ge(2.0 * b)
        ratio30 = has_ref & cr.ge(3.0 * b)
        delta03 = low48.notna() & cr.ge(low48 + 0.3)
        out = np.select([ratio30, cr.ge(4.0) & (ratio15 | delta03), ratio20, ratio15], [3.0, 3.0, 2.0, 1.0], default=0.0)
        out = pd.Series(out, index=df.index, dtype='float64')
        out.loc[~has_ref] = np.nan
        return out
    events['kdigo_creatinine_stage_rolling_recomputed'] = rolling_stage(events)
    events['stage_hist_median'] = historical_ratio_stage(events, 'hist_outpt_median')
    events['stage_hist_mean'] = historical_ratio_stage(events, 'hist_outpt_mean')
    events['stage_hist_recent'] = historical_ratio_stage(events, 'hist_outpt_most_recent')
    stage_stay = events.groupby('stay_id', as_index=False)[['kdigo_creatinine_stage_rolling_recomputed', 'stage_hist_median', 'stage_hist_mean', 'stage_hist_recent']].max()
    premorbid_cr = target[['stay_id']].merge(stage_stay, on='stay_id', how='left', validate='1:1').merge(rolling_cr[['stay_id', 'kdigo_creatinine_stage', 'creatinine_evaluable']], on='stay_id', how='left', validate='1:1').merge(hist, on='stay_id', how='left', validate='1:1')
    same = premorbid_cr.kdigo_creatinine_stage_rolling_recomputed.fillna(-99).eq(premorbid_cr.kdigo_creatinine_stage.fillna(-99))
    mismatch = int((~same).sum())
    if mismatch:
        wcsv(premorbid_cr.loc[~same], 'rolling_creatinine_reconstruction_mismatch.csv')
        raise AssertionError(f'Rolling creatinine reproduction failed: mismatches={mismatch}')

    def rowmax(a, b):
        return pd.concat([pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')], axis=1).max(axis=1, skipna=True).where(pd.to_numeric(a, errors='coerce').notna() | pd.to_numeric(b, errors='coerce').notna())
    premorbid_cr['kdigo_creatinine_stage_rolling'] = premorbid_cr['kdigo_creatinine_stage']
    premorbid_cr['kdigo_creatinine_stage_premorbid_median'] = rowmax(premorbid_cr.kdigo_creatinine_stage_rolling, premorbid_cr.stage_hist_median)
    premorbid_cr['kdigo_creatinine_stage_premorbid_mean_sensitivity'] = rowmax(premorbid_cr.kdigo_creatinine_stage_rolling, premorbid_cr.stage_hist_mean)
    premorbid_cr['kdigo_creatinine_stage_premorbid_recent_sensitivity'] = rowmax(premorbid_cr.kdigo_creatinine_stage_rolling, premorbid_cr.stage_hist_recent)
    event_stays = set(events.stay_id.astype(int))
    hist_eval = premorbid_cr.hist_outpt_median.notna() & premorbid_cr.stay_id.astype(int).isin(event_stays)
    premorbid_cr['creatinine_evaluable_premorbid'] = (premorbid_cr.creatinine_evaluable.fillna(0).eq(1) | hist_eval).astype('int8')
    premorbid_cr['new_creatinine_positive_from_hist_median'] = (premorbid_cr.kdigo_creatinine_stage_rolling.fillna(0).lt(1) & premorbid_cr.kdigo_creatinine_stage_premorbid_median.fillna(0).ge(1)).astype('int8')
    demoted = premorbid_cr.kdigo_creatinine_stage_rolling.fillna(-1) > premorbid_cr.kdigo_creatinine_stage_premorbid_median.fillna(-1)
    if demoted.any():
        raise AssertionError('Hybrid median creatinine unexpectedly demoted a rolling stage.')
    cr_qc = pd.DataFrame([{'target_stays': len(premorbid_cr), 'rolling_creatinine_recompute_mismatch': mismatch, 'rolling_creatinine_positive': int(premorbid_cr.kdigo_creatinine_stage_rolling.fillna(0).ge(1).sum()), 'premorbid_hybrid_median_positive': int(premorbid_cr.kdigo_creatinine_stage_premorbid_median.fillna(0).ge(1).sum()), 'new_positive_from_hist_median': int(premorbid_cr.new_creatinine_positive_from_hist_median.sum()), 'premorbid_hybrid_mean_positive_sensitivity': int(premorbid_cr.kdigo_creatinine_stage_premorbid_mean_sensitivity.fillna(0).ge(1).sum()), 'premorbid_hybrid_recent_positive_sensitivity': int(premorbid_cr.kdigo_creatinine_stage_premorbid_recent_sensitivity.fillna(0).ge(1).sum()), 'rolling_positive_demoted': int(demoted.sum())}])
    display_table(cr_qc)
    wcsv(cr_qc, '03_creatinine_hybrid_summary.csv')
    wcsv(premorbid_cr, '03_creatinine_hybrid_by_target_stay.csv')
    print('CREATININE HYBRID HARD GUARDS: PASS')
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    con_uo = duckdb.connect(database=':memory:')
    con_uo.register('target_df', target[['stay_id', 'intime', 'landmark_time']].copy())
    con_uo.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM target_df')
    con_uo.execute(f"\nCREATE VIEW oe_reliability AS\nSELECT stay_id,itemid,charttime,storetime\nFROM read_csv_auto(\n    '{qpath(OE)}',\n    header=true,\n    sample_size=200000\n)\nWHERE itemid IN (226557,226558,227488,227489)\n")
    affected = con_uo.execute('\nSELECT\n    t.stay_id,\n    MAX(CASE WHEN oe.itemid IN (226557,226558) THEN 1 ELSE 0 END) AS has_ureteral_stent_by_landmark,\n    MAX(CASE WHEN oe.itemid IN (227488,227489) THEN 1 ELSE 0 END) AS has_gu_irrigation_by_landmark\nFROM target_lm t\nJOIN oe_reliability oe USING(stay_id)\nWHERE oe.charttime>=t.intime\n  AND oe.charttime<=t.landmark_time\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime<=t.landmark_time\nGROUP BY t.stay_id\n').df()
    uo_rel = target[['stay_id']].merge(affected, on='stay_id', how='left', validate='1:1')
    for c in ['has_ureteral_stent_by_landmark', 'has_gu_irrigation_by_landmark']:
        uo_rel[c] = uo_rel[c].fillna(0).astype('int8')
    uo_rel['uo_unreliable_by_landmark'] = (uo_rel.has_ureteral_stent_by_landmark.eq(1) | uo_rel.has_gu_irrigation_by_landmark.eq(1)).astype('int8')
    uo_qc = pd.DataFrame([{'target_stays': len(uo_rel), 'stent_by_landmark': int(uo_rel.has_ureteral_stent_by_landmark.sum()), 'gu_irrigation_by_landmark': int(uo_rel.has_gu_irrigation_by_landmark.sum()), 'either_unreliable_by_landmark': int(uo_rel.uo_unreliable_by_landmark.sum())}])
    display_table(uo_qc)
    wcsv(uo_qc, '04_uo_landmark_reliability_mask_summary.csv')
    wcsv(uo_rel.loc[uo_rel.uo_unreliable_by_landmark.eq(1)], '04_uo_landmark_reliability_affected_stays.csv')
    need_primary = ['stay_id', 'positive_stage_ibw', 'uo_positive_ibw', 'absence_evaluable_ibw', 'active_rrt_any_by_landmark', 'combined_status_ibw_anyrrt', 'combined_stage_ibw_anyrrt']
    miss = [c for c in need_primary if c not in primary.columns]
    if miss:
        raise KeyError(f'Primary combined missing: {miss}')
    target_owned = set(target.columns) - {'stay_id'}
    primary_payload = primary.drop(columns=[c for c in target_owned if c in primary.columns]).copy()
    cand = target.merge(primary_payload, on='stay_id', how='left', validate='1:1').merge(premorbid_cr[['stay_id', 'kdigo_creatinine_stage_rolling', 'kdigo_creatinine_stage_premorbid_median', 'kdigo_creatinine_stage_premorbid_mean_sensitivity', 'kdigo_creatinine_stage_premorbid_recent_sensitivity', 'creatinine_evaluable_premorbid', 'hist_outpt_n', 'hist_outpt_median', 'hist_outpt_mean', 'hist_outpt_most_recent', 'hist_outpt_first_time', 'hist_outpt_last_time', 'new_creatinine_positive_from_hist_median']], on='stay_id', how='left', validate='1:1').merge(uo_rel, on='stay_id', how='left', validate='1:1')
    required_target_cols = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'prior_documented_hf', krt_col]
    missing_target_cols = [col for col in required_target_cols if col not in cand.columns]
    if missing_target_cols:
        raise AssertionError(f'alternative assembly lost target-owned columns: {missing_target_cols}')
    unexpected_merge_suffixes = [col for col in cand.columns if col.endswith('_x') or col.endswith('_y')]
    if unexpected_merge_suffixes:
        raise AssertionError(f'alternative assembly produced merge suffixes; resolve collisions explicitly: {unexpected_merge_suffixes}')
    cand['urine_output_stage_reference'] = cand['positive_stage_ibw']
    cand['urine_output_positive_reference'] = cand['uo_positive_ibw']
    cand['urine_output_absence_evaluable_reference'] = cand['absence_evaluable_ibw']
    reliable_uo = cand.uo_unreliable_by_landmark.fillna(0).eq(0)
    cand['urine_output_stage_reliability_adjusted'] = pd.to_numeric(cand.urine_output_stage_reference, errors='coerce').where(reliable_uo)
    cand['urine_output_positive_reliability_adjusted'] = cand.urine_output_stage_reliability_adjusted.fillna(0).ge(1).astype('int8')
    cand['urine_output_absence_evaluable_reliability_adjusted'] = (cand.urine_output_absence_evaluable_reference.fillna(0).eq(1) & reliable_uo).astype('int8')

    def combined_variant(df, cr_col, uo_masked, label):
        x = df.copy()
        cr_stage = pd.to_numeric(x[cr_col], errors='coerce')
        cr_eval = x.creatinine_evaluable_premorbid.fillna(0).eq(1)
        if uo_masked:
            u_stage = pd.to_numeric(x.urine_output_stage_reliability_adjusted, errors='coerce')
            u_eval = x.urine_output_absence_evaluable_reliability_adjusted.fillna(0).eq(1)
        else:
            u_stage = pd.to_numeric(x.urine_output_stage_reference, errors='coerce')
            u_eval = x.urine_output_absence_evaluable_reference.fillna(0).eq(1)
        rrt_pos = x.active_rrt_any_by_landmark.fillna(0).eq(1)
        positive = cr_stage.fillna(-1).ge(1) | u_stage.fillna(-1).ge(1) | rrt_pos
        stage = pd.concat([cr_stage.fillna(-1), u_stage.fillna(-1)], axis=1).max(axis=1)
        stage = stage.where(stage >= 1, np.nan).mask(rrt_pos, 3)
        status = np.where(positive, 'positive', np.where(cr_eval & u_eval, 'negative', 'indeterminate'))
        x['_variant'] = label
        x['_combined_status'] = status
        x['_combined_stage'] = stage
        x['_cr_stage_used'] = cr_stage
        x['_uo_stage_used'] = u_stage
        x['_uo_masked'] = int(uo_masked)
        q = x.loc[x._combined_status.eq('positive')].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
        sel = q.drop_duplicates('subject_id', keep='first').copy()
        return (x, q, sel)
    reference_qualifying = cand.loc[cand.combined_status_ibw_anyrrt.eq('positive')].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
    reference_selection = reference_qualifying.drop_duplicates('subject_id', keep='first').copy()
    if set(reference_selection.stay_id.astype(int)) != set(reference_idx.stay_id.astype(int)):
        raise AssertionError('Alternative assembly no longer reproduces the reference cohort.')
    variants = {}
    variants['hybrid_median_fixed6_unmasked'] = combined_variant(cand, 'kdigo_creatinine_stage_premorbid_median', False, 'hybrid_median_fixed6_unmasked')
    variants['hybrid_median_fixed6_uo_masked'] = combined_variant(cand, 'kdigo_creatinine_stage_premorbid_median', True, 'hybrid_median_fixed6_uo_masked')
    variants['hybrid_mean_fixed6_uo_masked_sens'] = combined_variant(cand, 'kdigo_creatinine_stage_premorbid_mean_sensitivity', True, 'hybrid_mean_fixed6_uo_masked_sens')
    variants['hybrid_recent_fixed6_uo_masked_sens'] = combined_variant(cand, 'kdigo_creatinine_stage_premorbid_recent_sensitivity', True, 'hybrid_recent_fixed6_uo_masked_sens')
    rows = [{**membership_compare(reference_idx, reference_selection, 'reference_cohort'), 'qualifying_stays': len(reference_qualifying), 'creatinine_positive_selected': int(reference_selection.kdigo_creatinine_stage_rolling.fillna(0).ge(1).sum()), 'uo_unreliable_selected': int(reference_selection.uo_unreliable_by_landmark.fillna(0).eq(1).sum())}]
    for label, (full, q, sel) in variants.items():
        rows.append({**membership_compare(reference_idx, sel, label), 'qualifying_stays': len(q), 'creatinine_positive_selected': int(sel._cr_stage_used.fillna(0).ge(1).sum()), 'uo_unreliable_selected': int(sel.uo_unreliable_by_landmark.fillna(0).eq(1).sum())})
    membership = pd.DataFrame(rows)
    display_table(membership)
    wcsv(membership, 'premorbid_proxy_membership_comparison.csv')
    premorbid_target, q8, premorbid_selection = variants['hybrid_median_fixed6_uo_masked']
    bad_unreliable = premorbid_selection[premorbid_selection.uo_unreliable_by_landmark.fillna(0).eq(1) & premorbid_selection._cr_stage_used.fillna(0).lt(1) & premorbid_selection.active_rrt_any_by_landmark.fillna(0).ne(1)]
    if len(bad_unreliable):
        raise AssertionError('A selected premorbid-proxy stay relies only on a urine-output stream that should be unevaluable.')
    print('RECOMMENDED alternative N:', len(premorbid_selection))
    s = premorbid_selection.copy()
    s['creatinine_positive_premorbid'] = s._cr_stage_used.fillna(0).ge(1)
    s['urine_output_positive_premorbid'] = s._uo_stage_used.fillna(0).ge(1)
    s['rrt_positive_premorbid'] = s.active_rrt_any_by_landmark.fillna(0).eq(1)
    s['new_hist_cr_pos'] = s.new_creatinine_positive_from_hist_median.fillna(0).eq(1)
    s['component_pattern'] = 'Cr' + s.creatinine_positive_premorbid.astype(int).astype(str) + '_UO' + s.urine_output_positive_premorbid.astype(int).astype(str) + '_RRT' + s.rrt_positive_premorbid.astype(int).astype(str)
    component = s.groupby('component_pattern', as_index=False).agg(stays=('stay_id', 'size')).sort_values('stays', ascending=False)
    stage_dist = s['_combined_stage'].map(lambda x: 'UNK' if pd.isna(x) else str(int(x))).value_counts(dropna=False).rename_axis('combined_stage').reset_index(name='n')
    reference_map = reference_idx[['subject_id', 'stay_id']].rename(columns={'stay_id': 'reference_stay_id'})
    premorbid_map = premorbid_selection[['subject_id', 'stay_id']].rename(columns={'stay_id': 'premorbid_stay_id'})
    delta = reference_map.merge(premorbid_map, on='subject_id', how='outer', indicator=True)
    delta['same_stay'] = delta.reference_stay_id.fillna(-1).eq(delta.premorbid_stay_id.fillna(-2)).astype('int8')
    delta = delta.merge(s[['subject_id', 'stay_id', 'new_hist_cr_pos', 'creatinine_positive_premorbid', 'urine_output_positive_premorbid', 'rrt_positive_premorbid', 'uo_unreliable_by_landmark', '_combined_stage']].rename(columns={'stay_id': 'premorbid_stay_id'}), on=['subject_id', 'premorbid_stay_id'], how='left', validate='1:1')
    required_selected_cols = ['subject_id', 'stay_id', 'prior_documented_hf', krt_col, '_combined_status', '_combined_stage']
    missing_selected_cols = [col for col in required_selected_cols if col not in s.columns]
    if missing_selected_cols:
        raise AssertionError(f'Recommended-premorbid proxy cohort missing required columns: {missing_selected_cols}')
    if not s['prior_documented_hf'].eq(1).all():
        raise AssertionError('premorbid proxy cohort contains non-prior-HF stay.')
    if not s[krt_col].fillna(0).eq(0).all():
        raise AssertionError('premorbid proxy cohort contains chronic-KRT stay.')
    if s.subject_id.duplicated().any():
        raise AssertionError('premorbid proxy cohort contains duplicate subject.')
    if not s._combined_status.eq('positive').all():
        raise AssertionError('premorbid proxy cohort contains non-positive status.')
    selected_qc = pd.DataFrame([{'premorbid_proxy_subjects': len(s), 'premorbid_proxy_unique_stays': s.stay_id.nunique(), 'prior_hf_all': int(s['prior_documented_hf'].eq(1).all()), 'chronic_krt_all_excluded': int(s[krt_col].fillna(0).eq(0).all()), 'cr_positive': int(s.creatinine_positive_premorbid.sum()), 'uo_positive': int(s.urine_output_positive_premorbid.sum()), 'active_rrt_positive': int(s.rrt_positive_premorbid.sum()), 'new_historical_cr_positive_selected': int(s.new_hist_cr_pos.sum()), 'selected_with_uo_unreliable_but_cr_or_rrt_supported': int((s.uo_unreliable_by_landmark.fillna(0).eq(1) & (s.creatinine_positive_premorbid | s.rrt_positive_premorbid)).sum()), 'selected_relying_only_on_unreliable_uo': len(bad_unreliable)}])
    display_table(selected_qc)
    display_table(component)
    display_table(stage_dist)
    wcsv(selected_qc, 'premorbid_proxy_selected_summary.csv')
    wcsv(component, 'premorbid_proxy_component_patterns.csv')
    wcsv(stage_dist, 'premorbid_proxy_stage_distribution.csv')
    wcsv(delta, 'premorbid_proxy_membership_detail.csv')
    premorbid_index = premorbid_selection.copy()
    premorbid_index['kdigo_creatinine_stage_rolling_reference'] = premorbid_index['kdigo_creatinine_stage']
    premorbid_index['creatinine_evaluable_rolling_reference'] = premorbid_index['creatinine_evaluable']
    premorbid_index['combined_stage_reference'] = premorbid_index['combined_stage_ibw_anyrrt']
    premorbid_index['combined_status_reference'] = premorbid_index['combined_status_ibw_anyrrt']
    premorbid_index['kdigo_creatinine_stage'] = premorbid_index['kdigo_creatinine_stage_premorbid_median']
    premorbid_index['creatinine_evaluable'] = premorbid_index['creatinine_evaluable_premorbid']
    premorbid_index['positive_stage_ibw'] = premorbid_index['urine_output_stage_reliability_adjusted']
    premorbid_index['uo_positive_ibw'] = premorbid_index['urine_output_positive_reliability_adjusted']
    premorbid_index['absence_evaluable_ibw'] = premorbid_index['urine_output_absence_evaluable_reliability_adjusted']
    premorbid_index['combined_stage_ibw_anyrrt'] = premorbid_index['_combined_stage']
    premorbid_index['combined_status_ibw_anyrrt'] = premorbid_index['_combined_status']
    premorbid_index['creatinine_method'] = 'max(rolling_48h_7d, premorbid_outpatient_proxy_median_7_365d)'
    premorbid_index['urine_output_method'] = 'event_gap6_full_IBW'
    premorbid_index['urine_output_reliability_rule'] = 'UO unevaluable if ureteral stent or GU irrigation available by landmark'
    premorbid_index['heart_failure_method'] = 'prior_completed_admission_direct_HF'
    premorbid_index['chronic_krt_method'] = 'chronic_krt_definite'
    premorbid_index['rrt_method'] = 'active_acute_RRT_by_landmark'
    drop_internal = [c for c in premorbid_index.columns if c.startswith('_')]
    premorbid_index = premorbid_index.drop(columns=drop_internal)
    assert_outcome_blind(premorbid_index, 'premorbid proxy cohort')
    assert_unique(premorbid_index, 'subject_id', 'premorbid proxy cohort')
    assert_unique(premorbid_index, 'stay_id', 'premorbid proxy cohort')
    P_PREMORBID_CR = OUT / 'premorbid_creatinine.parquet'
    P_PREMORBID_TARGET = OUT / 'premorbid_creatinine_target.parquet'
    P_PREMORBID_INDEX = OUT / 'premorbid_creatinine_cohort_preoutcome.parquet'
    premorbid_cr.to_parquet(P_PREMORBID_CR, index=False)
    premorbid_target.to_parquet(P_PREMORBID_TARGET, index=False)
    premorbid_index.to_parquet(P_PREMORBID_INDEX, index=False)
    premorbid_sha = sha256_file(P_PREMORBID_INDEX)
    artifact_qc = pd.DataFrame([{'artifact': P_PREMORBID_CR.name, 'rows': len(premorbid_cr), 'sha256': sha256_file(P_PREMORBID_CR)}, {'artifact': P_PREMORBID_TARGET.name, 'rows': len(premorbid_target), 'sha256': sha256_file(P_PREMORBID_TARGET)}, {'artifact': P_PREMORBID_INDEX.name, 'rows': len(premorbid_index), 'sha256': premorbid_sha}])
    display_table(artifact_qc)
    wcsv(artifact_qc, 'premorbid_proxy_artifact_hashes.csv')
    membership_row = membership.loc[membership.variant.eq('hybrid_median_fixed6_uo_masked')].iloc[0]
    manifest = {
        'status': 'PASS',
        'outcome_used': False,
        'reference_cohort_n': int(len(reference_idx)),
        'premorbid_proxy_cohort_n': int(len(premorbid_index)),
        'reference_cohort_current_run_sha256': reference_sha,
        'premorbid_proxy_current_run_sha256': premorbid_sha,
        'definition': {
            'landmark': '+24 h ICU',
            'heart_failure': 'prior completed-admission direct HF',
            'chronic_krt': 'prespecified chronic-KRT exclusion',
            'creatinine': 'maximum of rolling 48h/7d stage and premorbid outpatient-proxy median 7–365d ratio stage',
            'urine_output': 'event_gap6_full + IBW',
            'urine_output_reliability': 'urine-output criterion unevaluable when landmark-safe ureteral-stent or GU-irrigation evidence exists',
            'rrt': 'active acute RRT by landmark contributes KDIGO Stage 3 after chronic-KRT exclusion',
            'episode': 'first qualifying ICU stay per subject',
        },
        'membership_vs_reference': {
            'added_subjects': int(membership_row.subject_added_vs_reference),
            'dropped_subjects': int(membership_row.subject_dropped_vs_reference),
            'same_subject_replaced_stay': int(membership_row.same_subject_replaced_stay),
        },
        'hard_guards': {
            'reference_exact_reconstruction': True,
            'rolling_creatinine_exact_reproduction': True,
            'no_rolling_creatinine_stage_demotion': True,
            'outcome_blind': True,
            'prior_hf_all_selected': True,
            'chronic_krt_all_selected_excluded': True,
            'unreliable_uo_cannot_qualify_alone': True,
        },
        'sensitivity_variants': [
            'hybrid_median_fixed6_unmasked',
            'hybrid_mean_fixed6_uo_masked_sensitivity',
            'hybrid_recent_fixed6_uo_masked_sensitivity',
        ],
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    wjson(manifest, 'premorbid_creatinine_manifest.json')
    print('PREMORBID CREATININE PROXY: PASS')
    print('Premorbid-proxy cohort N:', len(premorbid_index))
    print('Current-run cohort SHA256:', premorbid_sha)


if __name__ == "__main__":
    main()
