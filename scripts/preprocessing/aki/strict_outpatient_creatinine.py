#!/usr/bin/env python3
"""08 Strict Outpatient Creatinine.

Remove hospital-contact creatinine from history.

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
    import json
    import numpy as np
    import pandas as pd
    import duckdb
    EXPECTED_REFERENCE_N = 3251
    EXPECTED_PREMORBID_N = 3495
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT))).resolve()
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', str(_RAW_DATA_ROOT))).resolve()
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f'MIMIC-IV root not found: {MIMIC_ROOT}')
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    HISTORICAL_DIR = PHENOTYPE_DIR / 'historical_creatinine'
    PREMORBID_DIR = HISTORICAL_DIR / 'premorbid_proxy'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'historical_creatinine' / 'strict_outpatient'
    QC.mkdir(parents=True, exist_ok=True)

    def req(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: null {key}')
        if df[key].duplicated().any():
            raise AssertionError(f'{label}: duplicate {key}')

    def assert_outcome_blind(df, label):
        forbidden = ('expire', 'mortality', 'death', 'outcome', 'auc', 'calibration')
        bad = [c for c in df.columns if any((x in c.lower() for x in forbidden))]
        if bad:
            raise AssertionError(f'{label} contains prohibited fields: {bad}')

    def wcsv(df, name):
        p = QC / name
        df.to_csv(p, index=False)
        return p

    def wjson(obj, name):
        p = QC / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p

    def membership_compare(reference, alternative, label):
        r = reference[['subject_id', 'stay_id']].copy()
        a = alternative[['subject_id', 'stay_id']].copy()
        assert_unique(r, 'subject_id', 'reference')
        assert_unique(a, 'subject_id', label)
        rm = dict(zip(r.subject_id.astype(int), r.stay_id.astype(int)))
        am = dict(zip(a.subject_id.astype(int), a.stay_id.astype(int)))
        rs, aset = (set(rm), set(am))
        common = rs & aset
        replaced = sum((rm[s] != am[s] for s in common))
        return {'variant': label, 'subjects': len(a), 'added_vs_premorbid': len(aset - rs), 'dropped_vs_premorbid': len(rs - aset), 'common_subjects': len(common), 'replaced_selected_stay': int(replaced), 'same_subject_same_stay': int(len(common) - replaced), 'stay_symmetric_difference': len(set(rm.values()) ^ set(am.values()))}
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT  :', MIMIC_ROOT)
    print('QC OUTPUT   :', QC)
    P_LM = req(COHORT_DIR / 'landmark_population.parquet')
    P_PH = req(COHORT_DIR / 'heart_failure_krt_phenotypes.parquet')
    P_REFERENCE = req(COHORT_DIR / 'reference_cohort_preoutcome.parquet')
    P_PREMORBID = req(PREMORBID_DIR / 'premorbid_creatinine_cohort_preoutcome.parquet')
    P_PREMORBID_TARGET = req(PREMORBID_DIR / 'premorbid_creatinine_target.parquet')
    P_PREMORBID_CR = req(PREMORBID_DIR / 'premorbid_creatinine.parquet')
    lm = pd.read_parquet(P_LM, columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time'])
    ph = pd.read_parquet(P_PH)
    reference_idx = pd.read_parquet(P_REFERENCE)
    premorbid_idx = pd.read_parquet(P_PREMORBID)
    premorbid_target = pd.read_parquet(P_PREMORBID_TARGET)
    premorbid_cr_saved = pd.read_parquet(P_PREMORBID_CR)
    for label, df in [('landmark', lm), ('phenotype', ph), ('reference cohort', reference_idx), ('premorbid cohort', premorbid_idx), ('premorbid target', premorbid_target), ('premorbid creatinine', premorbid_cr_saved)]:
        assert_outcome_blind(df, label)
    if len(reference_idx) != EXPECTED_REFERENCE_N or reference_idx.subject_id.nunique() != EXPECTED_REFERENCE_N:
        raise AssertionError('Reference cohort N mismatch.')
    if len(premorbid_idx) != EXPECTED_PREMORBID_N or premorbid_idx.subject_id.nunique() != EXPECTED_PREMORBID_N:
        raise AssertionError('Premorbid-proxy cohort N mismatch.')
    krt_col = 'chronic_krt_definite'
    target = lm.merge(ph[['stay_id', 'prior_documented_hf', krt_col]], on='stay_id', how='left', validate='1:1')
    target = target[target.prior_documented_hf.eq(1) & target[krt_col].fillna(0).eq(0)].copy()
    assert_unique(target, 'stay_id', 'target')
    if set(target.stay_id.astype(int)) != set(premorbid_target.stay_id.astype(int)):
        raise AssertionError('Target membership differs from saved premorbid target.')
    guards = pd.DataFrame([{'reference_cohort_n': len(reference_idx), 'premorbid_cohort_n': len(premorbid_idx), 'target_stays': len(target), 'outcome_blind': 1}])
    display_table(guards)
    wcsv(guards, '01_hard_guards.csv')
    TR = req(MIMIC_ROOT / 'hosp' / 'transfers.csv.gz')
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.execute(f"\nCREATE TEMP TABLE transfers_all AS\nSELECT\n    subject_id,hadm_id,eventtype,careunit,intime,outtime\nFROM read_csv_auto(\n    '{qpath(TR)}',header=true,sample_size=200000\n)\n")
    transfer_profile = con.execute("\nSELECT\n    LOWER(COALESCE(eventtype,'<null>')) AS eventtype,\n    COALESCE(careunit,'<NULL>') AS careunit,\n    COUNT(*) AS rows,\n    SUM(CASE WHEN intime IS NULL THEN 1 ELSE 0 END) AS intime_null,\n    SUM(CASE WHEN outtime IS NULL THEN 1 ELSE 0 END) AS outtime_null,\n    SUM(CASE\n        WHEN intime IS NOT NULL\n         AND outtime IS NOT NULL\n         AND outtime>intime\n        THEN 1 ELSE 0 END\n    ) AS valid_positive_duration_rows\nFROM transfers_all\nGROUP BY 1,2\nORDER BY rows DESC\n").df()
    eventtype_profile = con.execute("\nSELECT\n    LOWER(COALESCE(eventtype,'<null>')) AS eventtype,\n    COUNT(*) AS rows,\n    SUM(CASE WHEN careunit IS NULL THEN 1 ELSE 0 END) AS careunit_null,\n    SUM(CASE WHEN outtime IS NULL THEN 1 ELSE 0 END) AS outtime_null,\n    SUM(CASE\n        WHEN intime IS NOT NULL\n         AND outtime IS NOT NULL\n         AND outtime>intime\n        THEN 1 ELSE 0 END\n    ) AS valid_positive_duration_rows\nFROM transfers_all\nGROUP BY 1\nORDER BY rows DESC\n").df()
    display_table(eventtype_profile)
    wcsv(eventtype_profile, '02_transfer_eventtype_profile.csv')
    wcsv(transfer_profile, '02_transfer_eventtype_careunit_profile.csv')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    con.register('target_df', target[['subject_id', 'stay_id', 'admittime', 'intime', 'landmark_time']].copy())
    con.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM target_df')
    con.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT\n    subject_id,hadm_id,charttime,storetime,valuenum AS creatinine\nFROM read_csv_auto(\n    '{qpath(LAB)}',header=true,sample_size=200000\n)\nWHERE itemid=50912\n  AND valuenum>0\n  AND valuenum<=150\n  AND charttime IS NOT NULL\n  AND storetime IS NOT NULL\n")
    con.execute(f"\nCREATE TEMP TABLE admissions AS\nSELECT\n    subject_id,hadm_id,admittime,dischtime,edregtime,edouttime\nFROM read_csv_auto(\n    '{qpath(ADM)}',header=true,sample_size=200000\n)\n")
    con.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT\n    t.stay_id,t.subject_id,t.admittime,t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS storetime,\n    AVG(c.creatinine) AS creatinine,\n    MAX(CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS has_labevent_hadm_id\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\nGROUP BY\n    t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    con.execute("\nCREATE TEMP TABLE valid_physical_transfers AS\nSELECT\n    subject_id,hadm_id,\n    LOWER(COALESCE(eventtype,'')) AS eventtype,\n    careunit,intime,outtime\nFROM transfers_all\nWHERE intime IS NOT NULL\n  AND outtime IS NOT NULL\n  AND outtime>intime\n  AND careunit IS NOT NULL\n  AND LOWER(COALESCE(eventtype,'')) IN ('ed','admit','transfer')\n")
    con.execute("\nCREATE TEMP TABLE valid_ed_transfers AS\nSELECT *\nFROM valid_physical_transfers\nWHERE eventtype='ed'\n   OR LOWER(careunit) LIKE '%emergency%'\n")
    con.execute('\nCREATE TEMP TABLE hist_classified AS\nSELECT\n    h.*,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM admissions a\n        WHERE a.subject_id=h.subject_id\n          AND a.admittime IS NOT NULL\n          AND a.dischtime IS NOT NULL\n          AND h.charttime>=a.admittime\n          AND h.charttime<=a.dischtime\n    ) THEN 1 ELSE 0 END AS overlaps_admission,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM admissions a\n        WHERE a.subject_id=h.subject_id\n          AND a.edregtime IS NOT NULL\n          AND a.edouttime IS NOT NULL\n          AND a.edouttime>a.edregtime\n          AND h.charttime>=a.edregtime\n          AND h.charttime<=a.edouttime\n    ) THEN 1 ELSE 0 END AS overlaps_admissions_ed_interval,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM valid_ed_transfers e\n        WHERE e.subject_id=h.subject_id\n          AND h.charttime>=e.intime\n          AND h.charttime<=e.outtime\n    ) THEN 1 ELSE 0 END AS overlaps_valid_ed_transfer,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM valid_physical_transfers tr\n        WHERE tr.subject_id=h.subject_id\n          AND h.charttime>=tr.intime\n          AND h.charttime<=tr.outtime\n    ) THEN 1 ELSE 0 END AS overlaps_valid_physical_transfer\n\nFROM hist_same_time h\n')
    baselines = con.execute('\nSELECT\n    t.stay_id,\n\n    -- Exact premorbid proxy: admission-time overlap only.\n    COUNT(h.charttime) FILTER (\n      WHERE h.overlaps_admission=0\n    ) AS premorbid_proxy_n,\n    MEDIAN(h.creatinine) FILTER (\n      WHERE h.overlaps_admission=0\n    ) AS premorbid_proxy_median,\n\n    -- Corrected ED exclusion: known valid ED intervals, no open-ended transfer rows.\n    COUNT(h.charttime) FILTER (\n      WHERE h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n    ) AS valid_ed_excluded_n,\n    MEDIAN(h.creatinine) FILTER (\n      WHERE h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n    ) AS valid_ed_excluded_median,\n\n    -- Add hadm_id-null requirement.\n    COUNT(h.charttime) FILTER (\n      WHERE h.has_labevent_hadm_id=0\n        AND h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n    ) AS hadm_null_valid_ed_n,\n    MEDIAN(h.creatinine) FILTER (\n      WHERE h.has_labevent_hadm_id=0\n        AND h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n    ) AS hadm_null_valid_ed_median,\n\n    -- Strict outpatient proxy.\n    COUNT(h.charttime) FILTER (\n      WHERE h.has_labevent_hadm_id=0\n        AND h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n        AND h.overlaps_valid_physical_transfer=0\n    ) AS strict_outpatient_n,\n    MEDIAN(h.creatinine) FILTER (\n      WHERE h.has_labevent_hadm_id=0\n        AND h.overlaps_admission=0\n        AND h.overlaps_admissions_ed_interval=0\n        AND h.overlaps_valid_ed_transfer=0\n        AND h.overlaps_valid_physical_transfer=0\n    ) AS strict_outpatient_median,\n\n    SUM(CASE\n      WHEN h.overlaps_admission=0\n       AND (h.overlaps_admissions_ed_interval=1 OR h.overlaps_valid_ed_transfer=1)\n      THEN 1 ELSE 0 END\n    ) AS current_proxy_values_with_known_ed_contact,\n\n    SUM(CASE\n      WHEN h.overlaps_admission=0\n       AND h.has_labevent_hadm_id=1\n      THEN 1 ELSE 0 END\n    ) AS current_proxy_values_with_hadm_id,\n\n    SUM(CASE\n      WHEN h.overlaps_admission=0\n       AND h.overlaps_valid_physical_transfer=1\n      THEN 1 ELSE 0 END\n    ) AS current_proxy_values_with_valid_physical_transfer\n\nFROM target_lm t\nLEFT JOIN hist_classified h USING(stay_id)\nGROUP BY t.stay_id\n').df()
    assert_unique(baselines, 'stay_id', 'baselines')
    chk = premorbid_cr_saved[['stay_id', 'hist_outpt_n', 'hist_outpt_median']].merge(baselines[['stay_id', 'premorbid_proxy_n', 'premorbid_proxy_median']], on='stay_id', how='left', validate='1:1')
    same_n = chk.hist_outpt_n.fillna(0).astype(int).eq(chk.premorbid_proxy_n.fillna(0).astype(int))
    same_med = np.isclose(chk.hist_outpt_median.fillna(-999), chk.premorbid_proxy_median.fillna(-999), rtol=0, atol=1e-10)
    if not (same_n & same_med).all():
        bad = chk.loc[~(same_n & same_med)]
        wcsv(bad, 'premorbid_baseline_mismatch.csv')
        raise AssertionError(f'Premorbid baseline reproduction mismatch: {len(bad)}')
    baseline_summary = pd.DataFrame([{'target_stays': len(baselines), 'premorbid_available': int(baselines.premorbid_proxy_median.notna().sum()), 'valid_ed_excluded_available': int(baselines.valid_ed_excluded_median.notna().sum()), 'hadm_null_valid_ed_available': int(baselines.hadm_null_valid_ed_median.notna().sum()), 'strict_outpatient_available': int(baselines.strict_outpatient_median.notna().sum()), 'stays_with_current_proxy_known_ed_contact': int(baselines.current_proxy_values_with_known_ed_contact.fillna(0).gt(0).sum()), 'stays_with_current_proxy_hadm_id': int(baselines.current_proxy_values_with_hadm_id.fillna(0).gt(0).sum()), 'stays_with_current_proxy_valid_physical_transfer': int(baselines.current_proxy_values_with_valid_physical_transfer.fillna(0).gt(0).sum()), 'premorbid_exact_reproduction': 1}])
    display_table(baseline_summary)
    wcsv(baseline_summary, '03_baseline_definition_summary.csv')
    wcsv(baselines, '03_baseline_definition_by_target_stay.csv')
    con.execute("\nCREATE TEMP TABLE cr_same_time AS\nSELECT\n    t.stay_id,t.intime,t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS storetime,\n    AVG(c.creatinine) AS creatinine\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.intime,t.landmark_time,c.charttime\n")
    con.execute("\nCREATE TEMP TABLE cr_refs AS\nSELECT\n    *,\n    MIN(creatinine) OVER(\n      PARTITION BY stay_id ORDER BY charttime\n      RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n      AND INTERVAL '1 microsecond' PRECEDING\n    ) AS low48\nFROM cr_same_time\n")
    con.register('baselines_df', baselines)
    con.execute('CREATE TEMP TABLE baselines_tbl AS SELECT * FROM baselines_df')
    events = con.execute('\nSELECT\n    r.stay_id,r.charttime,r.creatinine,r.low48,\n    b.premorbid_proxy_median,\n    b.valid_ed_excluded_median,\n    b.hadm_null_valid_ed_median,\n    b.strict_outpatient_median\nFROM cr_refs r\nJOIN target_lm t USING(stay_id)\nLEFT JOIN baselines_tbl b USING(stay_id)\nWHERE r.charttime>=t.intime\n  AND r.charttime<=t.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()

    def historical_stage(df, baseline_col):
        cr = pd.to_numeric(df['creatinine'], errors='coerce')
        b = pd.to_numeric(df[baseline_col], errors='coerce')
        low48 = pd.to_numeric(df['low48'], errors='coerce')
        has = b.notna()
        ratio15 = has & cr.ge(1.5 * b)
        ratio20 = has & cr.ge(2.0 * b)
        ratio30 = has & cr.ge(3.0 * b)
        delta03 = low48.notna() & cr.ge(low48 + 0.3)
        out = np.select([ratio30, cr.ge(4.0) & (ratio15 | delta03), ratio20, ratio15], [3.0, 3.0, 2.0, 1.0], default=0.0)
        out = pd.Series(out, index=df.index, dtype='float64')
        out.loc[~has] = np.nan
        return out
    variant_cols = {'premorbid_proxy': 'premorbid_proxy_median', 'ed_excluded_proxy': 'valid_ed_excluded_median', 'hadm_null_ed_excluded_proxy': 'hadm_null_valid_ed_median', 'strict_outpatient_proxy': 'strict_outpatient_median'}
    for label, col in variant_cols.items():
        events[f'hist_stage__{label}'] = historical_stage(events, col)
    hist_stage_cols = [f'hist_stage__{x}' for x in variant_cols]
    stage_stay = events.groupby('stay_id', as_index=False)[hist_stage_cols].max()
    stage_df = target[['stay_id']].merge(stage_stay, on='stay_id', how='left', validate='1:1').merge(premorbid_cr_saved[['stay_id', 'kdigo_creatinine_stage_rolling', 'kdigo_creatinine_stage_premorbid_median']], on='stay_id', how='left', validate='1:1').merge(baselines, on='stay_id', how='left', validate='1:1')

    def rowmax(a, b):
        aa = pd.to_numeric(a, errors='coerce')
        bb = pd.to_numeric(b, errors='coerce')
        out = pd.concat([aa, bb], axis=1).max(axis=1, skipna=True)
        return out.where(aa.notna() | bb.notna())
    for label in variant_cols:
        stage_df[f'hybrid__{label}'] = rowmax(stage_df.kdigo_creatinine_stage_rolling, stage_df[f'hist_stage__{label}'])
    same = stage_df['hybrid__premorbid_proxy'].fillna(-99).eq(stage_df.kdigo_creatinine_stage_premorbid_median.fillna(-99))
    if not same.all():
        bad = stage_df.loc[~same]
        wcsv(bad, 'premorbid_stage_reproduction_mismatch.csv')
        raise AssertionError(f'Premorbid hybrid stage mismatch={len(bad)}')
    stage_summary = []
    ref = stage_df['hybrid__premorbid_proxy']
    for label in variant_cols:
        cur = stage_df[f'hybrid__{label}']
        stage_summary.append({'variant': label, 'creatinine_positive_target_stays': int(cur.fillna(0).ge(1).sum()), 'stage_changed_vs_premorbid': int((~cur.fillna(-99).eq(ref.fillna(-99))).sum()), 'premorbid_positive_to_variant_negative': int((ref.fillna(0).ge(1) & cur.fillna(0).lt(1)).sum()), 'premorbid_proxy_negative_to_variant_positive': int((ref.fillna(0).lt(1) & cur.fillna(0).ge(1)).sum())})
    stage_summary = pd.DataFrame(stage_summary)
    display_table(stage_summary)
    wcsv(stage_summary, '04_hybrid_creatinine_stage_summary.csv')
    wcsv(stage_df, '04_hybrid_creatinine_stage_by_target_stay.csv')
    needed = ['stay_id', 'subject_id', 'intime', 'urine_output_stage_reliability_adjusted', 'active_rrt_any_by_landmark']
    miss = [c for c in needed if c not in premorbid_target.columns]
    if miss:
        raise KeyError(f'Saved premorbid target missing: {miss}')
    x = premorbid_target[needed].merge(stage_df[['stay_id'] + [f'hybrid__{label}' for label in variant_cols]], on='stay_id', how='left', validate='1:1')

    def select_variant(label):
        cr_col = f'hybrid__{label}'
        z = x.copy()
        cr = pd.to_numeric(z[cr_col], errors='coerce')
        uo = pd.to_numeric(z.urine_output_stage_reliability_adjusted, errors='coerce')
        rrt = z.active_rrt_any_by_landmark.fillna(0).eq(1)
        positive = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        q = z.loc[positive].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
        sel = q.drop_duplicates('subject_id', keep='first').copy()
        return (q, sel)
    rows = []
    detail = []
    sels = {}
    for label in variant_cols:
        q, sel = select_variant(label)
        sels[label] = sel
        cmp = membership_compare(premorbid_idx, sel, label)
        cmp['qualifying_stays'] = len(q)
        cmp['creatinine_positive_selected'] = int(sel[f'hybrid__{label}'].fillna(0).ge(1).sum())
        rows.append(cmp)
        d = premorbid_idx[['subject_id', 'stay_id']].rename(columns={'stay_id': 'premorbid_stay_id'}).merge(sel[['subject_id', 'stay_id']].rename(columns={'stay_id': 'variant_stay_id'}), on='subject_id', how='outer', indicator=True)
        d['variant'] = label
        detail.append(d)
    cohort_summary = pd.DataFrame(rows)
    cur = cohort_summary.loc[cohort_summary.variant.eq('premorbid_proxy')].iloc[0]
    if int(cur.subjects) != EXPECTED_PREMORBID_N or int(cur.added_vs_premorbid) != 0 or int(cur.dropped_vs_premorbid) != 0 or (int(cur.replaced_selected_stay) != 0):
        raise AssertionError('Premorbid cohort not exactly reproduced.')
    display_table(cohort_summary)
    wcsv(cohort_summary, 'cohort_membership_by_baseline_proxy.csv')
    wcsv(pd.concat(detail, ignore_index=True), '05_subject_membership_detail.csv')
    removed = con.execute('\nSELECT\n    stay_id,subject_id,charttime,creatinine,\n    has_labevent_hadm_id,\n    overlaps_admission,\n    overlaps_admissions_ed_interval,\n    overlaps_valid_ed_transfer,\n    overlaps_valid_physical_transfer\nFROM hist_classified\nWHERE overlaps_admission=0\n  AND (\n      has_labevent_hadm_id=1\n      OR overlaps_admissions_ed_interval=1\n      OR overlaps_valid_ed_transfer=1\n      OR overlaps_valid_physical_transfer=1\n  )\nORDER BY stay_id,charttime\n').df()
    prov = pd.DataFrame([{'removed_from_strict_measurements': len(removed), 'removed_from_strict_stays': removed.stay_id.nunique(), 'with_hadm_id': int(removed.has_labevent_hadm_id.eq(1).sum()), 'overlap_admissions_ed_interval': int(removed.overlaps_admissions_ed_interval.eq(1).sum()), 'overlap_valid_ed_transfer': int(removed.overlaps_valid_ed_transfer.eq(1).sum()), 'overlap_valid_physical_transfer': int(removed.overlaps_valid_physical_transfer.eq(1).sum()), 'median_removed_creatinine': float(removed.creatinine.median()) if len(removed) else np.nan}])
    display_table(prov)
    wcsv(prov, '06_removed_measurement_provenance_summary.csv')
    wcsv(removed, '06_removed_measurements_from_strict_proxy.csv')
    strict_row = cohort_summary.loc[cohort_summary.variant.eq('strict_outpatient_proxy')].iloc[0]
    manifest = {
        'status': 'PASS',
        'outcome_used': False,
        'reference_cohort_n': int(len(reference_idx)),
        'premorbid_cohort_n': int(len(premorbid_idx)),
        'strict_outpatient_proxy': {
            'subjects': int(strict_row.subjects),
            'added_vs_premorbid': int(strict_row.added_vs_premorbid),
            'dropped_vs_premorbid': int(strict_row.dropped_vs_premorbid),
            'replaced_selected_stay': int(strict_row.replaced_selected_stay),
        },
        'definition': [
            'historical window >7 to 365 days before index hospital admission',
            'labevents.hadm_id must be NULL',
            'no overlap with hospital admissions',
            'no overlap with admissions emergency-department registration/out interval',
            'no overlap with valid positive-duration emergency-department transfer',
            'no overlap with valid positive-duration physical hospital transfer',
            'median historical creatinine',
            'hybrid stage=max(rolling KDIGO stage,historical stage)',
        ],
        'hard_guards': {
            'premorbid_baseline_exact_reproduction': True,
            'premorbid_hybrid_stage_exact_reproduction': True,
            'premorbid_cohort_exact_reproduction': True,
            'null_transfer_outtime_never_treated_as_open_interval': True,
            'outcome_blind': True,
        },
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    wjson(manifest, 'strict_outpatient_audit_manifest.json')
    print('STRICT OUTPATIENT CREATININE AUDIT: PASS')
    print('Strict-outpatient cohort N:', int(strict_row.subjects))


if __name__ == "__main__":
    main()
