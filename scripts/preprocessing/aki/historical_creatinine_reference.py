#!/usr/bin/env python3
"""09 Historical Creatinine Reference.

Establish the historical-creatinine reference definition.

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
    EXPECTED_PREMORBID_N = 3495
    EXPECTED_STRICT_N = 3422
    EXPECTED_TARGET_N = 9972
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT))).resolve()
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', str(_RAW_DATA_ROOT))).resolve()
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f'MIMIC-IV root not found: {MIMIC_ROOT}')
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    HISTORICAL_DIR = PHENOTYPE_DIR / 'historical_creatinine'
    PREMORBID_DIR = HISTORICAL_DIR / 'premorbid_proxy'
    OUT = HISTORICAL_DIR / 'strict_outpatient'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'historical_creatinine' / 'strict_outpatient'
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

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

    def membership_sha(df):
        z = df[['subject_id', 'stay_id']].copy()
        z['subject_id'] = z['subject_id'].astype('int64')
        z['stay_id'] = z['stay_id'].astype('int64')
        z = z.sort_values(['subject_id', 'stay_id']).reset_index(drop=True)
        payload = 'subject_id,stay_id\n' + ''.join((f'{int(r.subject_id)},{int(r.stay_id)}\n' for r in z.itertuples(index=False)))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: null {key}')
        if df[key].duplicated().any():
            raise AssertionError(f'{label}: duplicate {key}')

    def outcome_like_columns(df):
        toks = ('expire', 'mortality', 'death', 'outcome', 'deathtime', 'hospital_expire')
        return [c for c in df.columns if any((t in c.lower() for t in toks))]

    def assert_outcome_blind(df, label):
        bad = outcome_like_columns(df)
        if bad:
            raise AssertionError(f'{label} contains outcome-like columns: {bad}')

    def wcsv(df, name):
        p = QC / name
        df.to_csv(p, index=False)
        return p

    def wjson(obj, name):
        p = QC / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT  :', MIMIC_ROOT)
    print('OUTPUT      :', OUT)
    P_LM = req(COHORT_DIR / 'landmark_population.parquet')
    P_REFERENCE = req(COHORT_DIR / 'reference_cohort_preoutcome.parquet')
    P_PREMORBID_INDEX = req(PREMORBID_DIR / 'premorbid_creatinine_cohort_preoutcome.parquet')
    P_PREMORBID_TARGET = req(PREMORBID_DIR / 'premorbid_creatinine_target.parquet')
    P_PREMORBID_CR = req(PREMORBID_DIR / 'premorbid_creatinine.parquet')
    lm_blind = pd.read_parquet(P_LM, columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time'])
    reference_idx = pd.read_parquet(P_REFERENCE)
    premorbid_idx = pd.read_parquet(P_PREMORBID_INDEX)
    premorbid_target = pd.read_parquet(P_PREMORBID_TARGET)
    premorbid_cr_saved = pd.read_parquet(P_PREMORBID_CR)
    for label, df in [('landmark', lm_blind), ('reference cohort', reference_idx), ('premorbid cohort', premorbid_idx), ('premorbid target', premorbid_target), ('premorbid creatinine', premorbid_cr_saved)]:
        assert_outcome_blind(df, label)
    assert_unique(reference_idx, 'subject_id', 'reference cohort')
    assert_unique(premorbid_idx, 'subject_id', 'premorbid cohort')
    assert_unique(premorbid_target, 'stay_id', 'premorbid target')
    assert_unique(premorbid_cr_saved, 'stay_id', 'premorbid creatinine')
    if len(reference_idx) != EXPECTED_REFERENCE_N:
        raise AssertionError(f'Reference cohort N={len(reference_idx)}, expected {EXPECTED_REFERENCE_N}.')
    if len(premorbid_idx) != EXPECTED_PREMORBID_N:
        raise AssertionError(f'Premorbid-proxy cohort N={len(premorbid_idx)}, expected {EXPECTED_PREMORBID_N}.')
    required_target = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'prior_documented_hf', 'chronic_krt_definite', 'urine_output_stage_reliability_adjusted', 'urine_output_absence_evaluable_reliability_adjusted', 'uo_unreliable_by_landmark', 'active_rrt_any_by_landmark']
    missing = [c for c in required_target if c not in premorbid_target.columns]
    if missing:
        raise KeyError(f'Premorbid target missing required columns: {missing}')
    if len(premorbid_target) != EXPECTED_TARGET_N:
        raise AssertionError(f'Expected {EXPECTED_TARGET_N:,} target stays; got {len(premorbid_target):,}.')
    if not premorbid_target['prior_documented_hf'].eq(1).all():
        raise AssertionError('Premorbid target contains a stay without prior documented heart failure.')
    if not premorbid_target['chronic_krt_definite'].fillna(0).eq(0).all():
        raise AssertionError('Premorbid target contains a chronic-KRT stay.')
    guards = pd.DataFrame([{'reference_cohort_n': len(reference_idx), 'premorbid_cohort_n': len(premorbid_idx), 'target_stays': len(premorbid_target), 'outcome_blind': 1}])
    display_table(guards)
    wcsv(guards, 'strict_outpatient_hard_guards.csv')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    TR = req(MIMIC_ROOT / 'hosp' / 'transfers.csv.gz')
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    target_core = premorbid_target[['subject_id', 'stay_id', 'admittime', 'intime', 'landmark_time']].copy()
    con.register('target_df', target_core)
    con.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM target_df')
    con.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT\n    subject_id,hadm_id,charttime,storetime,valuenum AS creatinine\nFROM read_csv_auto(\n    '{qpath(LAB)}',\n    header=true,\n    sample_size=200000\n)\nWHERE itemid=50912\n  AND valuenum>0\n  AND valuenum<=150\n  AND charttime IS NOT NULL\n  AND storetime IS NOT NULL\n")
    con.execute(f"\nCREATE TEMP TABLE admissions AS\nSELECT\n    subject_id,hadm_id,admittime,dischtime,edregtime,edouttime\nFROM read_csv_auto(\n    '{qpath(ADM)}',\n    header=true,\n    sample_size=200000\n)\n")
    con.execute(f"\nCREATE TEMP TABLE transfers_all AS\nSELECT\n    subject_id,hadm_id,eventtype,careunit,intime,outtime\nFROM read_csv_auto(\n    '{qpath(TR)}',\n    header=true,\n    sample_size=200000\n)\n")
    con.execute("\nCREATE TEMP TABLE valid_physical_transfers AS\nSELECT\n    subject_id,hadm_id,\n    LOWER(COALESCE(eventtype,'')) AS eventtype,\n    careunit,intime,outtime\nFROM transfers_all\nWHERE intime IS NOT NULL\n  AND outtime IS NOT NULL\n  AND outtime>intime\n  AND careunit IS NOT NULL\n  AND LOWER(COALESCE(eventtype,'')) IN ('ed','admit','transfer')\n")
    con.execute("\nCREATE TEMP TABLE valid_ed_transfers AS\nSELECT *\nFROM valid_physical_transfers\nWHERE eventtype='ed'\n   OR LOWER(careunit) LIKE '%emergency%'\n")
    con.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT\n    t.stay_id,t.subject_id,t.admittime,t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS storetime,\n    AVG(c.creatinine) AS creatinine,\n    MAX(CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS has_labevent_hadm_id\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\nGROUP BY\n    t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    con.execute('\nCREATE TEMP TABLE hist_classified AS\nSELECT\n    h.*,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM admissions a\n        WHERE a.subject_id=h.subject_id\n          AND a.admittime IS NOT NULL\n          AND a.dischtime IS NOT NULL\n          AND h.charttime>=a.admittime\n          AND h.charttime<=a.dischtime\n    ) THEN 1 ELSE 0 END AS overlaps_admission,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM admissions a\n        WHERE a.subject_id=h.subject_id\n          AND a.edregtime IS NOT NULL\n          AND a.edouttime IS NOT NULL\n          AND a.edouttime>a.edregtime\n          AND h.charttime>=a.edregtime\n          AND h.charttime<=a.edouttime\n    ) THEN 1 ELSE 0 END AS overlaps_admissions_ed_interval,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM valid_ed_transfers e\n        WHERE e.subject_id=h.subject_id\n          AND h.charttime>=e.intime\n          AND h.charttime<=e.outtime\n    ) THEN 1 ELSE 0 END AS overlaps_valid_ed_transfer,\n\n    CASE WHEN EXISTS (\n        SELECT 1\n        FROM valid_physical_transfers tr\n        WHERE tr.subject_id=h.subject_id\n          AND h.charttime>=tr.intime\n          AND h.charttime<=tr.outtime\n    ) THEN 1 ELSE 0 END AS overlaps_valid_physical_transfer\n\nFROM hist_same_time h\n')
    con.execute('\nCREATE TEMP TABLE hist_strict AS\nSELECT *\nFROM hist_classified\nWHERE has_labevent_hadm_id=0\n  AND overlaps_admission=0\n  AND overlaps_admissions_ed_interval=0\n  AND overlaps_valid_ed_transfer=0\n  AND overlaps_valid_physical_transfer=0\n')
    strict_baseline = con.execute('\nSELECT\n    t.stay_id,\n    COUNT(h.charttime) AS hist_strict_n,\n    MEDIAN(h.creatinine) AS hist_strict_median,\n    AVG(h.creatinine) AS hist_strict_mean_sensitivity,\n    ARG_MAX(h.creatinine,h.charttime) AS hist_strict_most_recent_sensitivity,\n    MIN(h.charttime) AS hist_strict_first_time,\n    MAX(h.charttime) AS hist_strict_last_time\nFROM target_lm t\nLEFT JOIN hist_strict h USING(stay_id)\nGROUP BY t.stay_id\n').df()
    assert_unique(strict_baseline, 'stay_id', 'strict baseline')
    baseline_qc = pd.DataFrame([{'target_stays': len(strict_baseline), 'strict_baseline_available': int(strict_baseline.hist_strict_median.notna().sum()), 'strict_baseline_availability_pct': float(100 * strict_baseline.hist_strict_median.notna().mean()), 'single_measurement': int(strict_baseline.hist_strict_n.eq(1).sum()), 'n_ge_2': int(strict_baseline.hist_strict_n.ge(2).sum())}])
    display_table(baseline_qc)
    wcsv(baseline_qc, '02_strict_historical_baseline_summary.csv')
    con.execute("\nCREATE TEMP TABLE cr_same_time AS\nSELECT\n    t.stay_id,t.intime,t.landmark_time,\n    c.charttime,\n    MAX(c.storetime) AS storetime,\n    AVG(c.creatinine) AS creatinine\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.intime,t.landmark_time,c.charttime\n")
    con.execute("\nCREATE TEMP TABLE cr_refs AS\nSELECT\n    *,\n    MIN(creatinine) OVER(\n      PARTITION BY stay_id ORDER BY charttime\n      RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n      AND INTERVAL '1 microsecond' PRECEDING\n    ) AS low48\nFROM cr_same_time\n")
    con.register('strict_baseline_df', strict_baseline)
    con.execute('CREATE TEMP TABLE strict_baseline_tbl AS SELECT * FROM strict_baseline_df')
    events = con.execute('\nSELECT\n    r.stay_id,r.charttime,r.creatinine,r.low48,\n    b.hist_strict_median,\n    b.hist_strict_mean_sensitivity,\n    b.hist_strict_most_recent_sensitivity\nFROM cr_refs r\nJOIN target_lm t USING(stay_id)\nLEFT JOIN strict_baseline_tbl b USING(stay_id)\nWHERE r.charttime>=t.intime\n  AND r.charttime<=t.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()

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
    events['stage_hist_strict_median'] = historical_stage(events, 'hist_strict_median')
    events['stage_hist_strict_mean_sensitivity'] = historical_stage(events, 'hist_strict_mean_sensitivity')
    events['stage_hist_strict_recent_sensitivity'] = historical_stage(events, 'hist_strict_most_recent_sensitivity')
    hist_stages = events.groupby('stay_id', as_index=False)[['stage_hist_strict_median', 'stage_hist_strict_mean_sensitivity', 'stage_hist_strict_recent_sensitivity']].max()
    strict_cr = premorbid_target[['stay_id']].merge(hist_stages, on='stay_id', how='left', validate='1:1').merge(strict_baseline, on='stay_id', how='left', validate='1:1').merge(premorbid_cr_saved[['stay_id', 'kdigo_creatinine_stage_rolling', 'creatinine_evaluable']], on='stay_id', how='left', validate='1:1')

    def rowmax(a, b):
        aa = pd.to_numeric(a, errors='coerce')
        bb = pd.to_numeric(b, errors='coerce')
        out = pd.concat([aa, bb], axis=1).max(axis=1, skipna=True)
        return out.where(aa.notna() | bb.notna())
    strict_cr['kdigo_creatinine_stage_strict_outpatient'] = rowmax(strict_cr.kdigo_creatinine_stage_rolling, strict_cr.stage_hist_strict_median)
    strict_cr['kdigo_creatinine_stage_historical_mean_sensitivity'] = rowmax(strict_cr.kdigo_creatinine_stage_rolling, strict_cr.stage_hist_strict_mean_sensitivity)
    strict_cr['kdigo_creatinine_stage_historical_recent_sensitivity'] = rowmax(strict_cr.kdigo_creatinine_stage_rolling, strict_cr.stage_hist_strict_recent_sensitivity)
    event_stays = set(events.stay_id.astype(int))
    hist_eval = strict_cr.hist_strict_median.notna() & strict_cr.stay_id.astype(int).isin(event_stays)
    strict_cr['creatinine_evaluable_strict_outpatient'] = (strict_cr.creatinine_evaluable.fillna(0).eq(1) | hist_eval).astype('int8')
    demoted = strict_cr.kdigo_creatinine_stage_rolling.fillna(-1) > strict_cr.kdigo_creatinine_stage_strict_outpatient.fillna(-1)
    if demoted.any():
        raise AssertionError('Combined historical-creatinine definition demotes the rolling creatinine stage.')
    strict_cr['new_positive_from_strict_history'] = (strict_cr.kdigo_creatinine_stage_rolling.fillna(0).lt(1) & strict_cr.kdigo_creatinine_stage_strict_outpatient.fillna(0).ge(1)).astype('int8')
    cr_qc = pd.DataFrame([{'rolling_rolling_creatinine_positive_target': int(strict_cr.kdigo_creatinine_stage_rolling.fillna(0).ge(1).sum()), 'strict_outpatient_creatinine_positive_target': int(strict_cr.kdigo_creatinine_stage_strict_outpatient.fillna(0).ge(1).sum()), 'new_positive_from_strict_history': int(strict_cr.new_positive_from_strict_history.sum()), 'mean_sensitivity_positive_target': int(strict_cr.kdigo_creatinine_stage_historical_mean_sensitivity.fillna(0).ge(1).sum()), 'recent_sensitivity_positive_target': int(strict_cr.kdigo_creatinine_stage_historical_recent_sensitivity.fillna(0).ge(1).sum()), 'rolling_stage_demoted': int(demoted.sum())}])
    display_table(cr_qc)
    wcsv(cr_qc, 'creatinine_summary.csv')
    cand = premorbid_target.merge(strict_cr[['stay_id', 'hist_strict_n', 'hist_strict_median', 'hist_strict_mean_sensitivity', 'hist_strict_most_recent_sensitivity', 'hist_strict_first_time', 'hist_strict_last_time', 'stage_hist_strict_median', 'stage_hist_strict_mean_sensitivity', 'stage_hist_strict_recent_sensitivity', 'kdigo_creatinine_stage_rolling', 'kdigo_creatinine_stage_strict_outpatient', 'kdigo_creatinine_stage_historical_mean_sensitivity', 'kdigo_creatinine_stage_historical_recent_sensitivity', 'creatinine_evaluable_strict_outpatient', 'new_positive_from_strict_history']], on='stay_id', how='left', validate='1:1')
    cr_stage = pd.to_numeric(cand['kdigo_creatinine_stage_strict_outpatient'], errors='coerce')
    uo_stage = pd.to_numeric(cand['urine_output_stage_reliability_adjusted'], errors='coerce')
    rrt_pos = cand['active_rrt_any_by_landmark'].fillna(0).eq(1)
    positive = cr_stage.fillna(0).ge(1) | uo_stage.fillna(0).ge(1) | rrt_pos
    combined_stage = pd.concat([cr_stage.fillna(-1), uo_stage.fillna(-1)], axis=1).max(axis=1)
    combined_stage = combined_stage.where(combined_stage >= 1, np.nan)
    combined_stage = combined_stage.mask(rrt_pos, 3)
    cr_eval = cand['creatinine_evaluable_strict_outpatient'].fillna(0).eq(1)
    uo_eval = cand['urine_output_absence_evaluable_reliability_adjusted'].fillna(0).eq(1)
    combined_status = np.where(positive, 'positive', np.where(cr_eval & uo_eval, 'negative', 'indeterminate'))
    cand['combined_stage_strict_outpatient'] = combined_stage
    cand['combined_status_strict_outpatient'] = combined_status
    qual = cand.loc[cand['combined_status_strict_outpatient'].eq('positive')].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
    sel = qual.drop_duplicates('subject_id', keep='first').copy()
    if len(sel) != EXPECTED_STRICT_N:
        raise AssertionError(f'Strict-outpatient cohort N mismatch: expected {EXPECTED_STRICT_N}, got {len(sel)}')
    if sel['subject_id'].duplicated().any():
        raise AssertionError('Duplicate subject in selected cohort.')
    if not sel['prior_documented_hf'].eq(1).all():
        raise AssertionError('Selected cohort contains non-prior-HF stay.')
    if not sel['chronic_krt_definite'].fillna(0).eq(0).all():
        raise AssertionError('Selected cohort contains chronic-KRT stay.')
    if not sel['combined_status_strict_outpatient'].eq('positive').all():
        raise AssertionError('Selected cohort contains non-positive phenotype.')
    bad_uo = sel[sel['uo_unreliable_by_landmark'].fillna(0).eq(1) & sel['kdigo_creatinine_stage_strict_outpatient'].fillna(0).lt(1) & sel['active_rrt_any_by_landmark'].fillna(0).ne(1)]
    if len(bad_uo):
        raise AssertionError('Selected cohort contains a stay relying only on unreliable UO evidence.')
    mem_sha = membership_sha(sel)
    reference_subjects = set(reference_idx['subject_id'].astype(int))
    strict_outpatient_subjects = set(sel['subject_id'].astype(int))
    common = reference_subjects & strict_outpatient_subjects
    reference_map = dict(zip(reference_idx['subject_id'].astype(int), reference_idx['stay_id'].astype(int)))
    strict_outpatient_map = dict(zip(sel['subject_id'].astype(int), sel['stay_id'].astype(int)))
    replaced_vs_reference = sum((reference_map[s] != strict_outpatient_map[s] for s in common))
    membership_qc = pd.DataFrame([{'selected_subjects': len(sel), 'membership_sha256': mem_sha, 'added_subjects_vs_reference': len(strict_outpatient_subjects - reference_subjects), 'dropped_subjects_vs_reference': len(reference_subjects - strict_outpatient_subjects), 'common_subjects_vs_reference': len(common), 'different_selected_stay_vs_reference': int(replaced_vs_reference), 'selected_relying_only_on_unreliable_uo': len(bad_uo)}])
    display_table(membership_qc)
    print('AUDITED MEMBERSHIP: PASS')
    index_preoutcome = sel.copy()
    if 'kdigo_creatinine_stage' in index_preoutcome.columns:
        index_preoutcome['kdigo_creatinine_stage_historical_proxy'] = index_preoutcome['kdigo_creatinine_stage']
    if 'creatinine_evaluable' in index_preoutcome.columns:
        index_preoutcome['creatinine_evaluable_historical_proxy'] = index_preoutcome['creatinine_evaluable']
    if 'combined_stage_ibw_anyrrt' in index_preoutcome.columns:
        index_preoutcome['combined_stage_historical_proxy'] = index_preoutcome['combined_stage_ibw_anyrrt']
    if 'combined_status_ibw_anyrrt' in index_preoutcome.columns:
        index_preoutcome['combined_status_historical_proxy'] = index_preoutcome['combined_status_ibw_anyrrt']
    index_preoutcome['kdigo_creatinine_stage'] = index_preoutcome['kdigo_creatinine_stage_strict_outpatient']
    index_preoutcome['creatinine_evaluable'] = index_preoutcome['creatinine_evaluable_strict_outpatient']
    index_preoutcome['positive_stage_ibw'] = index_preoutcome['urine_output_stage_reliability_adjusted']
    index_preoutcome['uo_positive_ibw'] = index_preoutcome['urine_output_positive_reliability_adjusted']
    index_preoutcome['absence_evaluable_ibw'] = index_preoutcome['urine_output_absence_evaluable_reliability_adjusted']
    index_preoutcome['combined_stage_ibw_anyrrt'] = index_preoutcome['combined_stage_strict_outpatient']
    index_preoutcome['combined_status_ibw_anyrrt'] = index_preoutcome['combined_status_strict_outpatient']
    index_preoutcome['creatinine_method'] = 'max(rolling_48h_7d, strict_outpatient_proxy_median_7_365d)'
    index_preoutcome['historical_creatinine_method'] = 'median SCr >7-365d pre-admission; hadm_id NULL; no admission/ED/valid physical-transfer overlap'
    index_preoutcome['urine_output_method'] = 'event_gap6_full + IBW'
    index_preoutcome['urine_output_reliability_rule'] = 'UO unevaluable when ureteral stent or GU irrigation available by landmark'
    index_preoutcome['heart_failure_method'] = 'prior completed-admission direct HF'
    index_preoutcome['chronic_krt_method'] = 'chronic_krt_definite'
    index_preoutcome['rrt_method'] = 'active acute RRT by landmark'
    drop_internal = [c for c in index_preoutcome.columns if c.startswith('_')]
    index_preoutcome = index_preoutcome.drop(columns=drop_internal, errors='ignore')
    bad = outcome_like_columns(index_preoutcome)
    if bad:
        index_preoutcome = index_preoutcome.drop(columns=bad)
    assert_outcome_blind(index_preoutcome, 'strict_outpatient PRE-OUTCOME INDEX')
    index_preoutcome = index_preoutcome.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').reset_index(drop=True)
    assert_unique(index_preoutcome, 'subject_id', 'pre-outcome subject index')
    assert_unique(index_preoutcome, 'stay_id', 'pre-outcome stay index')
    if len(index_preoutcome) != EXPECTED_STRICT_N:
        raise AssertionError('Pre-outcome index N changed before persistence.')
    P_STRICT_CR = OUT / 'strict_outpatient_creatinine.parquet'
    P_STRICT_TARGET = OUT / 'strict_outpatient_target.parquet'
    P_STRICT_INDEX = OUT / 'strict_outpatient_cohort_preoutcome.parquet'
    strict_cr.to_parquet(P_STRICT_CR, index=False)
    cand.to_parquet(P_STRICT_TARGET, index=False)
    index_preoutcome.to_parquet(P_STRICT_INDEX, index=False)

    # Re-read the persisted pre-outcome cohort and verify its scientific invariants.
    persisted = pd.read_parquet(P_STRICT_INDEX)
    assert_outcome_blind(persisted, 'strict-outpatient pre-outcome cohort')
    assert_unique(persisted, 'subject_id', 'strict-outpatient subject index')
    assert_unique(persisted, 'stay_id', 'strict-outpatient stay index')
    if len(persisted) != EXPECTED_STRICT_N:
        raise AssertionError(f'Strict-outpatient cohort N={len(persisted)}, expected {EXPECTED_STRICT_N}.')

    reference_subjects = set(reference_idx.subject_id.astype(int))
    strict_subjects = set(persisted.subject_id.astype(int))
    common_subjects = reference_subjects & strict_subjects
    reference_map = dict(zip(reference_idx.subject_id.astype(int), reference_idx.stay_id.astype(int)))
    strict_map = dict(zip(persisted.subject_id.astype(int), persisted.stay_id.astype(int)))
    replaced_vs_reference = sum(reference_map[s] != strict_map[s] for s in common_subjects)
    cohort_qc = pd.DataFrame([{
        'strict_outpatient_subjects': len(persisted),
        'added_subjects_vs_reference': len(strict_subjects - reference_subjects),
        'dropped_subjects_vs_reference': len(reference_subjects - strict_subjects),
        'different_selected_stay_vs_reference': int(replaced_vs_reference),
        'outcome_like_columns': len(outcome_like_columns(persisted)),
    }])
    display_table(cohort_qc)
    wcsv(cohort_qc, 'strict_outpatient_cohort_summary.csv')

    manifest = {
        'status': 'PASS',
        'n': int(len(persisted)),
        'outcome_attached': False,
        'definition': {
            'landmark': '+24 h ICU',
            'heart_failure': 'prior completed-admission direct heart-failure documentation',
            'chronic_krt': 'prespecified chronic-KRT exclusion',
            'creatinine': 'maximum of rolling 48 h / 7 d KDIGO stage and strict outpatient-proxy median creatinine from >7 to 365 days before admission',
            'historical_outpatient_proxy': 'hadm_id NULL; no hospital-admission, emergency-department, or valid physical-transfer overlap',
            'urine_output': 'event_gap6_full + ideal body weight',
            'urine_output_reliability': 'ureteral-stent or GU-irrigation evidence by landmark makes the urine-output criterion unevaluable',
            'rrt': 'active acute RRT by landmark contributes KDIGO Stage 3 after chronic-KRT exclusion',
            'episode_selection': 'first qualifying ICU stay per subject',
        },
        'hard_guards': {
            'outcome_blind': True,
            'prior_hf_all_selected': True,
            'chronic_krt_all_excluded': True,
            'unreliable_urine_output_cannot_qualify_alone': True,
            'rolling_creatinine_never_demoted': True,
            'one_selected_icu_stay_per_subject': True,
        },
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    (OUT / 'strict_outpatient_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('STRICT OUTPATIENT CREATININE: PASS')
    print('Strict-outpatient cohort N:', len(persisted))


if __name__ == "__main__":
    main()
