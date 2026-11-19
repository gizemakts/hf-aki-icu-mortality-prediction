#!/usr/bin/env python3
"""10 Post Transplant Krt.

Apply the temporal post-transplant chronic-KRT rule.

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
    try:
        import duckdb
    except Exception as e:
        raise ImportError('duckdb is required; install project requirements before running.') from e
    EXPECTED_N = 3422
    EXPECTED_TARGET_N = 9972
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT))).resolve()
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', str(_RAW_DATA_ROOT))).resolve()
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f'MIMIC-IV root not found: {MIMIC_ROOT}')
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    STRICT_DIR = PHENOTYPE_DIR / 'historical_creatinine' / 'strict_outpatient'
    COMBINED_DIR = PHENOTYPE_DIR / 'combined'
    OUT = PHENOTYPE_DIR / 'post_transplant'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'post_transplant'
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    def req(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def membership_sha(df):
        z = df[['subject_id', 'stay_id']].copy()
        z['subject_id'] = z.subject_id.astype('int64')
        z['stay_id'] = z.stay_id.astype('int64')
        z = z.sort_values(['subject_id', 'stay_id'])
        payload = 'subject_id,stay_id\n' + ''.join((f'{int(r.subject_id)},{int(r.stay_id)}\n' for r in z.itertuples(index=False)))
        return hashlib.sha256(payload.encode()).hexdigest()

    def hard_guard(idx, target=None):
        if idx.subject_id.duplicated().any() or idx.stay_id.duplicated().any():
            raise AssertionError('Strict-outpatient cohort index is not unique.')
        if len(idx) != EXPECTED_N:
            raise AssertionError(f'Strict-outpatient cohort N mismatch: {len(idx)}')
        if target is not None and len(target) != EXPECTED_TARGET_N:
            raise AssertionError(f'Strict-outpatient target N mismatch: {len(target)}')

    def first_positive(df, col='combined_positive'):
        q = df.loc[df[col]].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
        return q.drop_duplicates('subject_id', keep='first').copy()

    def membership_compare(canon, alt, label):
        cm = dict(zip(canon.subject_id.astype(int), canon.stay_id.astype(int)))
        am = dict(zip(alt.subject_id.astype(int), alt.stay_id.astype(int)))
        cs, as_ = (set(cm), set(am))
        common = cs & as_
        repl = sum((cm[s] != am[s] for s in common))
        return {'variant': label, 'subjects': len(alt), 'subject_added_vs_strict_outpatient': len(as_ - cs), 'subject_dropped_vs_strict_outpatient': len(cs - as_), 'common_subjects': len(common), 'same_subject_replaced_stay': int(repl), 'same_subject_same_stay': int(len(common) - repl), 'stay_symmetric_difference': len(set(cm.values()) ^ set(am.values()))}

    def wcsv(df, out, name):
        p = out / name
        df.to_csv(p, index=False)
        return p

    def wjson(obj, out, name):
        p = out / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT:', MIMIC_ROOT)
    print('OUTPUT:', OUT)
    P_TARGET = req(STRICT_DIR / 'strict_outpatient_target.parquet')
    P_INDEX = req(STRICT_DIR / 'strict_outpatient_cohort_preoutcome.parquet')
    P_LM = req(COHORT_DIR / 'landmark_population.parquet')
    P_PH = req(COHORT_DIR / 'heart_failure_krt_phenotypes.parquet')
    P_COMB = req(COMBINED_DIR / 'combined_kdigo_variants.parquet')
    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    DX = req(MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz')
    PROC = req(MIMIC_ROOT / 'hosp' / 'procedures_icd.csv.gz')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    TR = req(MIMIC_ROOT / 'hosp' / 'transfers.csv.gz')
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    strict_target = pd.read_parquet(P_TARGET)
    strict_idx = pd.read_parquet(P_INDEX, columns=['subject_id', 'stay_id', 'intime'])
    hard_guard(strict_idx, strict_target)
    lm = pd.read_parquet(P_LM, columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time'])
    ph = pd.read_parquet(P_PH)
    comb = pd.read_parquet(P_COMB)
    primary = comb.loc[comb.variant_id.eq('event_gap6_full')].copy()
    if primary.stay_id.duplicated().any():
        raise AssertionError('Primary event_gap6_full is not unique by stay_id.')
    need_ph = ['stay_id', 'prior_documented_hf', 'prior_eskd_history', 'prior_dialysis_dependence_history', 'chronic_krt_prior_evidence', 'current_eskd_code', 'current_dialysis_dependence_code', 'current_chronic_krt_state_code', 'current_dialysis_encounter_code']
    miss = [c for c in need_ph if c not in ph.columns]
    if miss:
        raise KeyError(f'phenotype table missing: {miss}')
    need_primary = ['stay_id', 'kdigo_creatinine_stage', 'creatinine_evaluable', 'positive_stage_ibw', 'absence_evaluable_ibw', 'active_rrt_any_by_landmark']
    miss = [c for c in need_primary if c not in primary.columns]
    if miss:
        raise KeyError(f'primary combined missing: {miss}')
    pool = lm.merge(ph[need_ph], on='stay_id', how='left', validate='1:1').merge(primary[need_primary], on='stay_id', how='left', validate='1:1')
    pool = pool.loc[pool.prior_documented_hf.eq(1) & pool.chronic_krt_prior_evidence.fillna(0).eq(1)].copy()
    print('Strict-outpatient cohort hard guards: PASS')
    print('Prior-HF stays excluded by prior-ever KRT state:', len(pool))
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('pool_df', pool[['subject_id', 'hadm_id', 'stay_id', 'admittime']])
    con.execute('CREATE TEMP TABLE pool AS SELECT * FROM pool_df')
    con.execute(f"\nCREATE TEMP TABLE admissions AS\nSELECT subject_id,hadm_id,admittime,dischtime,edregtime,edouttime\nFROM read_csv_auto('{qpath(ADM)}',header=true,sample_size=200000)\n")
    con.execute(f"\nCREATE TEMP TABLE dx AS\nSELECT subject_id,hadm_id,icd_version,\n       upper(replace(trim(icd_code),'.','')) AS code\nFROM read_csv_auto('{qpath(DX)}',header=true,sample_size=200000)\n")
    con.execute(f"\nCREATE TEMP TABLE proc AS\nSELECT subject_id,hadm_id,icd_version,chartdate,\n       upper(replace(trim(icd_code),'.','')) AS code\nFROM read_csv_auto('{qpath(PROC)}',header=true,sample_size=200000)\n")
    hist_dx = con.execute("\nWITH dxa AS (\n SELECT subject_id,hadm_id,\n   MAX(CASE WHEN (icd_version=9 AND code='5856') OR\n                 (icd_version=10 AND code='N186') THEN 1 ELSE 0 END) kf,\n   MAX(CASE WHEN (icd_version=9 AND code='V4511') OR\n                 (icd_version=10 AND code='Z992') THEN 1 ELSE 0 END) dep,\n   MAX(CASE WHEN (icd_version=9 AND code='V420') OR\n                 (icd_version=10 AND code='Z940') THEN 1 ELSE 0 END) tx_status,\n   MAX(CASE WHEN icd_version=10 AND code='T8612' THEN 1 ELSE 0 END) tx_failure\n FROM dx GROUP BY subject_id,hadm_id\n)\nSELECT p.stay_id,\n MAX(CASE WHEN d.kf=1 OR d.dep=1 THEN a.dischtime END) latest_prior_krt_state_time,\n MAX(CASE WHEN d.tx_status=1 THEN a.dischtime END) latest_prior_tx_status_time,\n MAX(CASE WHEN d.tx_failure=1 THEN a.dischtime END) latest_prior_tx_failure_time\nFROM pool p\nJOIN admissions a ON a.subject_id=p.subject_id\nJOIN dxa d ON d.subject_id=a.subject_id AND d.hadm_id=a.hadm_id\nWHERE a.hadm_id<>p.hadm_id\n  AND a.dischtime IS NOT NULL AND a.dischtime<=p.admittime\nGROUP BY p.stay_id\n").df()
    hist_proc = con.execute("\nSELECT p.stay_id,\n MAX(CAST(pr.chartdate AS TIMESTAMP)) latest_prior_tx_proc_time\nFROM pool p\nJOIN proc pr ON pr.subject_id=p.subject_id\nJOIN admissions a ON a.subject_id=pr.subject_id AND a.hadm_id=pr.hadm_id\nWHERE pr.hadm_id<>p.hadm_id\n  AND pr.chartdate IS NOT NULL\n  AND CAST(pr.chartdate AS TIMESTAMP)<p.admittime\n  AND a.dischtime IS NOT NULL AND a.dischtime<=p.admittime\n  AND ((pr.icd_version=9 AND pr.code='5569')\n       OR (pr.icd_version=10 AND pr.code LIKE '0TY%'))\nGROUP BY p.stay_id\n").df()
    current_tx = con.execute("\nSELECT p.stay_id,\n MAX(CASE WHEN (d.icd_version=9 AND d.code='V420') OR\n               (d.icd_version=10 AND d.code='Z940') THEN 1 ELSE 0 END)\n      AS current_tx_status,\n MAX(CASE WHEN d.icd_version=10 AND d.code='T8612' THEN 1 ELSE 0 END)\n      AS current_tx_failure\nFROM pool p\nLEFT JOIN dx d ON d.subject_id=p.subject_id AND d.hadm_id=p.hadm_id\nGROUP BY p.stay_id\n").df()
    cand = pool.merge(hist_dx, on='stay_id', how='left', validate='1:1').merge(hist_proc, on='stay_id', how='left', validate='1:1').merge(current_tx, on='stay_id', how='left', validate='1:1')
    cand['latest_prior_tx_time'] = pd.concat([pd.to_datetime(cand.latest_prior_tx_status_time), pd.to_datetime(cand.latest_prior_tx_proc_time)], axis=1).max(axis=1)
    cand['tx_after_krt'] = cand.latest_prior_tx_time.notna() & cand.latest_prior_krt_state_time.notna() & (pd.to_datetime(cand.latest_prior_tx_time) > pd.to_datetime(cand.latest_prior_krt_state_time))
    cand['failure_after_tx'] = cand.latest_prior_tx_failure_time.notna() & cand.latest_prior_tx_time.notna() & (pd.to_datetime(cand.latest_prior_tx_failure_time) >= pd.to_datetime(cand.latest_prior_tx_time))
    cand['current_tx_status'] = cand.current_tx_status.fillna(0).astype(int)
    cand['current_tx_failure'] = cand.current_tx_failure.fillna(0).astype(int)
    base_temporal = cand.tx_after_krt & ~cand.failure_after_tx.fillna(False) & cand.current_chronic_krt_state_code.fillna(0).eq(0) & cand.current_tx_failure.eq(0)
    cand['post_transplant_high_specificity_status_no_rrt'] = base_temporal & cand.current_tx_status.eq(1) & cand.active_rrt_any_by_landmark.fillna(0).eq(0)
    cand['post_transplant_temporal_no_rrt'] = base_temporal & cand.active_rrt_any_by_landmark.fillna(0).eq(0)
    cand['post_transplant_temporal_allow_acute_rrt'] = base_temporal
    flags = pd.DataFrame([{'prior_krt_pool_stays': len(cand), 'tx_after_krt_stays': int(cand.tx_after_krt.sum()), 'high_specificity_status_no_rrt_stays': int(cand.post_transplant_high_specificity_status_no_rrt.sum()), 'temporal_no_rrt_stays': int(cand.post_transplant_temporal_no_rrt.sum()), 'temporal_allow_acute_rrt_stays': int(cand.post_transplant_temporal_allow_acute_rrt.sum())}])
    display_table(flags)
    wcsv(flags, OUT, 'post_transplant_variant_counts.csv')
    work = cand.loc[cand.post_transplant_temporal_allow_acute_rrt].copy()
    print('Stays requiring strict-outpatient renal reconstruction:', len(work))
    con2 = duckdb.connect(database=':memory:')
    con2.execute('PRAGMA threads=4')
    con2.register('work_df', work[['subject_id', 'stay_id', 'admittime', 'intime', 'landmark_time']])
    con2.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM work_df')
    con2.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT subject_id,hadm_id,charttime,storetime,valuenum AS creatinine\nFROM read_csv_auto('{qpath(LAB)}',header=true,sample_size=200000)\nWHERE itemid=50912 AND valuenum>0 AND valuenum<=150\n  AND charttime IS NOT NULL AND storetime IS NOT NULL\n")
    con2.execute(f"\nCREATE TEMP TABLE admissions AS\nSELECT subject_id,hadm_id,admittime,dischtime,edregtime,edouttime\nFROM read_csv_auto('{qpath(ADM)}',header=true,sample_size=200000)\n")
    con2.execute(f"\nCREATE TEMP TABLE transfers_all AS\nSELECT subject_id,hadm_id,eventtype,careunit,intime,outtime\nFROM read_csv_auto('{qpath(TR)}',header=true,sample_size=200000)\n")
    con2.execute("\nCREATE TEMP TABLE valid_physical_transfers AS\nSELECT subject_id,hadm_id,LOWER(COALESCE(eventtype,'')) eventtype,\n       careunit,intime,outtime\nFROM transfers_all\nWHERE intime IS NOT NULL AND outtime IS NOT NULL AND outtime>intime\n  AND careunit IS NOT NULL\n  AND LOWER(COALESCE(eventtype,'')) IN ('ed','admit','transfer')\n")
    con2.execute("\nCREATE TEMP TABLE valid_ed_transfers AS\nSELECT * FROM valid_physical_transfers\nWHERE eventtype='ed' OR LOWER(careunit) LIKE '%emergency%'\n")
    con2.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime,\n       MAX(c.storetime) storetime,AVG(c.creatinine) creatinine,\n       MAX(CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END) has_labevent_hadm_id\nFROM target_lm t JOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    con2.execute('\nCREATE TEMP TABLE hist_classified AS\nSELECT h.*,\n CASE WHEN EXISTS(\n   SELECT 1 FROM admissions a\n   WHERE a.subject_id=h.subject_id\n     AND a.admittime IS NOT NULL AND a.dischtime IS NOT NULL\n     AND h.charttime>=a.admittime AND h.charttime<=a.dischtime\n ) THEN 1 ELSE 0 END overlaps_admission,\n CASE WHEN EXISTS(\n   SELECT 1 FROM admissions a\n   WHERE a.subject_id=h.subject_id\n     AND a.edregtime IS NOT NULL AND a.edouttime IS NOT NULL\n     AND a.edouttime>a.edregtime\n     AND h.charttime>=a.edregtime AND h.charttime<=a.edouttime\n ) THEN 1 ELSE 0 END overlaps_admissions_ed_interval,\n CASE WHEN EXISTS(\n   SELECT 1 FROM valid_ed_transfers e\n   WHERE e.subject_id=h.subject_id\n     AND h.charttime>=e.intime AND h.charttime<=e.outtime\n ) THEN 1 ELSE 0 END overlaps_valid_ed_transfer,\n CASE WHEN EXISTS(\n   SELECT 1 FROM valid_physical_transfers tr\n   WHERE tr.subject_id=h.subject_id\n     AND h.charttime>=tr.intime AND h.charttime<=tr.outtime\n ) THEN 1 ELSE 0 END overlaps_valid_physical_transfer\nFROM hist_same_time h\n')
    con2.execute('\nCREATE TEMP TABLE hist_strict AS\nSELECT * FROM hist_classified\nWHERE has_labevent_hadm_id=0 AND overlaps_admission=0\n  AND overlaps_admissions_ed_interval=0\n  AND overlaps_valid_ed_transfer=0 AND overlaps_valid_physical_transfer=0\n')
    baseline = con2.execute('\nSELECT t.stay_id,COUNT(h.charttime) hist_strict_n,\n       MEDIAN(h.creatinine) hist_strict_median\nFROM target_lm t LEFT JOIN hist_strict h USING(stay_id)\nGROUP BY t.stay_id\n').df()
    con2.execute("\nCREATE TEMP TABLE cr_same_time AS\nSELECT t.stay_id,t.intime,t.landmark_time,c.charttime,\n       MAX(c.storetime) storetime,AVG(c.creatinine) creatinine\nFROM target_lm t JOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.intime,t.landmark_time,c.charttime\n")
    con2.execute("\nCREATE TEMP TABLE cr_refs AS\nSELECT *,\n MIN(creatinine) OVER(\n   PARTITION BY stay_id ORDER BY charttime\n   RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n   AND INTERVAL '1 microsecond' PRECEDING\n ) low48\nFROM cr_same_time\n")
    con2.register('baseline_df', baseline)
    con2.execute('CREATE TEMP TABLE baseline_tbl AS SELECT * FROM baseline_df')
    events = con2.execute('\nSELECT r.stay_id,r.charttime,r.creatinine,r.low48,b.hist_strict_median\nFROM cr_refs r\nJOIN target_lm t USING(stay_id)\nLEFT JOIN baseline_tbl b USING(stay_id)\nWHERE r.charttime>=t.intime AND r.charttime<=t.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()

    def historical_stage(df):
        cr = pd.to_numeric(df.creatinine, errors='coerce')
        b = pd.to_numeric(df.hist_strict_median, errors='coerce')
        low48 = pd.to_numeric(df.low48, errors='coerce')
        has = b.notna()
        r15 = has & cr.ge(1.5 * b)
        r20 = has & cr.ge(2 * b)
        r30 = has & cr.ge(3 * b)
        d03 = low48.notna() & cr.ge(low48 + 0.3)
        out = np.select([r30, cr.ge(4.0) & (r15 | d03), r20, r15], [3.0, 3.0, 2.0, 1.0], default=0.0)
        s = pd.Series(out, index=df.index, dtype='float64')
        s.loc[~has] = np.nan
        return s
    events['stage_hist_strict'] = historical_stage(events)
    hs = events.groupby('stay_id', as_index=False).stage_hist_strict.max()
    renal = work[['stay_id', 'kdigo_creatinine_stage', 'creatinine_evaluable']].merge(baseline, on='stay_id', how='left', validate='1:1').merge(hs, on='stay_id', how='left', validate='1:1')
    a = pd.to_numeric(renal.kdigo_creatinine_stage, errors='coerce')
    b = pd.to_numeric(renal.stage_hist_strict, errors='coerce')
    renal['kdigo_creatinine_stage_post_transplant'] = pd.concat([a, b], axis=1).max(axis=1, skipna=True)
    renal['kdigo_creatinine_stage_post_transplant'] = renal.kdigo_creatinine_stage_post_transplant.where(a.notna() | b.notna())
    event_stays = set(events.stay_id.astype(int))
    renal['creatinine_evaluable_post_transplant'] = (renal.creatinine_evaluable.fillna(0).eq(1) | renal.hist_strict_median.notna() & renal.stay_id.astype(int).isin(event_stays)).astype(int)
    display_table(renal[['hist_strict_n', 'hist_strict_median', 'kdigo_creatinine_stage_post_transplant']].describe())
    con3 = duckdb.connect(database=':memory:')
    con3.register('work_df', work[['stay_id', 'intime', 'landmark_time']])
    con3.execute('CREATE TEMP TABLE work AS SELECT * FROM work_df')
    con3.execute(f"\nCREATE VIEW oe AS\nSELECT stay_id,itemid,charttime,storetime\nFROM read_csv_auto('{qpath(OE)}',header=true,sample_size=200000)\nWHERE itemid IN (226557,226558,227488,227489)\n")
    rel = con3.execute('\nSELECT w.stay_id,\n MAX(CASE WHEN o.itemid IN (226557,226558) THEN 1 ELSE 0 END) has_stent,\n MAX(CASE WHEN o.itemid IN (227488,227489) THEN 1 ELSE 0 END) has_gu_irrigation\nFROM work w LEFT JOIN oe o\n ON o.stay_id=w.stay_id\n AND o.charttime>=w.intime AND o.charttime<=w.landmark_time\n AND o.storetime IS NOT NULL AND o.storetime<=w.landmark_time\nGROUP BY w.stay_id\n').df()
    for c in ['has_stent', 'has_gu_irrigation']:
        rel[c] = rel[c].fillna(0).astype(int)
    rel['uo_unreliable_by_landmark'] = ((rel.has_stent == 1) | (rel.has_gu_irrigation == 1)).astype(int)
    work2 = work.merge(renal[['stay_id', 'hist_strict_n', 'hist_strict_median', 'kdigo_creatinine_stage_post_transplant', 'creatinine_evaluable_post_transplant']], on='stay_id', how='left', validate='1:1').merge(rel, on='stay_id', how='left', validate='1:1')
    reliable = work2.uo_unreliable_by_landmark.fillna(0).eq(0)
    work2['urine_output_stage_post_transplant'] = pd.to_numeric(work2.positive_stage_ibw, errors='coerce').where(reliable)
    work2['urine_output_evaluable_post_transplant'] = (work2.absence_evaluable_ibw.fillna(0).eq(1) & reliable).astype(int)
    cr = pd.to_numeric(work2.kdigo_creatinine_stage_post_transplant, errors='coerce')
    uo = pd.to_numeric(work2.urine_output_stage_post_transplant, errors='coerce')
    rrt = work2.active_rrt_any_by_landmark.fillna(0).eq(1)
    work2['aki_positive_post_transplant'] = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
    work2['kdigo_stage_post_transplant'] = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
    work2['kdigo_stage_post_transplant'] = work2.kdigo_stage_post_transplant.where(work2.kdigo_stage_post_transplant >= 1, np.nan).mask(rrt, 3)
    bad = work2[work2.uo_unreliable_by_landmark.eq(1) & cr.fillna(0).lt(1) & ~rrt & work2.aki_positive_post_transplant]
    if len(bad):
        raise AssertionError('Override alternative relies only on unreliable UO.')
    wcsv(work2, OUT, 'post_transplant_stay_reconstruction.csv')
    print('Strict-outpatient reconstruction complete.')
    basepos = strict_target.loc[strict_target.combined_status_strict_outpatient.eq('positive'), ['subject_id', 'stay_id', 'intime']].copy()
    basepos['source'] = 'strict_outpatient_target'

    def verification(flag_col, label):
        c = work2.loc[work2[flag_col].fillna(False) & work2.aki_positive_post_transplant, ['subject_id', 'stay_id', 'intime']].copy()
        c['source'] = 'post_transplant_override'
        allpos = pd.concat([basepos, c], ignore_index=True)
        sel = allpos.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').drop_duplicates('subject_id', keep='first')
        cmp = membership_compare(strict_idx, sel, label)
        cmp['override_positive_stays'] = len(c)
        cmp['override_positive_subjects'] = c.subject_id.nunique()
        cmp['selected_from_override'] = int(sel.source.eq('post_transplant_override').sum())
        return (sel, cmp)
    variants = [('post_transplant_high_specificity_status_no_rrt', 'post_transplant_high_specificity_status_no_rrt'), ('post_transplant_temporal_no_rrt', 'post_transplant_temporal_no_rrt'), ('post_transplant_temporal_allow_acute_rrt', 'post_transplant_temporal_allow_acute_rrt')]
    rows = []
    details = []
    for col, label in variants:
        sel, cmp = verification(col, label)
        rows.append(cmp)
        d = strict_idx[['subject_id', 'stay_id']].rename(columns={'stay_id': 'strict_outpatient_stay_id'}).merge(sel[['subject_id', 'stay_id', 'source']].rename(columns={'stay_id': 'alternative_stay_id', 'source': 'alternative_source'}), on='subject_id', how='outer', indicator=True)
        d['variant'] = label
        details.append(d)
    summary = pd.DataFrame(rows)
    display_table(summary)
    wcsv(summary, OUT, 'post_transplant_cohort_effect.csv')
    wcsv(pd.concat(details, ignore_index=True), OUT, 'post_transplant_membership_detail.csv')
    review_cols = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'latest_prior_krt_state_time', 'latest_prior_tx_time', 'current_tx_status', 'current_tx_failure', 'current_eskd_code', 'current_dialysis_dependence_code', 'current_chronic_krt_state_code', 'current_dialysis_encounter_code', 'active_rrt_any_by_landmark', 'hist_strict_n', 'hist_strict_median', 'kdigo_creatinine_stage_post_transplant', 'urine_output_stage_post_transplant', 'uo_unreliable_by_landmark', 'aki_positive_post_transplant', 'post_transplant_high_specificity_status_no_rrt', 'post_transplant_temporal_no_rrt', 'post_transplant_temporal_allow_acute_rrt']
    wcsv(work2[review_cols], OUT, 'post_transplant_review_detail.csv')
    manifest = {
        'status': 'PASS',
        'outcome_blind': True,
        'strict_outpatient_cohort_n': len(strict_idx),
        'purpose': 'evaluate whether prior KRT history over-excludes patients after kidney transplant',
        'variants': [
            'post_transplant_high_specificity_status_no_rrt',
            'post_transplant_temporal_no_rrt',
            'post_transplant_temporal_allow_acute_rrt',
        ],
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    wjson(manifest, OUT, 'post_transplant_manifest.json')
    print('POST-TRANSPLANT KRT EVALUATION: PASS')


if __name__ == "__main__":
    main()
