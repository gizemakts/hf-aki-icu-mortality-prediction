#!/usr/bin/env python3
"""15 Uo Kdigo Adjudication.

Perform outcome-blind UO/KDIGO adjudication.

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
from helpers._runtime import logical_source_path


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
        raise ImportError('duckdb is required; install the project requirements before running.') from e
    EXPECTED_STRICT_N = 3422
    EXPECTED_STRICT_TARGET_N = 9972
    EXPECTED_REFERENCE_N = 2972
    EXPECTED_REFERENCE_TARGET_N = 10083
    PROHIBITED_TOKENS = ('deathtime', 'hospital_expire', 'expire_flag', 'mortality', 'death', 'outcome', 'survival', 'auc', 'auroc', 'calibration', 'brier', 'cindex', 'c_index')

    PROJECT_ROOT = _REPOSITORY_ROOT
    MIMIC_ROOT = _RAW_DATA_ROOT
    required_raw = [
        MIMIC_ROOT / 'hosp' / 'admissions.csv.gz',
        MIMIC_ROOT / 'hosp' / 'labevents.csv.gz',
        MIMIC_ROOT / 'hosp' / 'transfers.csv.gz',
        MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz',
        MIMIC_ROOT / 'hosp' / 'procedures_icd.csv.gz',
        MIMIC_ROOT / 'icu' / 'outputevents.csv.gz',
        MIMIC_ROOT / 'icu' / 'inputevents.csv.gz',
        MIMIC_ROOT / 'icu' / 'procedureevents.csv.gz',
        MIMIC_ROOT / 'icu' / 'chartevents.csv.gz',
    ]
    missing_raw = [str(x) for x in required_raw if not x.is_file()]
    if missing_raw:
        raise FileNotFoundError(f'Missing required raw MIMIC files: {missing_raw}')

    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    STRICT_DIR = PHENOTYPE_DIR / 'historical_creatinine' / 'strict_outpatient'
    CONSECUTIVE_REFERENCE_DIR = PHENOTYPE_DIR / 'consecutive_urine_output_reference'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    POST_TRANSPLANT_DIR = PHENOTYPE_DIR / 'post_transplant'
    PERSISTENT_OLIGURIA_DIR = PHENOTYPE_DIR / 'persistent_oliguria'
    AUDIT = PHENOTYPE_DIR / 'adjudication'
    AUDIT.mkdir(parents=True, exist_ok=True)
    POLICY_PATH = AUDIT / 'adjudication_policy.json'

    def req(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    policy = json.loads(req(POLICY_PATH).read_text(encoding='utf-8'))
    if policy.get('status') != 'PASS' or policy.get('outcome_blind_adjudication') is not True:
        raise AssertionError('Current adjudication policy did not pass its outcome-blind guard.')

    INPUT_ACCESS = []

    def guard_column_names(cols, label):
        bad = [c for c in cols if any((t in str(c).lower() for t in PROHIBITED_TOKENS))]
        if bad:
            raise AssertionError(f'PROHIBITED outcome/model columns requested in {label}: {bad}')

    def record_access(path, columns, label):
        guard_column_names(columns, label)
        INPUT_ACCESS.append({'label': label, 'path': logical_source_path(path), 'columns': '|'.join(map(str, columns))})

    def read_parquet_allow(path, columns, label):
        path = req(path)
        columns = list(columns)
        record_access(path, columns, label)
        df = pd.read_parquet(path, columns=columns)
        guard_column_names(df.columns, label)
        return df

    def read_csv_allow(path, usecols, label, **kwargs):
        path = req(path)
        usecols = list(usecols)
        record_access(path, usecols, label)
        df = pd.read_csv(path, usecols=usecols, **kwargs)
        guard_column_names(df.columns, label)
        return df

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def membership_sha(df):
        z = df[['subject_id', 'stay_id']].copy()
        z['subject_id'] = pd.to_numeric(z.subject_id, errors='raise').astype('int64')
        z['stay_id'] = pd.to_numeric(z.stay_id, errors='raise').astype('int64')
        z = z.sort_values(['subject_id', 'stay_id']).reset_index(drop=True)
        payload = 'subject_id,stay_id\n' + ''.join((f'{int(r.subject_id)},{int(r.stay_id)}\n' for r in z.itertuples(index=False)))
        return hashlib.sha256(payload.encode()).hexdigest()

    def first_positive(df, positive_col='_positive'):
        q = df.loc[df[positive_col].fillna(False).astype(bool)].copy()
        q = q.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
        return q.drop_duplicates('subject_id', keep='first').copy()

    def stage_equal(a, b):
        aa = pd.to_numeric(a, errors='coerce').fillna(-999).to_numpy(float)
        bb = pd.to_numeric(b, errors='coerce').fillna(-999).to_numpy(float)
        return np.isclose(aa, bb, rtol=0, atol=1e-10)

    def membership_compare(reference, alternative, label, ref_label):
        rm = dict(zip(reference.subject_id.astype(int), reference.stay_id.astype(int)))
        am = dict(zip(alternative.subject_id.astype(int), alternative.stay_id.astype(int)))
        rs, aset = (set(rm), set(am))
        common = rs & aset
        repl = sum((rm[s] != am[s] for s in common))
        r_stays = set(rm.values())
        a_stays = set(am.values())
        inter = len(r_stays & a_stays)
        union = len(r_stays | a_stays)
        return {'variant': label, 'reference': ref_label, 'subjects': len(alternative), 'subject_added_vs_reference': len(aset - rs), 'subject_dropped_vs_reference': len(rs - aset), 'common_subjects': len(common), 'same_subject_replaced_stay': int(repl), 'same_subject_same_stay': int(len(common) - repl), 'stay_symmetric_difference': len(r_stays ^ a_stays), 'stay_jaccard': float(inter / union) if union else np.nan}

    def save_csv(df, name):
        p = AUDIT / name
        df.to_csv(p, index=False)
        return p

    def save_json(obj, name):
        p = AUDIT / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT  :', MIMIC_ROOT)
    print('ADJUDICATION:', AUDIT)

    P_STRICT_TARGET = STRICT_DIR / 'strict_outpatient_target.parquet'
    P_STRICT_INDEX = STRICT_DIR / 'strict_outpatient_cohort_preoutcome.parquet'
    P_STRICT_CR = STRICT_DIR / 'strict_outpatient_creatinine.parquet'
    P_REFERENCE_TARGET = CONSECUTIVE_REFERENCE_DIR / 'reference_target.parquet'
    P_REFERENCE_INDEX = CONSECUTIVE_REFERENCE_DIR / 'reference_cohort_preoutcome.parquet'
    P_LM = COHORT_DIR / 'landmark_population.parquet'
    P_UOBASE = URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet'
    P_POST_TRANSPLANT = POST_TRANSPLANT_DIR / 'post_transplant_stay_reconstruction.csv'
    P_PERSISTENT_UO = PERSISTENT_OLIGURIA_DIR / 'persistent_oliguria_stages.csv'

    strict_target = read_parquet_allow(
        P_STRICT_TARGET,
        ['subject_id', 'stay_id', 'intime', 'landmark_time',
         'kdigo_creatinine_stage_strict_outpatient', 'creatinine_evaluable_strict_outpatient',
         'urine_output_stage_reliability_adjusted', 'urine_output_absence_evaluable_reliability_adjusted',
         'active_rrt_any_by_landmark', 'uo_unreliable_by_landmark', 'combined_status_strict_outpatient'],
        'strict-outpatient target',
    )
    strict_index = read_parquet_allow(
        P_STRICT_INDEX, ['subject_id', 'stay_id', 'intime'], 'strict-outpatient cohort'
    )
    reference_target = read_parquet_allow(
        P_REFERENCE_TARGET,
        ['subject_id', 'stay_id', 'intime', 'landmark_time',
         'kdigo_creatinine_stage_strict_outpatient', 'urine_output_stage_consecutive',
         'active_rrt_any_by_landmark', 'aki_positive_consecutive_urine_output',
         'kdigo_stage_consecutive_urine_output', 'target_source_consecutive_urine_output'],
        'consecutive urine-output reference target',
    )
    reference_index = read_parquet_allow(
        P_REFERENCE_INDEX, ['subject_id', 'stay_id', 'intime'], 'consecutive urine-output reference cohort'
    )
    lm = read_parquet_allow(
        P_LM, ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time'],
        'landmark non-outcome fields',
    )
    uobase = read_parquet_allow(
        P_UOBASE, ['stay_id', 'admission_weight_kg', 'ibw_kg', 'height_cm', 'gender'],
        'urine-output weight/height base',
    )
    post_transplant = read_csv_allow(
        P_POST_TRANSPLANT,
        ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time',
         'post_transplant_temporal_allow_acute_rrt', 'kdigo_creatinine_stage_post_transplant',
         'creatinine_evaluable_post_transplant', 'active_rrt_any_by_landmark',
         'uo_unreliable_by_landmark', 'hist_strict_n', 'hist_strict_median',
         'latest_prior_krt_state_time', 'latest_prior_tx_status_time', 'latest_prior_tx_failure_time',
         'latest_prior_tx_proc_time', 'current_tx_status', 'current_tx_failure', 'current_eskd_code',
         'current_dialysis_dependence_code', 'current_chronic_krt_state_code',
         'current_dialysis_encounter_code'],
        'post-transplant reconstruction',
    )
    persistent_uo = read_csv_allow(
        P_PERSISTENT_UO,
        ['stay_id', 'urine_output_stage_consecutive', 'urine_output_stage_mean', 'valid_hours'],
        'persistent urine-output stage comparator',
    )

    if len(strict_target) != EXPECTED_STRICT_TARGET_N or len(strict_index) != EXPECTED_STRICT_N:
        raise AssertionError('Strict-outpatient cohort hard guard failed.')
    if len(reference_target) != EXPECTED_REFERENCE_TARGET_N or len(reference_index) != EXPECTED_REFERENCE_N:
        raise AssertionError('Consecutive urine-output reference cohort hard guard failed.')

    post_transplant_allowed = post_transplant.loc[
        post_transplant.post_transplant_temporal_allow_acute_rrt.fillna(False).astype(bool)
    ].copy()
    if len(post_transplant_allowed) != 111:
        raise AssertionError(
            f'Expected 111 temporal post-transplant target stays, got {len(post_transplant_allowed)}.'
        )

    strict_base = strict_target.merge(
        lm[['stay_id', 'hadm_id', 'admittime']], on='stay_id', how='left', validate='1:1'
    ).rename(columns={
        'kdigo_creatinine_stage_strict_outpatient': 'cr_stage',
        'creatinine_evaluable_strict_outpatient': 'cr_evaluable',
    })
    strict_base['target_source'] = 'strict_outpatient_renal_eligible'

    post_transplant_base = post_transplant_allowed[[
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time',
        'kdigo_creatinine_stage_post_transplant', 'creatinine_evaluable_post_transplant',
        'active_rrt_any_by_landmark', 'uo_unreliable_by_landmark'
    ]].copy().rename(columns={
        'kdigo_creatinine_stage_post_transplant': 'cr_stage',
        'creatinine_evaluable_post_transplant': 'cr_evaluable',
    })
    post_transplant_base['target_source'] = 'post_transplant_temporal'

    keep_cols = [
        'subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time',
        'cr_stage', 'cr_evaluable', 'active_rrt_any_by_landmark',
        'uo_unreliable_by_landmark', 'target_source'
    ]
    expanded = pd.concat([strict_base[keep_cols], post_transplant_base[keep_cols]], ignore_index=True)
    for c in ['admittime', 'intime', 'landmark_time']:
        expanded[c] = pd.to_datetime(expanded[c], errors='raise')
    for c in ['subject_id', 'hadm_id', 'stay_id']:
        expanded[c] = pd.to_numeric(expanded[c], errors='raise').astype('int64')
    if expanded.stay_id.duplicated().any() or len(expanded) != EXPECTED_REFERENCE_TARGET_N:
        raise AssertionError('Expanded renal-eligibility target failed uniqueness/count guard.')
    expanded = expanded.merge(
        uobase[['stay_id', 'admission_weight_kg', 'ibw_kg', 'height_cm', 'gender']],
        on='stay_id', how='left', validate='1:1'
    )
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    record_access(OE, ['stay_id', 'itemid', 'charttime', 'storetime', 'value'], 'raw outputevents')
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    COMPARTMENT_CASE = "\nCASE\n WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n ELSE NULL\nEND\n"
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('expanded_df', expanded[['stay_id', 'intime', 'landmark_time']])
    con.execute('CREATE TEMP TABLE target AS SELECT * FROM expanded_df')
    con.execute(f"\nCREATE VIEW oe AS\nSELECT stay_id,itemid,charttime,storetime,value\nFROM read_csv_auto('{qpath(OE)}',header=true,sample_size=200000)\n")
    con.execute(f'\nCREATE TEMP TABLE uo_events AS\nSELECT\n    t.stay_id,\n    {COMPARTMENT_CASE} AS compartment,\n    oe.charttime,\n    MAX(oe.storetime) AS storetime,\n    SUM(oe.value) AS volume_ml\nFROM target t\nJOIN oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL\n  AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime>=t.intime\n  AND oe.charttime<=t.landmark_time\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime<=t.landmark_time\nGROUP BY t.stay_id,compartment,oe.charttime\nHAVING compartment IS NOT NULL\n')
    con.execute('\nCREATE TEMP TABLE raw_intervals AS\nSELECT\n    stay_id,compartment,\n    LAG(charttime) OVER(PARTITION BY stay_id,compartment ORDER BY charttime) AS interval_start,\n    charttime AS interval_end,\n    volume_ml\nFROM uo_events\n')
    intervals = con.execute("\nSELECT\n    r.stay_id,r.compartment,r.interval_start,r.interval_end,r.volume_ml,\n    DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0 AS duration_hours,\n    r.volume_ml/(DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0) AS rate_ml_hr\nFROM raw_intervals r\nJOIN target t USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end>r.interval_start\n  AND r.interval_start>=t.intime\n  AND r.interval_end<=t.landmark_time\n  AND DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0<=6.0+1e-12\nORDER BY r.stay_id,r.interval_start\n").df()
    hour_rows = []
    for r in expanded[['stay_id', 'intime', 'landmark_time']].itertuples(index=False):
        cur = pd.Timestamp(r.intime).floor('h')
        lm = pd.Timestamp(r.landmark_time)
        while cur + pd.Timedelta(hours=1) <= lm:
            hour_rows.append((int(r.stay_id), cur, cur + pd.Timedelta(hours=1)))
            cur += pd.Timedelta(hours=1)
    hours = pd.DataFrame(hour_rows, columns=['stay_id', 'hour_start', 'hour_end'])
    conh = duckdb.connect(database=':memory:')
    conh.execute('PRAGMA threads=4')
    conh.register('hours_df', hours)
    conh.register('intervals_df', intervals)
    conh.execute('CREATE TEMP TABLE hours AS SELECT * FROM hours_df')
    conh.execute('CREATE TEMP TABLE intervals AS SELECT * FROM intervals_df')
    hourly = conh.execute("\nWITH ov0 AS (\n SELECT h.stay_id,h.hour_start,h.hour_end,\n        i.compartment,i.interval_start,i.interval_end,i.rate_ml_hr,\n        GREATEST(i.interval_start,h.hour_start) ov_start,\n        LEAST(i.interval_end,h.hour_end) ov_end\n FROM hours h\n LEFT JOIN intervals i\n   ON i.stay_id=h.stay_id\n  AND i.interval_end>h.hour_start\n  AND i.interval_start<h.hour_end\n),\nov AS (\n SELECT * FROM ov0 WHERE ov_end>ov_start\n),\nnum_span AS (\n SELECT stay_id,hour_start,hour_end,\n   SUM(rate_ml_hr*DATE_DIFF('second',ov_start,ov_end)/3600.0) AS numerator_ml,\n   DATE_DIFF('second',MIN(ov_start),MAX(ov_end))/3600.0 AS span_coverage_h,\n   COUNT(DISTINCT compartment) AS compartments_in_hour\n FROM ov\n GROUP BY stay_id,hour_start,hour_end\n),\nordered AS (\n SELECT *,\n   MAX(ov_end) OVER(\n     PARTITION BY stay_id,hour_start,hour_end\n     ORDER BY ov_start,ov_end\n     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING\n   ) AS prior_max_end\n FROM ov\n),\nmarked AS (\n SELECT *,\n   CASE WHEN prior_max_end IS NULL OR ov_start>prior_max_end THEN 1 ELSE 0 END AS new_group\n FROM ordered\n),\ngrp AS (\n SELECT *,\n   SUM(new_group) OVER(\n     PARTITION BY stay_id,hour_start,hour_end\n     ORDER BY ov_start,ov_end\n     ROWS UNBOUNDED PRECEDING\n   ) AS union_group\n FROM marked\n),\nmerged AS (\n SELECT stay_id,hour_start,hour_end,union_group,\n        MIN(ov_start) merged_start,MAX(ov_end) merged_end\n FROM grp\n GROUP BY stay_id,hour_start,hour_end,union_group\n),\nunion_cov AS (\n SELECT stay_id,hour_start,hour_end,\n   SUM(DATE_DIFF('second',merged_start,merged_end)/3600.0) AS union_coverage_h\n FROM merged\n GROUP BY stay_id,hour_start,hour_end\n)\nSELECT\n h.stay_id,h.hour_start,h.hour_end,\n ns.numerator_ml,ns.span_coverage_h,uc.union_coverage_h,\n COALESCE(ns.compartments_in_hour,0) AS compartments_in_hour,\n CASE WHEN ns.span_coverage_h>0.5 THEN ns.numerator_ml/ns.span_coverage_h END AS span_ml_hr,\n CASE WHEN uc.union_coverage_h>0.5 THEN ns.numerator_ml/uc.union_coverage_h END AS union_ml_hr\nFROM hours h\nLEFT JOIN num_span ns USING(stay_id,hour_start,hour_end)\nLEFT JOIN union_cov uc USING(stay_id,hour_start,hour_end)\nORDER BY h.stay_id,h.hour_end\n").df()
    hourly = hourly.merge(expanded[['stay_id', 'intime', 'uo_unreliable_by_landmark', 'admission_weight_kg', 'ibw_kg']], on='stay_id', how='left', validate='m:1')
    for source in ['span', 'union']:
        hourly[f'{source}_ibw'] = pd.to_numeric(hourly[f'{source}_ml_hr'], errors='coerce') / pd.to_numeric(hourly.ibw_kg, errors='coerce')
        hourly[f'{source}_abw'] = pd.to_numeric(hourly[f'{source}_ml_hr'], errors='coerce') / pd.to_numeric(hourly.admission_weight_kg, errors='coerce')
        bad = hourly.uo_unreliable_by_landmark.fillna(0).eq(1)
        hourly.loc[bad, [f'{source}_ibw', f'{source}_abw']] = np.nan

    def max_consecutive_valid(vals):
        best = cur = 0
        for ok in np.isfinite(vals):
            if ok:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def uo_stage(g, value_col, mode='mean', elapsed_guard=False, future_strict=False):
        g = g.sort_values('hour_end').reset_index(drop=True)
        vals = pd.to_numeric(g[value_col], errors='coerce').to_numpy(float)
        intime = pd.Timestamp(g.intime.iloc[0])
        best = 0
        window_specs = [(6, 6, 's1'), (12, 13 if future_strict else 12, 's2'), (12, 13 if future_strict else 12, 'anuria'), (24, 25 if future_strict else 24, 's3')]
        for nominal_W, row_W, kind in window_specs:
            for i in range(len(g)):
                j = i - row_W + 1
                if j < 0:
                    continue
                x = vals[j:i + 1]
                if not np.isfinite(x).all():
                    continue
                t = pd.to_datetime(g.hour_end.iloc[j:i + 1])
                if len(t) > 1:
                    diffs = np.diff(t.values.astype('datetime64[s]').astype('int64'))
                    if not np.all(diffs == 3600):
                        continue
                if elapsed_guard:
                    actual_start = max(pd.Timestamp(g.hour_start.iloc[j]), intime)
                    actual_end = pd.Timestamp(g.hour_end.iloc[i])
                    elapsed = (actual_end - actual_start).total_seconds() / 3600
                    required = float(row_W)
                    if elapsed + 1e-12 < required:
                        continue
                stat = float(np.mean(x)) if mode == 'mean' else float(np.max(x))
                if kind == 's1' and stat < 0.5:
                    best = max(best, 1)
                elif kind == 's2' and stat < 0.5:
                    best = max(best, 2)
                elif kind == 'anuria' and stat == 0:
                    best = max(best, 3)
                elif kind == 's3' and stat < 0.3:
                    best = max(best, 3)
        return float(best) if best >= 1 else np.nan

    def build_stages(value_col, mode, elapsed=False, future=False, label=''):
        rows = []
        for stay_id, g in hourly.groupby('stay_id', sort=False):
            vals = pd.to_numeric(g[value_col], errors='coerce').to_numpy(float)
            rows.append({'stay_id': int(stay_id), label: uo_stage(g, value_col, mode=mode, elapsed_guard=elapsed, future_strict=future), f'{label}_valid_hours': int(np.isfinite(vals).sum()), f'{label}_max_consecutive_valid': max_consecutive_valid(vals)})
        return pd.DataFrame(rows)
    stage_tables = []
    for denom in ['ibw', 'abw']:
        for mode in ['mean', 'cons']:
            lab = f'cal_{mode}_{denom}'
            stage_tables.append(build_stages(f'span_{denom}', mode, label=lab))
    stages = expanded[['stay_id']].copy()
    for t in stage_tables:
        stages = stages.merge(t, on='stay_id', how='left', validate='1:1')
    chk = persistent_uo.merge(stages[['stay_id', 'cal_cons_ibw', 'cal_mean_ibw']], on='stay_id', how='left', validate='1:1')
    if not stage_equal(chk.urine_output_stage_consecutive, chk.cal_cons_ibw).all():
        raise AssertionError('Independent calendar-hour consecutive urine-output reconstruction does not reproduce the upstream comparator.')
    if not stage_equal(chk.urine_output_stage_mean, chk.cal_mean_ibw).all():
        raise AssertionError('Independent calendar-hour mean urine-output reconstruction does not reproduce the upstream comparator.')

    def select_variant(stage_col, label, cr_col='cr_stage', rrt_override=None):
        q = expanded[['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'target_source', cr_col, 'cr_evaluable', 'active_rrt_any_by_landmark']].copy()
        q = q.merge(stages[['stay_id', stage_col]], on='stay_id', how='left', validate='1:1')
        cr = pd.to_numeric(q[cr_col], errors='coerce')
        uo = pd.to_numeric(q[stage_col], errors='coerce')
        rrt = q.active_rrt_any_by_landmark.fillna(0).eq(1) if rrt_override is None else pd.Series(rrt_override, index=q.index)
        q['_positive'] = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        st = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
        st = st.where(st.ge(1), np.nan).mask(rrt, 3.0)
        q['_combined_stage'] = st
        q['_variant'] = label
        return (first_positive(q), q)
    sel_cons_ibw, _ = select_variant('cal_cons_ibw', 'calendar_hour_consecutive_ibw_post_transplant')
    reference_membership = reference_index[['subject_id', 'stay_id']].sort_values(['subject_id', 'stay_id']).reset_index(drop=True)
    reproduced_membership = sel_cons_ibw[['subject_id', 'stay_id']].sort_values(['subject_id', 'stay_id']).reset_index(drop=True)
    if not reproduced_membership.equals(reference_membership):
        raise AssertionError('Calendar-hour consecutive urine output did not reproduce the consecutive urine-output reference cohort membership.')
    save_csv(pd.DataFrame(INPUT_ACCESS), 'adjudication_input_allowlist.csv')
    print('Preparation hard guards: PASS')
    print('Expanded eligible target stays:', len(expanded))
    sel_mean_ibw, target_mean_ibw = select_variant('cal_mean_ibw', 'calendar_hour_mean_ibw_post_transplant')
    cmp0 = pd.DataFrame([membership_compare(strict_index, sel_mean_ibw, 'calendar_hour_mean_ibw_post_transplant', 'strict_outpatient'), membership_compare(reference_index, sel_mean_ibw, 'calendar_hour_mean_ibw_post_transplant', 'consecutive_urine_output_reference'), membership_compare(strict_index, sel_cons_ibw, 'calendar_hour_consecutive_ibw_post_transplant', 'strict_outpatient'), membership_compare(reference_index, sel_cons_ibw, 'calendar_hour_consecutive_ibw_post_transplant', 'consecutive_urine_output_reference')])
    stage0 = sel_mean_ibw['_combined_stage'].value_counts(dropna=False).rename_axis('combined_stage').reset_index(name='subjects').sort_values('combined_stage', na_position='last')
    source0 = sel_mean_ibw.target_source.value_counts(dropna=False).rename_axis('target_source').reset_index(name='subjects')
    display_table(cmp0)
    display_table(stage0)
    display_table(source0)
    save_csv(cmp0, 'adjudication_primary_membership.csv')
    save_csv(stage0, 'adjudication_primary_stage_distribution.csv')
    save_csv(source0, 'adjudication_source_distribution.csv')
    save_csv(sel_mean_ibw, 'adjudication_selected_preoutcome.csv')
    sel_cons_abw, target_cons_abw = select_variant('cal_cons_abw', 'calendar_hour_consecutive_abw_post_transplant')
    sel_mean_abw, target_mean_abw = select_variant('cal_mean_abw', 'calendar_hour_mean_abw_post_transplant')
    bounded = expanded[['stay_id', 'admission_weight_kg']].copy()
    bounded['abw_plausible'] = pd.to_numeric(bounded.admission_weight_kg, errors='coerce').between(25, 300, inclusive='both')
    stages_bounded = stages.merge(bounded[['stay_id', 'abw_plausible']], on='stay_id', validate='1:1')
    for c in ['cal_mean_abw', 'cal_cons_abw']:
        stages_bounded[f'{c}_bounded'] = stages_bounded[c].where(stages_bounded.abw_plausible)
    stages = stages.merge(stages_bounded[['stay_id', 'cal_mean_abw_bounded', 'cal_cons_abw_bounded']], on='stay_id', how='left', validate='1:1')
    sel_cons_abw_b, _ = select_variant('cal_cons_abw_bounded', 'calendar_hour_consecutive_abw_25_300kg_post_transplant')
    sel_mean_abw_b, _ = select_variant('cal_mean_abw_bounded', 'calendar_hour_mean_abw_25_300kg_post_transplant')
    rows = []
    for sel, label in [(sel_cons_abw, 'consecutive_abw'), (sel_cons_abw_b, 'consecutive_abw_25_300kg'), (sel_mean_abw, 'mean_abw'), (sel_mean_abw_b, 'mean_abw_25_300kg')]:
        rows.append(membership_compare(sel_cons_ibw, sel, label, 'consecutive_ibw'))
        rows.append(membership_compare(sel_mean_ibw, sel, label, 'mean_ibw'))
    summary1 = pd.DataFrame(rows)
    weight_qc = pd.DataFrame([{'expanded_target_stays': len(expanded), 'abw_available': int(pd.to_numeric(expanded.admission_weight_kg, errors='coerce').gt(0).sum()), 'abw_lt25': int(pd.to_numeric(expanded.admission_weight_kg, errors='coerce').lt(25).sum()), 'abw_gt300': int(pd.to_numeric(expanded.admission_weight_kg, errors='coerce').gt(300).sum()), 'abw_plausible_25_300': int(pd.to_numeric(expanded.admission_weight_kg, errors='coerce').between(25, 300).sum())}])
    display_table(summary1)
    display_table(weight_qc)
    save_csv(summary1, 'abw_membership_similarity.csv')
    save_csv(weight_qc, 'abw_plausibility.csv')
    fixed = strict_index[['subject_id', 'stay_id', 'intime']].merge(expanded[['stay_id', 'cr_stage', 'active_rrt_any_by_landmark']], on='stay_id', how='left', validate='1:1').merge(stages[['stay_id', 'cal_mean_ibw', 'cal_cons_ibw', 'cal_mean_abw', 'cal_cons_abw']], on='stay_id', how='left', validate='1:1')
    rows = []
    details = []
    for c in ['cal_mean_ibw', 'cal_cons_ibw', 'cal_mean_abw', 'cal_cons_abw']:
        cr = pd.to_numeric(fixed.cr_stage, errors='coerce')
        uo = pd.to_numeric(fixed[c], errors='coerce')
        rrt = fixed.active_rrt_any_by_landmark.fillna(0).eq(1)
        pos = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        st = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
        st = st.where(st.ge(1), np.nan).mask(rrt, 3.0)
        rows.append({'variant': c, 'strict_outpatient_selected_stays': len(fixed), 'still_AKI_positive_at_same_stay': int(pos.sum()), 'no_longer_positive_at_same_stay': int((~pos).sum())})
        d = fixed[['subject_id', 'stay_id']].copy()
        d['variant'] = c
        d['positive_same_stay'] = pos
        d['combined_stage_same_stay'] = st
        details.append(d)
    summary2 = pd.DataFrame(rows)
    detail2 = pd.concat(details, ignore_index=True)
    display_table(summary2)
    save_csv(summary2, 'strict_outpatient_fixed_stay_summary.csv')
    save_csv(detail2, 'strict_outpatient_fixed_stay_detail.csv')
    first_eligible = expanded.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').drop_duplicates('subject_id', keep='first')[['subject_id', 'stay_id', 'intime', 'cr_stage', 'active_rrt_any_by_landmark']].merge(stages[['stay_id', 'cal_mean_ibw', 'cal_cons_ibw', 'cal_mean_abw', 'cal_cons_abw']], on='stay_id', how='left', validate='1:1')
    rows = []
    details = []
    for c in ['cal_mean_ibw', 'cal_cons_ibw', 'cal_mean_abw', 'cal_cons_abw']:
        cr = pd.to_numeric(first_eligible.cr_stage, errors='coerce')
        uo = pd.to_numeric(first_eligible[c], errors='coerce')
        rrt = first_eligible.active_rrt_any_by_landmark.fillna(0).eq(1)
        pos = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        rows.append({'variant': c, 'subjects_with_first_eligible_stay': len(first_eligible), 'AKI_positive_at_fixed_first_eligible_stay': int(pos.sum()), 'not_positive_at_fixed_first_eligible_stay': int((~pos).sum())})
        d = first_eligible[['subject_id', 'stay_id']].copy()
        d['variant'] = c
        d['positive'] = pos
        details.append(d)
    summary3 = pd.DataFrame(rows)
    display_table(summary3)
    save_csv(summary3, 'first_eligible_timezero_summary.csv')
    save_csv(pd.concat(details, ignore_index=True), 'first_eligible_timezero_detail.csv')
    asc = expanded[['stay_id', 'cr_stage', 'cr_evaluable', 'active_rrt_any_by_landmark', 'admission_weight_kg', 'ibw_kg']].merge(stages[['stay_id', 'cal_mean_ibw', 'cal_cons_ibw', 'cal_mean_abw', 'cal_cons_abw', 'cal_mean_ibw_max_consecutive_valid', 'cal_mean_abw_max_consecutive_valid']], on='stay_id', how='left', validate='1:1')
    asc['ibw_usable'] = pd.to_numeric(asc.ibw_kg, errors='coerce').gt(0)
    asc['abw_usable'] = pd.to_numeric(asc.admission_weight_kg, errors='coerce').gt(0)
    asc['abw_25_300'] = pd.to_numeric(asc.admission_weight_kg, errors='coerce').between(25, 300)

    def ascertainment(stage_col, maxvalid_col, weight_ok):
        pos = pd.to_numeric(asc[stage_col], errors='coerce').fillna(0).ge(1)
        uo_asc = weight_ok & pd.to_numeric(asc[maxvalid_col], errors='coerce').fillna(0).ge(6)
        return (np.select([pos, uo_asc & ~pos], ['ascertainable_positive', 'ascertainable_negative_stage0'], default='unascertainable'), uo_asc)
    rows = []
    for stage_col, maxv, wok, name in [('cal_mean_ibw', 'cal_mean_ibw_max_consecutive_valid', asc.ibw_usable, 'mean_ibw'), ('cal_cons_ibw', 'cal_mean_ibw_max_consecutive_valid', asc.ibw_usable, 'consecutive_ibw'), ('cal_mean_abw', 'cal_mean_abw_max_consecutive_valid', asc.abw_usable, 'mean_abw'), ('cal_cons_abw', 'cal_mean_abw_max_consecutive_valid', asc.abw_usable, 'consecutive_abw')]:
        status, uo_asc = ascertainment(stage_col, maxv, wok)
        tmp = pd.Series(status).value_counts()
        crpos = pd.to_numeric(asc.cr_stage, errors='coerce').fillna(0).ge(1)
        rrt = asc.active_rrt_any_by_landmark.fillna(0).eq(1)
        uopos = pd.to_numeric(asc[stage_col], errors='coerce').fillna(0).ge(1)
        combined_pos = crpos | rrt | uopos
        confirmed_negative = ~combined_pos & asc.cr_evaluable.fillna(0).eq(1) & uo_asc
        rows.append({'variant': name, 'uo_ascertainable_positive': int((status == 'ascertainable_positive').sum()), 'uo_ascertainable_negative_stage0': int((status == 'ascertainable_negative_stage0').sum()), 'uo_unascertainable': int((status == 'unascertainable').sum()), 'combined_positive': int(combined_pos.sum()), 'combined_confirmed_negative': int(confirmed_negative.sum()), 'no_positive_but_not_fully_negative': int((~combined_pos & ~confirmed_negative).sum())})
    summary4 = pd.DataFrame(rows)
    display_table(summary4)
    save_csv(summary4, 'urine_output_ascertainability.csv')
    elapsed_tables = []
    for denom in ['ibw', 'abw']:
        for mode in ['mean', 'cons']:
            lab = f'elapsed_{mode}_{denom}'
            elapsed_tables.append(build_stages(f'span_{denom}', mode, elapsed=True, label=lab))
    elapsed = expanded[['stay_id']].copy()
    for t in elapsed_tables:
        elapsed = elapsed.merge(t, on='stay_id', how='left', validate='1:1')
    rows = []
    for reference_stage, alt in [('cal_mean_ibw', 'elapsed_mean_ibw'), ('cal_cons_ibw', 'elapsed_cons_ibw'), ('cal_mean_abw', 'elapsed_mean_abw'), ('cal_cons_abw', 'elapsed_cons_abw')]:
        x = stages[['stay_id', reference_stage]].merge(elapsed[['stay_id', alt]], on='stay_id', validate='1:1')
        rows.append({'comparison': f'{reference_stage} -> {alt}', 'target_stage_changed': int((~stage_equal(x[reference_stage], x[alt])).sum()), 'positive_removed': int((pd.to_numeric(x[reference_stage], errors='coerce').fillna(0).ge(1) & pd.to_numeric(x[alt], errors='coerce').fillna(0).lt(1)).sum()), 'positive_added': int((pd.to_numeric(x[reference_stage], errors='coerce').fillna(0).lt(1) & pd.to_numeric(x[alt], errors='coerce').fillna(0).ge(1)).sum())})
    summary5 = pd.DataFrame(rows)
    stages = stages.merge(elapsed[['stay_id', 'elapsed_mean_ibw', 'elapsed_cons_ibw']], on='stay_id', how='left', validate='1:1')
    sel_em, target_em = select_variant('elapsed_mean_ibw', 'elapsed_guard_mean_ibw_post_transplant')
    sel_ec, target_ec = select_variant('elapsed_cons_ibw', 'elapsed_guard_consecutive_ibw_post_transplant')
    if len(sel_em) != 3457:
        raise AssertionError(f'Elapsed-duration mean-IBW cohort N={len(sel_em)}, expected 3457.')
    if len(sel_ec) != 2972:
        raise AssertionError(f'Elapsed-duration consecutive-IBW cohort N={len(sel_ec)}, expected 2972.')
    mem5 = pd.DataFrame([membership_compare(sel_mean_ibw, sel_em, 'elapsed_guard_mean_ibw', 'standard_mean_ibw'), membership_compare(sel_cons_ibw, sel_ec, 'elapsed_guard_consecutive_ibw', 'standard_consecutive_ibw')])
    display_table(summary5)
    display_table(mem5)
    save_csv(summary5, 'elapsed_duration_stage_effect.csv')
    save_csv(mem5, 'elapsed_duration_membership_effect.csv')
    hourly['span_minus_union_h'] = pd.to_numeric(hourly.span_coverage_h, errors='coerce') - pd.to_numeric(hourly.union_coverage_h, errors='coerce')
    cov6 = pd.DataFrame([{'calendar_hour_rows': len(hourly), 'hours_span_gt_union_gt1sec': int(hourly.span_minus_union_h.fillna(0).gt(1 / 3600).sum()), 'max_span_minus_union_minutes': float(60 * hourly.span_minus_union_h.fillna(0).max()), 'hours_multiple_compartments': int(pd.to_numeric(hourly.compartments_in_hour, errors='coerce').fillna(0).ge(2).sum()), 'multi_compartment_hours_with_span_union_gap': int((pd.to_numeric(hourly.compartments_in_hour, errors='coerce').fillna(0).ge(2) & hourly.span_minus_union_h.fillna(0).gt(1 / 3600)).sum())}])
    union_tables = []
    for denom in ['ibw', 'abw']:
        for mode in ['mean', 'cons']:
            lab = f'union_{mode}_{denom}'
            union_tables.append(build_stages(f'union_{denom}', mode, label=lab))
    union_stage = expanded[['stay_id']].copy()
    for t in union_tables:
        union_stage = union_stage.merge(t, on='stay_id', how='left', validate='1:1')
    rows = []
    for reference_stage, alt in [('cal_mean_ibw', 'union_mean_ibw'), ('cal_cons_ibw', 'union_cons_ibw'), ('cal_mean_abw', 'union_mean_abw'), ('cal_cons_abw', 'union_cons_abw')]:
        x = stages[['stay_id', reference_stage]].merge(union_stage[['stay_id', alt]], on='stay_id', validate='1:1')
        rows.append({'comparison': f'{reference_stage}->{alt}', 'target_stage_changed': int((~stage_equal(x[reference_stage], x[alt])).sum())})
    summary6 = pd.DataFrame(rows)
    display_table(cov6)
    display_table(summary6)
    save_csv(cov6, 'interval_union_coverage.csv')
    save_csv(summary6, 'interval_union_stage_effect.csv')
    save_csv(hourly.loc[hourly.span_minus_union_h.fillna(0).gt(1 / 3600), ['stay_id', 'hour_start', 'hour_end', 'compartments_in_hour', 'span_coverage_h', 'union_coverage_h', 'span_minus_union_h']], 'interval_union_gap_detail.csv')
    summary7 = pd.DataFrame([membership_compare(sel_mean_ibw, sel_cons_ibw, 'calendar_hour_consecutive_ibw_post_transplant', 'calendar_hour_mean_ibw_post_transplant'), membership_compare(sel_cons_ibw, sel_mean_ibw, 'calendar_hour_mean_ibw_post_transplant', 'calendar_hour_consecutive_ibw_post_transplant')])
    target7 = stages[['stay_id', 'cal_mean_ibw', 'cal_cons_ibw']].copy()
    target7['stage_discordant'] = ~stage_equal(target7.cal_mean_ibw, target7.cal_cons_ibw)
    display_table(summary7)
    save_csv(summary7, 'mean_vs_consecutive_ibw_membership.csv')
    save_csv(target7.loc[target7.stage_discordant], 'mean_vs_consecutive_ibw_target_discordance.csv')
    summary8 = pd.DataFrame([membership_compare(sel_mean_abw, sel_cons_abw, 'consecutive_abw', 'mean_abw'), membership_compare(sel_mean_abw_b, sel_cons_abw_b, 'consecutive_abw_25_300kg', 'mean_abw_25_300kg'), membership_compare(sel_mean_abw, sel_mean_abw_b, 'mean_abw_25_300kg', 'mean_abw'), membership_compare(sel_cons_abw, sel_cons_abw_b, 'consecutive_abw_25_300kg', 'consecutive_abw')])
    display_table(summary8)
    save_csv(summary8, 'mean_vs_consecutive_abw.csv')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    TR = req(MIMIC_ROOT / 'hosp' / 'transfers.csv.gz')
    record_access(LAB, ['subject_id', 'hadm_id', 'itemid', 'charttime', 'storetime', 'valuenum'], 'raw creatinine labevents')
    record_access(TR, ['subject_id', 'hadm_id', 'eventtype', 'careunit', 'intime', 'outtime'], 'raw transfers')
    adm_allow = read_csv_allow(ADM, ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'edregtime', 'edouttime'], 'admissions explicit non-outcome allowlist')
    tr_allow = read_csv_allow(TR, ['subject_id', 'hadm_id', 'eventtype', 'careunit', 'intime', 'outtime'], 'transfers allowlist')
    for c in ['admittime', 'dischtime', 'edregtime', 'edouttime']:
        adm_allow[c] = pd.to_datetime(adm_allow[c], errors='coerce')
    for c in ['intime', 'outtime']:
        tr_allow[c] = pd.to_datetime(tr_allow[c], errors='coerce')
    for c in ['admittime', 'dischtime', 'edregtime', 'edouttime']:
        if not pd.api.types.is_datetime64_any_dtype(adm_allow[c]):
            raise TypeError(f'Admissions timestamp normalization failed for {c}')
    for c in ['intime', 'outtime']:
        if not pd.api.types.is_datetime64_any_dtype(tr_allow[c]):
            raise TypeError(f'Transfers timestamp normalization failed for {c}')
    conc = duckdb.connect(database=':memory:')
    conc.execute('PRAGMA threads=4')
    conc.register('target_df', expanded[['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time']])
    conc.register('adm_df', adm_allow)
    conc.register('tr_df', tr_allow)
    conc.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM target_df')
    conc.execute('CREATE TEMP TABLE admissions AS SELECT * FROM adm_df')
    conc.execute('CREATE TEMP TABLE transfers_all AS SELECT * FROM tr_df')
    conc.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT subject_id,hadm_id,charttime,storetime,valuenum AS creatinine\nFROM read_csv_auto('{qpath(LAB)}',header=true,sample_size=200000)\nWHERE itemid=50912 AND valuenum>0 AND valuenum<=150\n  AND charttime IS NOT NULL AND storetime IS NOT NULL\n")
    conc.execute("\nCREATE TEMP TABLE valid_physical_transfers AS\nSELECT subject_id,hadm_id,LOWER(COALESCE(eventtype,'')) eventtype,\n       careunit,intime,outtime\nFROM transfers_all\nWHERE intime IS NOT NULL AND outtime IS NOT NULL AND outtime>intime\n  AND careunit IS NOT NULL\n  AND LOWER(COALESCE(eventtype,'')) IN ('ed','admit','transfer')\n")
    conc.execute("\nCREATE TEMP TABLE valid_ed_transfers AS\nSELECT * FROM valid_physical_transfers\nWHERE eventtype='ed' OR LOWER(careunit) LIKE '%emergency%'\n")
    conc.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT\n t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime,\n MAX(c.storetime) storetime,AVG(c.creatinine) creatinine,\n MAX(CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END) has_labevent_hadm_id\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    conc.execute('\nCREATE TEMP TABLE hist_strict AS\nSELECT h.*\nFROM hist_same_time h\nWHERE has_labevent_hadm_id=0\n AND NOT EXISTS (\n    SELECT 1 FROM admissions a\n    WHERE a.subject_id=h.subject_id\n      AND a.admittime IS NOT NULL AND a.dischtime IS NOT NULL\n      AND h.charttime>=a.admittime AND h.charttime<=a.dischtime\n )\n AND NOT EXISTS (\n    SELECT 1 FROM admissions a\n    WHERE a.subject_id=h.subject_id\n      AND a.edregtime IS NOT NULL AND a.edouttime IS NOT NULL\n      AND a.edouttime>a.edregtime\n      AND h.charttime>=a.edregtime AND h.charttime<=a.edouttime\n )\n AND NOT EXISTS (\n    SELECT 1 FROM valid_ed_transfers e\n    WHERE e.subject_id=h.subject_id\n      AND h.charttime>=e.intime AND h.charttime<=e.outtime\n )\n AND NOT EXISTS (\n    SELECT 1 FROM valid_physical_transfers tr\n    WHERE tr.subject_id=h.subject_id\n      AND h.charttime>=tr.intime AND h.charttime<=tr.outtime\n )\n')
    hist = conc.execute('\nSELECT stay_id,MEDIAN(creatinine) AS hist_strict_median\nFROM hist_strict GROUP BY stay_id\n').df()
    conc.execute("\nCREATE TEMP TABLE current_same_time AS\nSELECT\n t.stay_id,t.admittime,t.intime,t.landmark_time,c.charttime,\n AVG(c.creatinine) creatinine\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.admittime,t.intime,t.landmark_time,c.charttime\n")
    cur = conc.execute('\nSELECT\n  stay_id,\n  ARG_MIN(creatinine,charttime) FILTER(\n    WHERE charttime>=admittime AND charttime<=landmark_time\n  ) AS admission_first_cr,\n  MIN(creatinine) FILTER(\n    WHERE charttime>=admittime AND charttime<=landmark_time\n  ) AS admission_min_cr\nFROM current_same_time\nGROUP BY stay_id\n').df()
    refs = expanded[['stay_id']].merge(hist, on='stay_id', how='left', validate='1:1').merge(cur, on='stay_id', how='left', validate='1:1')
    refs['hier_ref_cr'] = refs.hist_strict_median
    refs['hier_ref_source'] = np.where(refs.hist_strict_median.notna(), 'historical_strict_median', '')
    m = refs.hier_ref_cr.isna() & refs.admission_first_cr.notna()
    refs.loc[m, 'hier_ref_cr'] = refs.loc[m, 'admission_first_cr']
    refs.loc[m, 'hier_ref_source'] = 'admission_first'
    m = refs.hier_ref_cr.isna() & refs.admission_min_cr.notna()
    refs.loc[m, 'hier_ref_cr'] = refs.loc[m, 'admission_min_cr']
    refs.loc[m, 'hier_ref_source'] = 'admission_min_to_landmark'
    refs.loc[refs.hier_ref_cr.isna(), 'hier_ref_source'] = 'unavailable'
    conc.register('refs_df', refs[['stay_id', 'hier_ref_cr']])
    conc.execute('CREATE TEMP TABLE refs_tbl AS SELECT * FROM refs_df')
    hier_events = conc.execute("\nWITH r AS (\n SELECT c.*,\n   MIN(creatinine) OVER(\n     PARTITION BY stay_id ORDER BY charttime\n     RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n     AND INTERVAL '1 microsecond' PRECEDING\n   ) AS low48\n FROM current_same_time c\n)\nSELECT r.*,x.hier_ref_cr\nFROM r JOIN target_lm t USING(stay_id)\nLEFT JOIN refs_tbl x USING(stay_id)\nWHERE r.charttime>=t.intime AND r.charttime<=t.landmark_time\nORDER BY stay_id,charttime\n").df()
    crv = pd.to_numeric(hier_events.creatinine, errors='coerce')
    b = pd.to_numeric(hier_events.hier_ref_cr, errors='coerce')
    low48 = pd.to_numeric(hier_events.low48, errors='coerce')
    ratio15 = b.notna() & crv.ge(1.5 * b)
    ratio20 = b.notna() & crv.ge(2 * b)
    ratio30 = b.notna() & crv.ge(3 * b)
    delta = low48.notna() & crv.ge(low48 + 0.3)
    hier_events['hier_stage_event'] = np.select([ratio30, crv.ge(4.0) & (ratio15 | delta), ratio20, ratio15 | delta], [3.0, 3.0, 2.0, 1.0], default=0.0)
    hier_stage = hier_events.groupby('stay_id', as_index=False).hier_stage_event.max().rename(columns={'hier_stage_event': 'cr_stage_hierarchical_sensitivity'})
    hier = expanded[['stay_id']].merge(hier_stage, on='stay_id', how='left', validate='1:1').merge(refs[['stay_id', 'hier_ref_source', 'hier_ref_cr']], on='stay_id', validate='1:1')
    stages = stages.merge(hier[['stay_id', 'cr_stage_hierarchical_sensitivity']], on='stay_id', validate='1:1')
    q = expanded[['subject_id', 'hadm_id', 'stay_id', 'intime', 'active_rrt_any_by_landmark']].merge(stages[['stay_id', 'cal_mean_ibw', 'cr_stage_hierarchical_sensitivity']], on='stay_id', validate='1:1')
    crh = pd.to_numeric(q.cr_stage_hierarchical_sensitivity, errors='coerce')
    uoh = pd.to_numeric(q.cal_mean_ibw, errors='coerce')
    rrt = q.active_rrt_any_by_landmark.fillna(0).eq(1)
    q['_positive'] = crh.fillna(0).ge(1) | uoh.fillna(0).ge(1) | rrt
    sel_hier = first_positive(q)
    summary9 = pd.DataFrame([membership_compare(sel_mean_ibw, sel_hier, 'hierarchical_reference_SCr_sensitivity', 'standard_cr_plus_mean_ibw')])
    display_table(summary9)
    save_csv(refs, 'hierarchical_creatinine_reference_source.csv')
    save_csv(summary9, 'hierarchical_creatinine_cohort_effect.csv')
    future_tables = []
    for denom in ['ibw', 'abw']:
        for mode in ['mean', 'cons']:
            lab = f'future_{mode}_{denom}'
            future_tables.append(build_stages(f'span_{denom}', mode, future=True, label=lab))
    future = expanded[['stay_id']].copy()
    for t in future_tables:
        future = future.merge(t, on='stay_id', how='left', validate='1:1')
    rows = []
    for standard, alt in [('cal_mean_ibw', 'future_mean_ibw'), ('cal_cons_ibw', 'future_cons_ibw'), ('cal_mean_abw', 'future_mean_abw'), ('cal_cons_abw', 'future_cons_abw')]:
        x = stages[['stay_id', standard]].merge(future[['stay_id', alt]], on='stay_id', validate='1:1')
        rows.append({'comparison': f'{standard}->{alt}', 'target_stage_changed': int((~stage_equal(x[standard], x[alt])).sum()), 'future_stage3_24h_path_is_structurally_unavailable_at_24h_landmark': True})
    summary10 = pd.DataFrame(rows)
    display_table(summary10)
    save_csv(summary10, 'strict_duration_sensitivity.csv')
    PROC_RRT = req(MIMIC_ROOT / 'icu' / 'procedureevents.csv.gz')
    IE_RRT = req(MIMIC_ROOT / 'icu' / 'inputevents.csv.gz')
    CE_RRT = req(MIMIC_ROOT / 'icu' / 'chartevents.csv.gz')
    record_access(PROC_RRT, ['stay_id', 'itemid', 'starttime', 'endtime', 'storetime', 'value', 'statusdescription'], 'procedureevents RRT')
    record_access(IE_RRT, ['stay_id', 'itemid', 'starttime', 'endtime', 'storetime', 'amount', 'statusdescription'], 'inputevents RRT')
    record_access(CE_RRT, ['stay_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum'], 'chartevents RRT')
    RRT_PROC_ACTIVE = (225441, 225802, 225803, 225805, 225809, 225955)
    RRT_INPUT_ACTIVE = (227536, 227525)
    RRT_CHART_ACTIVE = (226499, 224154, 225183, 227438, 224191, 225806, 225807, 228004, 228005, 228006, 224144, 224145, 224153, 226457)
    rrt_stays = expanded.loc[expanded.active_rrt_any_by_landmark.fillna(0).eq(1), ['subject_id', 'stay_id', 'intime', 'landmark_time']].copy()

    def csv_header(path):
        return pd.read_csv(path, nrows=0).columns.tolist()
    proc_header = csv_header(PROC_RRT)
    ie_header = csv_header(IE_RRT)
    proc_status = 'pe.statusdescription' if 'statusdescription' in proc_header else 'NULL::VARCHAR'
    ie_status = 'ie.statusdescription' if 'statusdescription' in ie_header else 'NULL::VARCHAR'
    crr = duckdb.connect(database=':memory:')
    crr.execute('PRAGMA threads=4')
    crr.register('rrt_df', rrt_stays)
    crr.execute('CREATE TEMP TABLE rrt_target AS SELECT * FROM rrt_df')
    crr.execute(f"CREATE VIEW pe AS SELECT * FROM read_csv_auto('{qpath(PROC_RRT)}',header=true,sample_size=200000) WHERE itemid IN {RRT_PROC_ACTIVE}")
    crr.execute(f"CREATE VIEW ie AS SELECT * FROM read_csv_auto('{qpath(IE_RRT)}',header=true,sample_size=200000) WHERE itemid IN {RRT_INPUT_ACTIVE}")
    crr.execute(f"CREATE VIEW ce AS SELECT stay_id,itemid,charttime,storetime,value,valuenum FROM read_csv_auto('{qpath(CE_RRT)}',header=true,sample_size=200000) WHERE itemid IN {RRT_CHART_ACTIVE + (225965,)}")
    ev = crr.execute(f"\nSELECT r.stay_id,'procedureevents' source,pe.itemid,pe.starttime event_time,{proc_status} statusdescription\nFROM rrt_target r JOIN pe USING(stay_id)\nWHERE pe.value IS NOT NULL AND pe.starttime<=r.landmark_time\n  AND (pe.endtime IS NULL OR pe.endtime>=r.intime)\n  AND pe.storetime IS NOT NULL AND pe.storetime<=r.landmark_time\nUNION ALL\nSELECT r.stay_id,'inputevents' source,ie.itemid,ie.starttime,{ie_status}\nFROM rrt_target r JOIN ie USING(stay_id)\nWHERE ie.amount>0 AND ie.starttime<=r.landmark_time\n  AND (ie.endtime IS NULL OR ie.endtime>=r.intime)\n  AND ie.storetime IS NOT NULL AND ie.storetime<=r.landmark_time\nUNION ALL\nSELECT r.stay_id,'chartevents' source,ce.itemid,ce.charttime,NULL::VARCHAR\nFROM rrt_target r JOIN ce USING(stay_id)\nWHERE ce.itemid IN {RRT_CHART_ACTIVE}\n  AND ce.charttime>=r.intime AND ce.charttime<=r.landmark_time\n  AND ce.storetime IS NOT NULL AND ce.storetime<=r.landmark_time\n  AND ce.value IS NOT NULL\nUNION ALL\nSELECT r.stay_id,'pd_in_use' source,ce.itemid,ce.charttime,NULL::VARCHAR\nFROM rrt_target r JOIN ce USING(stay_id)\nWHERE ce.itemid=225965 AND lower(trim(ce.value))='in use'\n  AND ce.charttime>=r.intime AND ce.charttime<=r.landmark_time\n  AND ce.storetime IS NOT NULL AND ce.storetime<=r.landmark_time\n").df()
    ev['status_norm'] = ev.statusdescription.fillna('').astype(str).str.lower().str.strip()
    ev['clearly_non_delivered'] = ev.status_norm.str.contains('rewrit|cancel|error', regex=True, na=False)
    by = ev.groupby('stay_id', as_index=False).agg(evidence_rows=('itemid', 'size'), bad_rows=('clearly_non_delivered', 'sum'), all_evidence_bad=('clearly_non_delivered', 'all'))
    by = rrt_stays[['stay_id']].merge(by, on='stay_id', how='left', validate='1:1')
    by['evidence_rows'] = by.evidence_rows.fillna(0).astype(int)
    by['all_evidence_bad'] = by.all_evidence_bad.fillna(False)
    if by.evidence_rows.eq(0).any():
        raise AssertionError('RRT-positive stay without reconstructed delivery evidence.')

    # Independently assess the accepted outcome-blind RRT evidence classification.
    # This regression check does not alter phenotype assignment and uses the
    # specified source records to assess the target-population safety criterion.
    policy_ev = crr.execute(f"""
SELECT r.subject_id,r.stay_id,'procedureevents' source,pe.itemid,
       pe.starttime event_time,TRY_CAST(pe.value AS DOUBLE) numeric_value
FROM rrt_target r JOIN pe USING(stay_id)
WHERE pe.value IS NOT NULL AND pe.starttime<=r.landmark_time
  AND (pe.endtime IS NULL OR pe.endtime>=r.intime)
  AND pe.storetime IS NOT NULL AND pe.storetime<=r.landmark_time
UNION ALL
SELECT r.subject_id,r.stay_id,'inputevents_crrt' source,ie.itemid,
       ie.starttime,TRY_CAST(ie.amount AS DOUBLE)
FROM rrt_target r JOIN ie USING(stay_id)
WHERE ie.amount>0 AND ie.starttime<=r.landmark_time
  AND (ie.endtime IS NULL OR ie.endtime>=r.intime)
  AND ie.storetime IS NOT NULL AND ie.storetime<=r.landmark_time
UNION ALL
SELECT r.subject_id,r.stay_id,'chartevents_active' source,ce.itemid,
       ce.charttime,TRY_CAST(ce.valuenum AS DOUBLE)
FROM rrt_target r JOIN ce USING(stay_id)
WHERE ce.itemid IN {RRT_CHART_ACTIVE}
  AND ce.charttime>=r.intime AND ce.charttime<=r.landmark_time
  AND ce.storetime IS NOT NULL AND ce.storetime<=r.landmark_time
  AND ce.value IS NOT NULL
UNION ALL
SELECT r.subject_id,r.stay_id,'pd_in_use' source,ce.itemid,
       ce.charttime,NULL::DOUBLE
FROM rrt_target r JOIN ce USING(stay_id)
WHERE ce.itemid=225965 AND lower(trim(ce.value))='in use'
  AND ce.charttime>=r.intime AND ce.charttime<=r.landmark_time
  AND ce.storetime IS NOT NULL AND ce.storetime<=r.landmark_time
""").df()
    policy_numeric = pd.to_numeric(policy_ev['numeric_value'], errors='coerce')
    policy_ev['strong_clearance_original'] = (
        (policy_ev.source.eq('procedureevents') & policy_ev.itemid.isin({225441, 225802, 225803, 225805, 225809}))
        | (policy_ev.source.eq('chartevents_active') & policy_ev.itemid.isin({224153, 224154, 228005, 228006}) & policy_numeric.gt(0))
        | (policy_ev.source.eq('inputevents_crrt') & policy_ev.itemid.isin({227525, 227536}) & policy_numeric.gt(0))
    )
    policy_ev['is_hd_output'] = (
        policy_ev.source.eq('chartevents_active')
        & policy_ev.itemid.eq(226499)
        & policy_numeric.notna()
    )
    policy_ev['definitive_rrt_treatment_evidence'] = policy_ev['strong_clearance_original'] | policy_ev['is_hd_output']
    policy_ev['is_direct_scuf'] = policy_ev.source.eq('procedureevents') & policy_ev.itemid.eq(225955)
    policy_by = policy_ev.groupby(['subject_id', 'stay_id'], as_index=False).agg(
        evidence_rows=('itemid', 'size'),
        has_definitive_rrt=('definitive_rrt_treatment_evidence', 'max'),
        has_direct_scuf=('is_direct_scuf', 'max'),
        has_hd_output=('is_hd_output', 'max'),
    )
    policy_by = rrt_stays[['subject_id', 'stay_id']].merge(
        policy_by, on=['subject_id', 'stay_id'], how='left', validate='1:1'
    )
    for col in ['has_definitive_rrt', 'has_direct_scuf', 'has_hd_output']:
        policy_by[col] = policy_by[col].fillna(False).astype(bool)
    no_definitive = set(policy_by.loc[~policy_by.has_definitive_rrt, 'stay_id'].astype('int64'))
    no_def_detail = policy_ev.loc[policy_ev.stay_id.astype('int64').isin(no_definitive)].copy()
    no_def_detail['is_uf_output_row'] = no_def_detail.source.eq('chartevents_active') & no_def_detail.itemid.eq(226457)
    no_def_by = no_def_detail.groupby('stay_id', as_index=False).agg(
        all_rows_are_uf_output=('is_uf_output_row', 'all')
    )
    rrt_policy_summary = pd.DataFrame([{
        'active_rrt_target_stays': int(len(policy_by)),
        'direct_scuf_target_stays': int(policy_by.has_direct_scuf.sum()),
        'itemid_226499_target_stays': int(policy_by.has_hd_output.sum()),
        'isolated_uf_output_only_target_stays': int(no_def_by.all_rows_are_uf_output.sum()),
        'unknown_weak_rrt_patterns': int((~no_def_by.all_rows_are_uf_output).sum()),
    }])
    display_table(by[['evidence_rows', 'all_evidence_bad']].agg(['count', 'sum']))
    save_csv(ev, 'rrt_status_evidence.csv')
    save_csv(by, 'rrt_stay_summary.csv')
    save_csv(policy_by, 'rrt_policy_evidence_by_stay.csv')
    save_csv(rrt_policy_summary, 'rrt_policy_regression_summary.csv')
    DX = req(MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz')
    PRICD = req(MIMIC_ROOT / 'hosp' / 'procedures_icd.csv.gz')
    record_access(DX, ['subject_id', 'hadm_id', 'icd_version', 'icd_code'], 'diagnoses ICD transplant audit')
    record_access(PRICD, ['subject_id', 'hadm_id', 'icd_version', 'chartdate', 'icd_code'], 'procedures ICD transplant audit')
    for c in ['admittime', 'dischtime', 'edregtime', 'edouttime']:
        adm_allow[c] = pd.to_datetime(adm_allow[c], errors='coerce')
    txbase = post_transplant_allowed[['subject_id', 'hadm_id', 'stay_id', 'admittime']].copy()
    txbase['admittime'] = pd.to_datetime(txbase['admittime'], errors='coerce')
    for c in ['admittime', 'dischtime', 'edregtime', 'edouttime']:
        if not pd.api.types.is_datetime64_any_dtype(adm_allow[c]):
            raise TypeError(f'Admissions timestamp normalization failed for {c}')
    if not pd.api.types.is_datetime64_any_dtype(txbase['admittime']):
        raise TypeError('txbase.admittime timestamp normalization failed')
    if txbase['admittime'].isna().any():
        bad = txbase.loc[txbase['admittime'].isna(), ['subject_id', 'hadm_id', 'stay_id']].copy()
        display_table(bad)
        raise AssertionError(f'Missing/unparseable admittime in {len(bad)} post-transplant override stays.')
    ct = duckdb.connect(database=':memory:')
    ct.register('ov_df', txbase)
    ct.register('adm_df', adm_allow)
    ct.execute('CREATE TEMP TABLE ov AS SELECT * FROM ov_df')
    ct.execute('CREATE TEMP TABLE admissions AS SELECT * FROM adm_df')
    ct.execute(f"\nCREATE TEMP TABLE dx AS\nSELECT\n    subject_id,\n    hadm_id,\n    icd_version,\n    upper(replace(trim(icd_code),'.','')) AS code\nFROM read_csv_auto(\n    '{qpath(DX)}',\n    header=true,\n    sample_size=200000\n)\n")
    ct.execute(f"\nCREATE TEMP TABLE pr AS\nSELECT\n    subject_id,\n    hadm_id,\n    icd_version,\n    chartdate,\n    upper(replace(trim(icd_code),'.','')) AS code\nFROM read_csv_auto(\n    '{qpath(PRICD)}',\n    header=true,\n    sample_size=200000\n)\n")
    dtype_check = ct.execute('\nSELECT\n    typeof(o.admittime) AS ov_admittime_type,\n    typeof(a.admittime) AS admissions_admittime_type,\n    typeof(a.dischtime) AS admissions_dischtime_type\nFROM ov o\nLEFT JOIN admissions a\n  ON a.subject_id=o.subject_id\nLIMIT 1\n').df()
    display_table(dtype_check)
    if any(('VARCHAR' in str(x).upper() for x in dtype_check.iloc[0].tolist())):
        raise TypeError(f'Timestamp column still registered as VARCHAR: {dtype_check.iloc[0].to_dict()}')
    tx = ct.execute("\nWITH current AS (\n    SELECT\n        o.stay_id,\n\n        MAX(\n            CASE\n                WHEN\n                    (d.icd_version=9 AND d.code IN ('5856','V4511'))\n                    OR\n                    (d.icd_version=10 AND d.code IN ('N186','Z992'))\n                THEN 1 ELSE 0\n            END\n        ) AS current_chronic_krt,\n\n        MAX(\n            CASE\n                WHEN d.icd_version=10 AND d.code='T8612'\n                THEN 1 ELSE 0\n            END\n        ) AS current_tx_failure,\n\n        MAX(\n            CASE\n                WHEN\n                    (d.icd_version=9 AND d.code='99681')\n                    OR\n                    (d.icd_version=10 AND d.code LIKE 'T861%')\n                THEN 1 ELSE 0\n            END\n        ) AS current_tx_complication\n\n    FROM ov o\n    LEFT JOIN dx d\n      ON d.subject_id=o.subject_id\n     AND d.hadm_id=o.hadm_id\n    GROUP BY o.stay_id\n),\n\nhist AS (\n    SELECT\n        o.stay_id,\n\n        MAX(\n            CASE\n                WHEN\n                    (d.icd_version=9 AND d.code='99681')\n                    OR\n                    (d.icd_version=10 AND d.code LIKE 'T861%')\n                THEN a.dischtime\n            END\n        ) AS prior_tx_complication_time,\n\n        MAX(\n            CASE\n                WHEN d.icd_version=10 AND d.code='T8612'\n                THEN a.dischtime\n            END\n        ) AS prior_tx_failure_time\n\n    FROM ov o\n    JOIN admissions a\n      ON a.subject_id=o.subject_id\n    JOIN dx d\n      ON d.subject_id=a.subject_id\n     AND d.hadm_id=a.hadm_id\n\n    WHERE a.hadm_id<>o.hadm_id\n      AND a.dischtime IS NOT NULL\n      AND a.dischtime<=o.admittime\n\n    GROUP BY o.stay_id\n)\n\nSELECT\n    o.*,\n    c.current_chronic_krt,\n    c.current_tx_failure,\n    c.current_tx_complication,\n    h.prior_tx_complication_time,\n    h.prior_tx_failure_time\n\nFROM ov o\nLEFT JOIN current c USING(stay_id)\nLEFT JOIN hist h USING(stay_id)\n").df()
    tx = tx.merge(post_transplant[['stay_id', 'latest_prior_tx_status_time', 'latest_prior_tx_proc_time', 'latest_prior_tx_failure_time']], on='stay_id', how='left', validate='1:1')
    tx['latest_prior_tx_time'] = pd.concat([pd.to_datetime(tx.latest_prior_tx_status_time, errors='coerce'), pd.to_datetime(tx.latest_prior_tx_proc_time, errors='coerce')], axis=1).max(axis=1)
    tx['failure_after_latest_tx'] = pd.to_datetime(tx.prior_tx_failure_time, errors='coerce').notna() & pd.to_datetime(tx.latest_prior_tx_time, errors='coerce').notna() & (pd.to_datetime(tx.prior_tx_failure_time, errors='coerce') >= pd.to_datetime(tx.latest_prior_tx_time, errors='coerce'))
    tx['complication_after_latest_tx'] = pd.to_datetime(tx.prior_tx_complication_time, errors='coerce').notna() & pd.to_datetime(tx.latest_prior_tx_time, errors='coerce').notna() & (pd.to_datetime(tx.prior_tx_complication_time, errors='coerce') >= pd.to_datetime(tx.latest_prior_tx_time, errors='coerce'))
    tx['hard_contradiction'] = tx.current_chronic_krt.fillna(0).eq(1) | tx.current_tx_failure.fillna(0).eq(1) | tx.failure_after_latest_tx.fillna(False)
    summary12 = pd.DataFrame([{'override_target_stays': len(tx), 'hard_contradictions': int(tx.hard_contradiction.sum()), 'current_or_later_complication_context': int((tx.current_tx_complication.fillna(0).eq(1) | tx.complication_after_latest_tx.fillna(False)).sum())}])
    display_table(summary12)
    save_csv(summary12, 'transplant_override_summary.csv')
    save_csv(tx, 'transplant_override_detail.csv')
    print('POST-TRANSPLANT CONTRADICTION AUDIT: PASS')
    conc.execute("\nCREATE OR REPLACE TEMP TABLE hist_same_time_minmax AS\nSELECT\n    t.stay_id,\n    t.subject_id,\n    t.admittime,\n    t.landmark_time,\n    c.charttime,\n\n    AVG(c.creatinine) AS cr_avg,\n    MIN(c.creatinine) AS cr_min,\n    MAX(c.creatinine) AS cr_max,\n\n    COUNT(*) AS raw_rows,\n\n    MAX(\n        CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END\n    ) AS has_labevent_hadm_id\n\nFROM target_lm t\nJOIN creat_all c\n  ON c.subject_id=t.subject_id\n\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\n\nGROUP BY\n    t.stay_id,\n    t.subject_id,\n    t.admittime,\n    t.landmark_time,\n    c.charttime\n")
    conc.execute('\nCREATE OR REPLACE TEMP TABLE hist_mm_strict AS\nSELECT h.*\n\nFROM hist_same_time_minmax h\n\nWHERE has_labevent_hadm_id=0\n\n  AND NOT EXISTS (\n      SELECT 1\n      FROM admissions a\n      WHERE a.subject_id=h.subject_id\n        AND a.admittime IS NOT NULL\n        AND a.dischtime IS NOT NULL\n        AND h.charttime>=a.admittime\n        AND h.charttime<=a.dischtime\n  )\n\n  AND NOT EXISTS (\n      SELECT 1\n      FROM admissions a\n      WHERE a.subject_id=h.subject_id\n        AND a.edregtime IS NOT NULL\n        AND a.edouttime IS NOT NULL\n        AND a.edouttime>a.edregtime\n        AND h.charttime>=a.edregtime\n        AND h.charttime<=a.edouttime\n  )\n\n  AND NOT EXISTS (\n      SELECT 1\n      FROM valid_ed_transfers e\n      WHERE e.subject_id=h.subject_id\n        AND h.charttime>=e.intime\n        AND h.charttime<=e.outtime\n  )\n\n  AND NOT EXISTS (\n      SELECT 1\n      FROM valid_physical_transfers tr\n      WHERE tr.subject_id=h.subject_id\n        AND h.charttime>=tr.intime\n        AND h.charttime<=tr.outtime\n  )\n')
    histmm = conc.execute('\nSELECT\n    stay_id,\n\n    MEDIAN(cr_avg) AS hist_avg,\n    MEDIAN(cr_min) AS hist_min,\n    MEDIAN(cr_max) AS hist_max,\n\n    SUM(\n        CASE\n            WHEN raw_rows>1 AND cr_max>cr_min THEN 1\n            ELSE 0\n        END\n    ) AS discordant_duplicate_hist_charttimes\n\nFROM hist_mm_strict\nGROUP BY stay_id\n').df()
    conc.execute("\nCREATE OR REPLACE TEMP TABLE cur_mm AS\nSELECT\n    t.stay_id,\n    t.intime,\n    t.landmark_time,\n    c.charttime,\n\n    AVG(c.creatinine) AS cr_avg,\n    MIN(c.creatinine) AS cr_min,\n    MAX(c.creatinine) AS cr_max,\n\n    COUNT(*) AS raw_rows\n\nFROM target_lm t\nJOIN creat_all c\n  ON c.subject_id=t.subject_id\n\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\n\nGROUP BY\n    t.stay_id,\n    t.intime,\n    t.landmark_time,\n    c.charttime\n")
    dup13 = conc.execute('\nSELECT\n    COUNT(*) AS same_charttime_rows,\n\n    SUM(\n        CASE WHEN raw_rows>1 THEN 1 ELSE 0 END\n    ) AS duplicate_charttimes,\n\n    SUM(\n        CASE WHEN raw_rows>1 AND cr_max>cr_min THEN 1 ELSE 0 END\n    ) AS discordant_duplicate_charttimes,\n\n    COUNT(\n        DISTINCT CASE\n            WHEN raw_rows>1 AND cr_max>cr_min THEN stay_id\n        END\n    ) AS stays_with_discordant_duplicates\n\nFROM cur_mm\n').df()
    try:
        conc.unregister('histmm_df')
    except Exception:
        pass
    conc.register('histmm_df', histmm)
    conc.execute('\nCREATE OR REPLACE TEMP TABLE histmm_tbl AS\nSELECT * FROM histmm_df\n')

    def mm_stage(agg):
        ccol = {'AVG': 'cr_avg', 'MIN': 'cr_min', 'MAX': 'cr_max'}[agg]
        bcol = {'AVG': 'hist_avg', 'MIN': 'hist_min', 'MAX': 'hist_max'}[agg]
        q = f"\n    WITH refs AS (\n        SELECT\n            *,\n\n            MIN({ccol}) OVER(\n                PARTITION BY stay_id\n                ORDER BY charttime\n                RANGE BETWEEN\n                    INTERVAL '48 hours' PRECEDING\n                    AND INTERVAL '1 microsecond' PRECEDING\n            ) AS low48,\n\n            MIN({ccol}) OVER(\n                PARTITION BY stay_id\n                ORDER BY charttime\n                RANGE BETWEEN\n                    INTERVAL '7 days' PRECEDING\n                    AND INTERVAL '1 microsecond' PRECEDING\n            ) AS low7\n\n        FROM cur_mm\n    ),\n\n    ev AS (\n        SELECT\n            r.*,\n            h.{bcol} AS histbase,\n\n            -- Rolling 48-hour / 7-day KDIGO creatinine semantics.\n            CASE\n\n                WHEN low48 IS NULL\n                 AND low7 IS NULL\n                THEN NULL\n\n                WHEN low7 IS NOT NULL\n                 AND {ccol}>=3.0*low7\n                THEN 3\n\n                WHEN {ccol}>=4.0\n                 AND (\n                    (low48 IS NOT NULL AND {ccol}>=low48+0.3)\n                    OR\n                    (low7 IS NOT NULL AND {ccol}>=1.5*low7)\n                 )\n                THEN 3\n\n                WHEN low7 IS NOT NULL\n                 AND {ccol}>=2.0*low7\n                THEN 2\n\n                WHEN low48 IS NOT NULL\n                 AND {ccol}>=low48+0.3\n                THEN 1\n\n                WHEN low7 IS NOT NULL\n                 AND {ccol}>=1.5*low7\n                THEN 1\n\n                ELSE 0\n\n            END AS rolling_stage,\n\n            -- Strict historical-baseline component.\n            --\n            -- Important:\n            -- delta-only historical phenotype logic is NOT duplicated here because the historical\n            -- component is combined with the independent rolling\n            -- component. The 48-h delta is used only in the >=4 mg/dL Stage-3\n            -- branch, matching the strict-outpatient implementation.\n            CASE\n\n                WHEN h.{bcol} IS NULL\n                THEN NULL\n\n                WHEN {ccol}>=3.0*h.{bcol}\n                THEN 3\n\n                WHEN {ccol}>=4.0\n                 AND (\n                    {ccol}>=1.5*h.{bcol}\n                    OR\n                    (low48 IS NOT NULL AND {ccol}>=low48+0.3)\n                 )\n                THEN 3\n\n                WHEN {ccol}>=2.0*h.{bcol}\n                THEN 2\n\n                WHEN {ccol}>=1.5*h.{bcol}\n                THEN 1\n\n                ELSE 0\n\n            END AS hist_stage\n\n        FROM refs r\n\n        LEFT JOIN histmm_tbl h\n          USING(stay_id)\n\n        WHERE r.charttime>=r.intime\n          AND r.charttime<=r.landmark_time\n    ),\n\n    stay_stage AS (\n        SELECT\n            stay_id,\n            MAX(rolling_stage) AS rolling_stage,\n            MAX(hist_stage) AS hist_stage\n        FROM ev\n        GROUP BY stay_id\n    )\n\n    SELECT\n        t.stay_id,\n\n        s.rolling_stage,\n        s.hist_stage,\n\n        CASE\n\n            WHEN s.rolling_stage IS NULL\n             AND s.hist_stage IS NULL\n            THEN NULL\n\n            ELSE GREATEST(\n                COALESCE(s.rolling_stage,-1),\n                COALESCE(s.hist_stage,-1)\n            )\n\n        END AS combined_stage\n\n    FROM target_lm t\n\n    LEFT JOIN stay_stage s\n      USING(stay_id)\n\n    ORDER BY t.stay_id\n    "
        out = conc.execute(q).df()
        return out.rename(columns={'rolling_stage': f'rolling_{agg.lower()}', 'hist_stage': f'hist_{agg.lower()}', 'combined_stage': f'cr_{agg.lower()}'})
    mm = expanded[['stay_id']].copy()
    for agg in ['AVG', 'MIN', 'MAX']:
        z = mm_stage(agg)
        mm = mm.merge(z, on='stay_id', how='left', validate='1:1')
    avg_chk = expanded[['stay_id', 'target_source', 'cr_stage']].merge(mm[['stay_id', 'rolling_avg', 'hist_avg', 'cr_avg']], on='stay_id', how='left', validate='1:1')
    avg_same = stage_equal(avg_chk.cr_stage, avg_chk.cr_avg)
    avg_mismatch_n = int((~avg_same).sum())
    print('AVG creatinine reproduction mismatch:', avg_mismatch_n)
    if avg_mismatch_n:
        bad = avg_chk.loc[~avg_same].copy()
        save_csv(bad, 'creatinine_average_reproduction_mismatch.csv')
        display_table(bad.groupby(['target_source', 'cr_stage', 'cr_avg'], dropna=False).size().reset_index(name='n'))
        raise AssertionError(f'AVG creatinine stress reconstruction mismatch in {avg_mismatch_n} stays. Do NOT interpret MIN/MAX until AVG mismatch is zero.')
    print('AVG creatinine reconstruction: EXACT PASS')
    summary13 = pd.DataFrame([{'variant': 'MIN', 'target_stage_changed': int((~stage_equal(mm.cr_avg, mm.cr_min)).sum()), 'positive_added_vs_AVG': int((pd.to_numeric(mm.cr_avg, errors='coerce').fillna(0).lt(1) & pd.to_numeric(mm.cr_min, errors='coerce').fillna(0).ge(1)).sum()), 'positive_removed_vs_AVG': int((pd.to_numeric(mm.cr_avg, errors='coerce').fillna(0).ge(1) & pd.to_numeric(mm.cr_min, errors='coerce').fillna(0).lt(1)).sum())}, {'variant': 'MAX', 'target_stage_changed': int((~stage_equal(mm.cr_avg, mm.cr_max)).sum()), 'positive_added_vs_AVG': int((pd.to_numeric(mm.cr_avg, errors='coerce').fillna(0).lt(1) & pd.to_numeric(mm.cr_max, errors='coerce').fillna(0).ge(1)).sum()), 'positive_removed_vs_AVG': int((pd.to_numeric(mm.cr_avg, errors='coerce').fillna(0).ge(1) & pd.to_numeric(mm.cr_max, errors='coerce').fillna(0).lt(1)).sum())}])
    display_table(dup13)
    display_table(summary13)
    save_csv(dup13, 'creatinine_duplicate_summary.csv')
    save_csv(summary13, 'creatinine_aggregation_effect.csv')
    save_csv(mm, 'creatinine_aggregation_by_stay.csv')
    print('CREATININE AGGREGATION AUDIT: PASS')
    covstay = hourly.groupby('stay_id', as_index=False).agg(generated_calendar_hours=('hour_end', 'size'), valid_span_ibw=('span_ibw', lambda s: int(pd.to_numeric(s, errors='coerce').notna().sum())), valid_span_abw=('span_abw', lambda s: int(pd.to_numeric(s, errors='coerce').notna().sum())), multi_compartment_hours=('compartments_in_hour', lambda s: int(pd.to_numeric(s, errors='coerce').fillna(0).ge(2).sum())))
    covstay = covstay.merge(stages[['stay_id', 'cal_mean_ibw_max_consecutive_valid', 'cal_mean_abw_max_consecutive_valid']], on='stay_id', how='left', validate='1:1').merge(expanded[['stay_id', 'height_cm', 'ibw_kg', 'admission_weight_kg', 'cr_stage', 'active_rrt_any_by_landmark', 'uo_unreliable_by_landmark']], on='stay_id', validate='1:1')
    missing14 = pd.DataFrame([{'expanded_target_stays': len(expanded), 'missing_height': int(pd.to_numeric(expanded.height_cm, errors='coerce').isna().sum()), 'missing_or_nonpositive_ibw': int((~pd.to_numeric(expanded.ibw_kg, errors='coerce').gt(0)).sum()), 'missing_or_nonpositive_abw': int((~pd.to_numeric(expanded.admission_weight_kg, errors='coerce').gt(0)).sum()), 'uo_unreliable_stays': int(expanded.uo_unreliable_by_landmark.fillna(0).eq(1).sum()), 'no_valid_IBW_hour': int(covstay.valid_span_ibw.eq(0).sum()), 'IBW_max_consecutive_lt6': int(pd.to_numeric(covstay.cal_mean_ibw_max_consecutive_valid, errors='coerce').fillna(0).lt(6).sum()), 'ABW_max_consecutive_lt6': int(pd.to_numeric(covstay.cal_mean_abw_max_consecutive_valid, errors='coerce').fillna(0).lt(6).sum())}])

    def entry_pattern(selected, stage_col):
        q = selected[['subject_id', 'stay_id']].merge(expanded[['stay_id', 'cr_stage', 'active_rrt_any_by_landmark']], on='stay_id', validate='1:1').merge(stages[['stay_id', stage_col]], on='stay_id', validate='1:1')
        cr = pd.to_numeric(q.cr_stage, errors='coerce').fillna(0).ge(1)
        uo = pd.to_numeric(q[stage_col], errors='coerce').fillna(0).ge(1)
        rr = q.active_rrt_any_by_landmark.fillna(0).eq(1)
        q['pattern'] = 'Cr' + cr.astype(int).astype(str) + '_UO' + uo.astype(int).astype(str) + '_RRT' + rr.astype(int).astype(str)
        return q.pattern.value_counts().rename_axis('pattern').reset_index(name='subjects')
    patterns = []
    for sel, stage_col, label in [(sel_mean_ibw, 'cal_mean_ibw', 'mean_ibw'), (sel_cons_ibw, 'cal_cons_ibw', 'consecutive_ibw'), (sel_mean_abw, 'cal_mean_abw', 'mean_abw'), (sel_cons_abw, 'cal_cons_abw', 'consecutive_abw')]:
        p = entry_pattern(sel, stage_col)
        p['variant'] = label
        patterns.append(p)
    patterns = pd.concat(patterns, ignore_index=True)
    display_table(missing14)
    display_table(patterns)
    save_csv(missing14, 'ascertainment_coverage_summary.csv')
    save_csv(covstay, 'ascertainment_coverage_by_stay.csv')
    save_csv(patterns, 'ascertainment_entry_patterns.csv')
    adjudication_variants = [
        ('calendar_hour_mean_ibw_post_transplant', sel_mean_ibw),
        ('calendar_hour_consecutive_ibw_post_transplant', sel_cons_ibw),
        ('calendar_hour_mean_abw_post_transplant', sel_mean_abw),
        ('calendar_hour_consecutive_abw_post_transplant', sel_cons_abw),
        ('calendar_hour_mean_abw_25_300kg_post_transplant', sel_mean_abw_b),
        ('calendar_hour_consecutive_abw_25_300kg_post_transplant', sel_cons_abw_b),
    ]
    variant_summary = []
    for label, sel in adjudication_variants:
        stage = pd.to_numeric(sel['_combined_stage'], errors='coerce')
        variant_summary.append({
            'variant': label,
            'subjects': len(sel),
            'membership_sha256': membership_sha(sel),
            'kdigo_stage_1_n': int(stage.eq(1).sum()),
            'kdigo_stage_2_n': int(stage.eq(2).sum()),
            'kdigo_stage_3_n': int(stage.eq(3).sum()),
        })
    variant_summary = pd.DataFrame(variant_summary)
    save_csv(variant_summary, 'adjudication_variant_summary.csv')
    save_csv(pd.DataFrame(INPUT_ACCESS).drop_duplicates(), 'adjudication_input_allowlist.csv')

    checks_completed = [
        'upstream_cohort_guards',
        'calendar_hour_urine_output_reproduction',
        'body_weight_denominator_comparison',
        'fixed_stay_comparison',
        'first_eligible_timezero_comparison',
        'urine_output_ascertainability',
        'elapsed_duration_guard',
        'interval_union_coverage',
        'mean_vs_consecutive_comparison',
        'hierarchical_creatinine_reference',
        'strict_duration_sensitivity',
        'rrt_status_evidence',
        'post_transplant_contradiction_audit',
        'creatinine_aggregation_audit',
        'ascertainment_coverage_summary',
    ]
    manifest = {
        'status': 'PASS',
        'outcome_blind': True,
        'strict_outpatient_cohort_n': int(len(strict_index)),
        'consecutive_reference_cohort_n': int(len(reference_index)),
        'expanded_target_stays': int(len(expanded)),
        'post_transplant_temporal_stays': int(len(post_transplant_allowed)),
        'checks_completed': checks_completed,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    save_json(manifest, 'adjudication_manifest.json')


    # Persist the adjudicated primary phenotype for downstream cohort assembly.
    primary_target = target_em.merge(
        stages[['stay_id', 'elapsed_cons_ibw', 'cal_mean_ibw_max_consecutive_valid']],
        on='stay_id', how='left', validate='1:1'
    ).merge(
        expanded[['stay_id', 'ibw_kg', 'uo_unreliable_by_landmark']],
        on='stay_id', how='left', validate='1:1'
    )
    primary_target = primary_target.rename(columns={
        'cr_stage': 'creatinine_stage',
        'cr_evaluable': 'creatinine_evaluable',
        'elapsed_mean_ibw': 'urine_output_stage',
        'elapsed_cons_ibw': 'urine_output_stage_consecutive_sensitivity',
        '_positive': 'aki_positive',
        '_combined_stage': 'kdigo_stage',
        '_variant': 'phenotype_variant',
        'cal_mean_ibw_max_consecutive_valid': 'urine_output_max_consecutive_valid_hours',
    })
    primary_target['urine_output_ascertainable'] = (
        pd.to_numeric(primary_target.ibw_kg, errors='coerce').gt(0)
        & pd.to_numeric(primary_target.urine_output_max_consecutive_valid_hours, errors='coerce').fillna(0).ge(6)
        & ~primary_target.uo_unreliable_by_landmark.fillna(0).eq(1)
    )
    primary_target['aki_negative_confirmed'] = (
        ~primary_target.aki_positive.fillna(False).astype(bool)
        & primary_target.creatinine_evaluable.fillna(0).eq(1)
        & primary_target.urine_output_ascertainable
    )
    primary_target['aki_status'] = np.select(
        [primary_target.aki_positive.fillna(False).astype(bool), primary_target.aki_negative_confirmed],
        ['positive', 'confirmed_negative'],
        default='no_positive_criterion_identified_incomplete_ascertainment',
    )
    identity_cols = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time']
    phenotype_cols = [
        'stay_id', 'target_source', 'creatinine_stage', 'creatinine_evaluable',
        'urine_output_stage', 'urine_output_stage_consecutive_sensitivity',
        'urine_output_max_consecutive_valid_hours', 'urine_output_ascertainable',
        'uo_unreliable_by_landmark', 'ibw_kg', 'active_rrt_any_by_landmark',
        'aki_positive', 'kdigo_stage', 'aki_status'
    ]
    primary_index = sel_em[identity_cols].merge(
        primary_target[phenotype_cols], on='stay_id', how='left', validate='1:1'
    )
    primary_target.to_parquet(AUDIT / 'adjudicated_aki_target_preoutcome.parquet', index=False)
    primary_index.to_parquet(AUDIT / 'adjudicated_primary_cohort_preoutcome.parquet', index=False)
    sel_ec.to_parquet(AUDIT / 'consecutive_urine_output_cohort_preoutcome.parquet', index=False)

    print('=' * 78)
    print('OUTCOME-BLIND URINE-OUTPUT/KDIGO ADJUDICATION: PASS')
    print('=' * 78)
    display_table(variant_summary)


if __name__ == "__main__":
    main()
