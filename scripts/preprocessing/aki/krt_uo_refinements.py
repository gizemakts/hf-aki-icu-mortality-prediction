#!/usr/bin/env python3
"""12 Krt Uo Refinements.

Combine KRT and urine-output refinements.

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
    try:
        import duckdb
    except Exception as e:
        raise ImportError('duckdb is required; install project requirements before running.') from e
    EXPECTED_STRICT_N = 3422
    EXPECTED_STRICT_TARGET_N = 9972
    EXPECTED_EXPANDED_TARGET_N = 10083
    EXPECTED_CONSECUTIVE_N = 2972
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT))).resolve()
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', str(_RAW_DATA_ROOT))).resolve()
    if not MIMIC_ROOT.is_dir():
        raise FileNotFoundError(f'MIMIC-IV root not found: {MIMIC_ROOT}')
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    STRICT_DIR = PHENOTYPE_DIR / 'historical_creatinine' / 'strict_outpatient'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    POST_TRANSPLANT_DIR = PHENOTYPE_DIR / 'post_transplant'
    PERSISTENT_OLIGURIA_DIR = PHENOTYPE_DIR / 'persistent_oliguria'
    OUT = PHENOTYPE_DIR / 'consecutive_urine_output'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'consecutive_urine_output'
    OUT.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    def req(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def first_positive(df, col='aki_positive_consecutive_urine_output'):
        q = df.loc[df[col]].sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
        return q.drop_duplicates('subject_id', keep='first').copy()

    def membership_compare(canon, alt, label):
        cm = dict(zip(canon.subject_id.astype(int), canon.stay_id.astype(int)))
        am = dict(zip(alt.subject_id.astype(int), alt.stay_id.astype(int)))
        cs, as_ = (set(cm), set(am))
        common = cs & as_
        repl = sum((cm[s] != am[s] for s in common))
        return {'variant': label, 'subjects': len(alt), 'subject_added_vs_strict_outpatient': len(as_ - cs), 'subject_dropped_vs_strict_outpatient': len(cs - as_), 'common_subjects': len(common), 'same_subject_replaced_stay': int(repl), 'same_subject_same_stay': int(len(common) - repl), 'stay_symmetric_difference': len(set(cm.values()) ^ set(am.values()))}

    def wcsv(df, name):
        p = OUT / name
        df.to_csv(p, index=False)
        return p

    def wjson(obj, name):
        p = OUT / name
        p.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
        return p
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('MIMIC_ROOT:', MIMIC_ROOT)
    print('OUT:', OUT)
    P_STRICT_TARGET = req(STRICT_DIR / 'strict_outpatient_target.parquet')
    P_STRICT_INDEX = req(STRICT_DIR / 'strict_outpatient_cohort_preoutcome.parquet')
    P_UOBASE = req(URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet')
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    P_POST_TRANSPLANT = req(POST_TRANSPLANT_DIR / 'post_transplant_stay_reconstruction.csv')
    P_POST_TRANSPLANT_REVIEW = req(POST_TRANSPLANT_DIR / 'post_transplant_review_detail.csv')
    P_POST_TRANSPLANT_MANIFEST = req(POST_TRANSPLANT_DIR / 'post_transplant_manifest.json')
    P_OLIGURIA_STAGES = req(PERSISTENT_OLIGURIA_DIR / 'persistent_oliguria_stages.csv')
    P_OLIGURIA_SUMMARY = req(PERSISTENT_OLIGURIA_DIR / 'persistent_oliguria_cohort_effect.csv')
    P_OLIGURIA_MANIFEST = req(PERSISTENT_OLIGURIA_DIR / 'persistent_oliguria_manifest.json')
    post_transplant_manifest = json.loads(P_POST_TRANSPLANT_MANIFEST.read_text())
    oliguria_manifest = json.loads(P_OLIGURIA_MANIFEST.read_text())
    if not post_transplant_manifest.get('outcome_blind') or not oliguria_manifest.get('outcome_blind'):
        raise AssertionError('Upstream post-transplant or persistent-oliguria output is not outcome-blind.')
    strict_target = pd.read_parquet(P_STRICT_TARGET)
    strict_idx = pd.read_parquet(P_STRICT_INDEX, columns=['subject_id', 'stay_id', 'intime'])
    uobase = pd.read_parquet(P_UOBASE, columns=['stay_id', 'ibw_kg', 'height_cm', 'gender'])
    if len(strict_target) != EXPECTED_STRICT_TARGET_N:
        raise AssertionError(f'Strict-outpatient target N={len(strict_target)}, expected {EXPECTED_STRICT_TARGET_N}.')
    if len(strict_idx) != EXPECTED_STRICT_N:
        raise AssertionError(f'Strict-outpatient cohort N={len(strict_idx)}, expected {EXPECTED_STRICT_N}.')
    if strict_idx.subject_id.duplicated().any() or strict_idx.stay_id.duplicated().any():
        raise AssertionError('Strict-outpatient cohort uniqueness failed.')
    oliguria_stages = pd.read_csv(P_OLIGURIA_STAGES)
    oliguria_summary = pd.read_csv(P_OLIGURIA_SUMMARY)
    if len(oliguria_stages) != EXPECTED_STRICT_TARGET_N:
        raise AssertionError('Persistent-oliguria target-stage count mismatch.')
    reference_row = oliguria_summary.loc[oliguria_summary.variant.eq('reference_urine_output')]
    if len(reference_row) != 1:
        raise AssertionError('Persistent-oliguria reference comparator missing.')
    rr = reference_row.iloc[0]
    if int(rr.subjects) != EXPECTED_STRICT_N or int(rr.subject_added_vs_strict_outpatient) != 0 or int(rr.subject_dropped_vs_strict_outpatient) != 0:
        raise AssertionError('Persistent-oliguria step failed strict-outpatient self-reproduction.')
    post_transplant_review = pd.read_csv(P_POST_TRANSPLANT_REVIEW)
    post_transplant_reconstruction = pd.read_csv(P_POST_TRANSPLANT)
    print('Strict-outpatient + post-transplant + persistent-oliguria hard guards: PASS')
    override = post_transplant_review.loc[post_transplant_review.post_transplant_temporal_allow_acute_rrt.astype(bool)].copy()
    if override.stay_id.duplicated().any():
        raise AssertionError('Override stays not unique.')
    needed = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'kdigo_creatinine_stage_post_transplant', 'creatinine_evaluable_post_transplant', 'active_rrt_any_by_landmark', 'uo_unreliable_by_landmark', 'current_chronic_krt_state_code', 'current_tx_failure']
    miss = [c for c in needed if c not in post_transplant_reconstruction.columns]
    if miss:
        raise KeyError(f'Post-transplant reconstruction missing: {miss}')
    ov = override[['stay_id']].merge(post_transplant_reconstruction[needed], on='stay_id', how='left', validate='1:1')
    if ov.kdigo_creatinine_stage_post_transplant.isna().all():
        print('Warning: no override alternative has evaluable creatinine; UO/RRT may still qualify.')
    overlap = set(strict_target.stay_id.astype(int)) & set(ov.stay_id.astype(int))
    if overlap:
        raise AssertionError(f'Override stays overlap reference target: {len(overlap)}')
    print('Post-transplant override stays:', len(ov))
    print('Unique override subjects:', ov.subject_id.nunique())
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    COMPARTMENT_CASE = "\nCASE\n WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n ELSE NULL\nEND\n"
    ov = ov.copy()
    ov['intime'] = pd.to_datetime(ov['intime'], errors='raise')
    ov['landmark_time'] = pd.to_datetime(ov['landmark_time'], errors='raise')
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('ov_df', ov[['stay_id', 'intime', 'landmark_time']])
    con.execute('\nCREATE TEMP TABLE target AS\nSELECT * FROM ov_df\n')
    con.execute(f"\nCREATE VIEW oe AS\nSELECT\n    stay_id,\n    itemid,\n    charttime,\n    storetime,\n    value\nFROM read_csv_auto(\n    '{qpath(OE)}',\n    header=true,\n    sample_size=200000\n)\n")
    con.execute(f'\nCREATE TEMP TABLE uo_events AS\nSELECT\n    t.stay_id,\n    {COMPARTMENT_CASE} AS compartment,\n    oe.charttime,\n    MAX(oe.storetime) AS storetime,\n    SUM(oe.value) AS volume_ml\nFROM target t\nJOIN oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL\n  AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime >= CAST(t.intime AS TIMESTAMP)\n  AND oe.charttime <= CAST(t.landmark_time AS TIMESTAMP)\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime <= CAST(t.landmark_time AS TIMESTAMP)\nGROUP BY\n    t.stay_id,\n    compartment,\n    oe.charttime\nHAVING compartment IS NOT NULL\n')
    con.execute('\nCREATE TEMP TABLE raw_intervals AS\nSELECT\n    stay_id,\n    compartment,\n    LAG(charttime) OVER(\n        PARTITION BY stay_id,compartment\n        ORDER BY charttime\n    ) AS interval_start,\n    charttime AS interval_end,\n    volume_ml\nFROM uo_events\n')
    intervals = con.execute("\nSELECT\n    r.stay_id,\n    r.compartment,\n    r.interval_start,\n    r.interval_end,\n    r.volume_ml,\n    DATE_DIFF(\n        'second',\n        r.interval_start,\n        r.interval_end\n    )/3600.0 AS duration_hours,\n    r.volume_ml / (\n        DATE_DIFF(\n            'second',\n            r.interval_start,\n            r.interval_end\n        )/3600.0\n    ) AS rate_ml_hr\nFROM raw_intervals r\nJOIN target t USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end > r.interval_start\n  AND r.interval_start >= CAST(t.intime AS TIMESTAMP)\n  AND r.interval_end <= CAST(t.landmark_time AS TIMESTAMP)\n  AND DATE_DIFF(\n        'second',\n        r.interval_start,\n        r.interval_end\n      )/3600.0 <= 6.0 + 1e-12\nORDER BY\n    r.stay_id,\n    r.interval_start\n").df()
    hour_rows = []
    for r in ov[['stay_id', 'intime', 'landmark_time']].itertuples(index=False):
        cur = pd.Timestamp(r.intime).floor('h')
        lm = pd.Timestamp(r.landmark_time)
        while cur + pd.Timedelta(hours=1) <= lm:
            hour_rows.append((int(r.stay_id), cur, cur + pd.Timedelta(hours=1)))
            cur += pd.Timedelta(hours=1)
    hours = pd.DataFrame(hour_rows, columns=['stay_id', 'hour_start', 'hour_end'])
    conh = duckdb.connect(database=':memory:')
    conh.register('hours_df', hours)
    conh.register('intervals_df', intervals)
    conh.execute('\nCREATE TEMP TABLE hours AS\nSELECT * FROM hours_df\n')
    conh.execute('\nCREATE TEMP TABLE intervals AS\nSELECT * FROM intervals_df\n')
    hourly = conh.execute("\nWITH ovl AS (\n    SELECT\n        h.stay_id,\n        h.hour_start,\n        h.hour_end,\n        i.interval_start,\n        i.interval_end,\n        i.rate_ml_hr,\n        GREATEST(\n            i.interval_start,\n            h.hour_start\n        ) AS ov_start,\n        LEAST(\n            i.interval_end,\n            h.hour_end\n        ) AS ov_end\n    FROM hours h\n    LEFT JOIN intervals i\n      ON i.stay_id = h.stay_id\n     AND i.interval_end > h.hour_start\n     AND i.interval_start < h.hour_end\n),\n\ncalc AS (\n    SELECT\n        stay_id,\n        hour_start,\n        hour_end,\n\n        SUM(\n            CASE\n                WHEN ov_end > ov_start\n                THEN rate_ml_hr *\n                     DATE_DIFF(\n                         'second',\n                         ov_start,\n                         ov_end\n                     )/3600.0\n                ELSE 0\n            END\n        ) AS numerator_ml,\n\n        DATE_DIFF(\n            'second',\n            GREATEST(\n                hour_start,\n                MIN(interval_start)\n            ),\n            LEAST(\n                hour_end,\n                MAX(interval_end)\n            )\n        )/3600.0 AS proportion_covered\n\n    FROM ovl\n    GROUP BY\n        stay_id,\n        hour_start,\n        hour_end\n)\n\nSELECT\n    stay_id,\n    hour_start,\n    hour_end,\n    proportion_covered,\n\n    CASE\n        WHEN proportion_covered > 0.5\n        THEN numerator_ml / proportion_covered\n        ELSE NULL\n    END AS hourly_adjusted_ml_hr\n\nFROM calc\nORDER BY\n    stay_id,\n    hour_end\n").df()
    hourly = hourly.merge(uobase[['stay_id', 'ibw_kg']], on='stay_id', how='left', validate='m:1')
    hourly = hourly.merge(ov[['stay_id', 'uo_unreliable_by_landmark']], on='stay_id', how='left', validate='m:1')
    hourly['hourly_adjusted_ml_kg_hr'] = hourly['hourly_adjusted_ml_hr'] / pd.to_numeric(hourly['ibw_kg'], errors='coerce')
    hourly.loc[hourly['uo_unreliable_by_landmark'].fillna(0).eq(1), 'hourly_adjusted_ml_kg_hr'] = np.nan

    def stage_cons(g):
        g = g.sort_values('hour_end').reset_index(drop=True)
        vals = pd.to_numeric(g['hourly_adjusted_ml_kg_hr'], errors='coerce').to_numpy(float)
        best = 0
        for i in range(len(g)):
            for W in (6, 12, 24):
                if i - W + 1 < 0:
                    continue
                x = vals[i - W + 1:i + 1]
                if not np.isfinite(x).all():
                    continue
                t = pd.to_datetime(g['hour_end'].iloc[i - W + 1:i + 1])
                if len(t) > 1:
                    diffs = np.diff(t.values.astype('datetime64[s]').astype('int64'))
                    if not np.all(diffs == 3600):
                        continue
                mx = float(np.max(x))
                if W == 24 and mx < 0.3:
                    best = max(best, 3)
                if W == 12:
                    if mx == 0:
                        best = max(best, 3)
                    if mx < 0.5:
                        best = max(best, 2)
                if W == 6 and mx < 0.5:
                    best = max(best, 1)
        return pd.Series({'urine_output_stage_consecutive': float(best) if best >= 1 else np.nan, 'valid_hour_count': int(np.isfinite(vals).sum())})
    ov_uo = hourly.groupby('stay_id', sort=False).apply(stage_cons).reset_index()
    ov_uo = ov[['stay_id']].merge(ov_uo, on='stay_id', how='left', validate='1:1')
    ov_uo['valid_hour_count'] = ov_uo['valid_hour_count'].fillna(0).astype(int)
    display_table(ov_uo.describe())
    wcsv(ov_uo, 'post_transplant_consecutive_urine_output.csv')
    base = strict_target[['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'kdigo_creatinine_stage_strict_outpatient', 'active_rrt_any_by_landmark', 'combined_status_strict_outpatient', 'combined_stage_strict_outpatient']].copy()
    base = base.merge(oliguria_stages[['stay_id', 'urine_output_stage_consecutive']], on='stay_id', how='left', validate='1:1')
    base['target_source_consecutive_urine_output'] = 'strict-outpatient_eligible_target'
    ext = ov.merge(ov_uo, on='stay_id', how='left', validate='1:1')
    ext = ext.rename(columns={'kdigo_creatinine_stage_post_transplant': 'kdigo_creatinine_stage_strict_outpatient'})
    ext['target_source_consecutive_urine_output'] = 'post_transplant_temporal_override'
    keep = ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'kdigo_creatinine_stage_strict_outpatient', 'active_rrt_any_by_landmark', 'urine_output_stage_consecutive', 'target_source_consecutive_urine_output']
    consecutive_target = pd.concat([base[keep], ext[keep]], ignore_index=True)
    if consecutive_target.stay_id.duplicated().any():
        raise AssertionError('expanded target duplicate stay_id.')
    cr = pd.to_numeric(consecutive_target.kdigo_creatinine_stage_strict_outpatient, errors='coerce')
    uo = pd.to_numeric(consecutive_target.urine_output_stage_consecutive, errors='coerce')
    rrt = consecutive_target.active_rrt_any_by_landmark.fillna(0).eq(1)
    consecutive_target['aki_positive_consecutive_urine_output'] = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
    consecutive_target['kdigo_stage_consecutive_urine_output'] = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
    consecutive_target['kdigo_stage_consecutive_urine_output'] = consecutive_target.kdigo_stage_consecutive_urine_output.where(consecutive_target.kdigo_stage_consecutive_urine_output >= 1, np.nan)
    consecutive_target.loc[rrt, 'kdigo_stage_consecutive_urine_output'] = 3.0
    consecutive_target['aki_status_consecutive_urine_output'] = np.where(consecutive_target.aki_positive_consecutive_urine_output, 'positive', 'not_positive')
    selected = first_positive(consecutive_target)
    if len(consecutive_target) != EXPECTED_EXPANDED_TARGET_N:
        raise AssertionError(f'Expanded target N={len(consecutive_target)}, expected {EXPECTED_EXPANDED_TARGET_N}.')
    if len(selected) != EXPECTED_CONSECUTIVE_N:
        raise AssertionError(f'Consecutive-urine-output cohort N={len(selected)}, expected {EXPECTED_CONSECUTIVE_N}.')
    selected = selected.sort_values(['subject_id', 'stay_id']).reset_index(drop=True)
    if selected.subject_id.duplicated().any() or selected.stay_id.duplicated().any():
        raise AssertionError('consecutive-urine-output selected index is not unique.')
    print('expanded target stays:', len(consecutive_target))
    print('consecutive-urine-output primary subjects:', len(selected))
    print('Selected from transplant override:', int(selected.target_source_consecutive_urine_output.eq('post_transplant_temporal_override').sum()))
    cmp_primary = pd.DataFrame([membership_compare(strict_idx, selected, 'consecutive_urine_output_plus_post_transplant')])
    base_mean = strict_target.loc[strict_target.combined_status_strict_outpatient.eq('positive'), ['subject_id', 'stay_id', 'intime']].copy()
    base_mean['source'] = 'strict_outpatient_positive'
    ovmean = post_transplant_reconstruction.copy()
    cr_m = pd.to_numeric(ovmean.kdigo_creatinine_stage_post_transplant, errors='coerce')
    uo_m = pd.to_numeric(ovmean.urine_output_stage_post_transplant, errors='coerce')
    rrt_m = ovmean.active_rrt_any_by_landmark.fillna(0).eq(1)
    ovmean['positive_mean'] = cr_m.fillna(0).ge(1) | uo_m.fillna(0).ge(1) | rrt_m
    ovmean = ovmean.loc[ovmean.post_transplant_temporal_allow_acute_rrt.fillna(False) & ovmean.positive_mean, ['subject_id', 'stay_id', 'intime']].copy()
    ovmean['source'] = 'post_transplant_override'
    sel_mean = pd.concat([base_mean, ovmean], ignore_index=True).sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').drop_duplicates('subject_id', keep='first')
    cons21 = oliguria_summary.loc[oliguria_summary.variant.eq('calendar_hour_consecutive_urine_output')].iloc[0]
    cmp_all = pd.concat([cmp_primary, pd.DataFrame([membership_compare(strict_idx, sel_mean, 'mean_urine_output_plus_post_transplant')]), pd.DataFrame([{'variant': 'consecutive_urine_output_without_post_transplant', 'subjects': int(cons21.subjects), 'subject_added_vs_strict_outpatient': int(cons21.subject_added_vs_strict_outpatient), 'subject_dropped_vs_strict_outpatient': int(cons21.subject_dropped_vs_strict_outpatient), 'common_subjects': int(cons21.common_subjects), 'same_subject_replaced_stay': int(cons21.same_subject_replaced_stay), 'same_subject_same_stay': int(cons21.same_subject_same_stay), 'stay_symmetric_difference': int(cons21.stay_symmetric_difference)}])], ignore_index=True)
    component_patterns = pd.DataFrame({'cr_positive': cr.fillna(0).ge(1), 'uo_positive': uo.fillna(0).ge(1), 'rrt_positive': rrt})
    component_patterns['pattern'] = 'Cr' + component_patterns.cr_positive.astype(int).astype(str) + '_UO' + component_patterns.uo_positive.astype(int).astype(str) + '_RRT' + component_patterns.rrt_positive.astype(int).astype(str)
    component_summary = component_patterns.loc[consecutive_target.aki_positive_consecutive_urine_output.to_numpy()].groupby('pattern').size().reset_index(name='positive_target_stays')
    sel_detail = selected[['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time', 'kdigo_creatinine_stage_strict_outpatient', 'urine_output_stage_consecutive', 'active_rrt_any_by_landmark', 'kdigo_stage_consecutive_urine_output', 'target_source_consecutive_urine_output']].copy()
    display_table(cmp_all)
    display_table(component_summary)
    display_table(sel_detail.target_source_consecutive_urine_output.value_counts())
    wcsv(cmp_all, 'consecutive_urine_output_membership_comparison.csv')
    wcsv(component_summary, 'consecutive_urine_output_component_patterns.csv')
    wcsv(consecutive_target, 'consecutive_urine_output_target_table.csv')
    wcsv(sel_detail, 'consecutive_urine_output_selected_index.csv')
    P_TARGET_OUT = OUT / 'consecutive_urine_output_target.parquet'
    P_INDEX_OUT = OUT / 'consecutive_urine_output_cohort_preoutcome.parquet'

    def normalize_preoutcome_types(df, label):
        df = df.copy()
        for c in ['subject_id', 'hadm_id', 'stay_id']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='raise')
                if df[c].isna().any():
                    raise AssertionError(f'{label}: {c} contains null values after numeric conversion.')
                df[c] = df[c].astype('int64')
        for c in ['admittime', 'intime', 'landmark_time']:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='raise')
                if df[c].isna().any():
                    raise AssertionError(f'{label}: {c} contains null values after datetime conversion.')
        for c in ['kdigo_creatinine_stage_strict_outpatient', 'urine_output_stage_consecutive', 'active_rrt_any_by_landmark', 'kdigo_stage_consecutive_urine_output']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'aki_positive_consecutive_urine_output' in df.columns:
            df['aki_positive_consecutive_urine_output'] = df['aki_positive_consecutive_urine_output'].fillna(False).astype(bool)
        for c in ['target_source_consecutive_urine_output', 'aki_status_consecutive_urine_output']:
            if c in df.columns:
                df[c] = df[c].astype('string')
        return df
    consecutive_target = normalize_preoutcome_types(consecutive_target, 'consecutive_target')
    selected = normalize_preoutcome_types(selected, 'selected')
    dtype_audit = pd.DataFrame({'column': consecutive_target.columns, 'dtype': [str(consecutive_target[c].dtype) for c in consecutive_target.columns]})
    object_cols = [c for c in consecutive_target.columns if consecutive_target[c].dtype == 'object']
    if object_cols:
        raise AssertionError(f'consecutive_target still contains unsafe object-dtype columns before Parquet persistence: {object_cols}')
    display_table(dtype_audit)
    wcsv(dtype_audit, 'consecutive_urine_output_dtype_audit.csv')
    consecutive_target.to_parquet(P_TARGET_OUT, index=False)
    selected.to_parquet(P_INDEX_OUT, index=False)
    target_disk = pd.read_parquet(P_TARGET_OUT)
    index_disk = pd.read_parquet(P_INDEX_OUT)
    if len(target_disk) != len(consecutive_target):
        raise AssertionError('Target disk roundtrip N mismatch.')
    if len(index_disk) != len(selected):
        raise AssertionError('Index disk roundtrip N mismatch.')
    expected_membership = selected[['subject_id','stay_id']].sort_values(['subject_id','stay_id']).reset_index(drop=True)
    disk_membership = index_disk[['subject_id','stay_id']].sort_values(['subject_id','stay_id']).reset_index(drop=True)
    if not expected_membership.equals(disk_membership):
        raise AssertionError('Index membership changed during Parquet roundtrip.')
    summary = {
        'status': 'PASS',
        'outcome_blind': True,
        'strict_outpatient_cohort_n': EXPECTED_STRICT_N,
        'expanded_target_n': int(len(target_disk)),
        'consecutive_urine_output_cohort_n': int(len(index_disk)),
        'selected_post_transplant_override_n': int(index_disk['target_source_consecutive_urine_output'].eq('post_transplant_temporal_override').sum()),
        'primary_changes': [
            'persistent/consecutive-hour urine-output interpretation on fixed6 + ideal-body-weight reconstruction',
            'later kidney transplant may override stale prior kidney-failure/dialysis evidence when no later/current chronic-state or graft-failure evidence remains',
        ],
        'unchanged': [
            '+24 h ICU landmark',
            'prior completed-admission direct heart failure',
            'strict-outpatient creatinine KDIGO component',
            'landmark-safe urine-output stent/GU-irrigation reliability mask',
            'active acute RRT contributes KDIGO Stage 3 after chronic-state exclusion',
            'first qualifying positive ICU episode per subject',
        ],
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    wjson(summary, 'consecutive_urine_output_manifest.json')
    print('CONSECUTIVE URINE-OUTPUT PHENOTYPE: PASS')
    print('Cohort N:', len(index_disk))


if __name__ == "__main__":
    main()
