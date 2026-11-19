#!/usr/bin/env python3
"""11 Persistent Oliguria.

Apply persistent oliguria interpretation.

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
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    STRICT_DIR = PHENOTYPE_DIR / 'historical_creatinine' / 'strict_outpatient'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    OUT = PHENOTYPE_DIR / 'persistent_oliguria'
    QC = PROJECT_ROOT / 'quality_control' / 'phenotype' / 'persistent_oliguria'
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
    P_UOBASE = req(URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet')
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    target = pd.read_parquet(P_TARGET, columns=['subject_id', 'stay_id', 'intime', 'landmark_time', 'kdigo_creatinine_stage_strict_outpatient', 'urine_output_stage_reliability_adjusted', 'active_rrt_any_by_landmark', 'uo_unreliable_by_landmark'])
    idx = pd.read_parquet(P_INDEX, columns=['subject_id', 'stay_id', 'intime'])
    base = pd.read_parquet(P_UOBASE, columns=['stay_id', 'ibw_kg', 'height_cm', 'gender'])
    hard_guard(idx, target)
    if base.stay_id.duplicated().any():
        raise AssertionError('Urine-output base is not unique by stay_id.')
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    COMPARTMENT_CASE = "\nCASE\n WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n ELSE NULL\nEND\n"
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('target_df', target[['stay_id', 'intime', 'landmark_time']])
    con.execute('CREATE TEMP TABLE target AS SELECT * FROM target_df')
    con.execute(f"\nCREATE VIEW oe AS\nSELECT stay_id,itemid,charttime,storetime,value\nFROM read_csv_auto('{qpath(OE)}',header=true,sample_size=200000)\n")
    con.execute(f'\nCREATE TEMP TABLE uo_events AS\nSELECT t.stay_id,{COMPARTMENT_CASE} compartment,oe.charttime,\n       MAX(oe.storetime) storetime,SUM(oe.value) volume_ml\nFROM target t JOIN oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime>=t.intime AND oe.charttime<=t.landmark_time\n  AND oe.storetime IS NOT NULL AND oe.storetime<=t.landmark_time\nGROUP BY t.stay_id,compartment,oe.charttime\nHAVING compartment IS NOT NULL\n')
    con.execute('\nCREATE TEMP TABLE raw_intervals AS\nSELECT stay_id,compartment,\n LAG(charttime) OVER(PARTITION BY stay_id,compartment ORDER BY charttime) interval_start,\n charttime interval_end,volume_ml\nFROM uo_events\n')
    intervals = con.execute("\nSELECT r.stay_id,r.compartment,r.interval_start,r.interval_end,r.volume_ml,\n DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0 duration_hours,\n r.volume_ml/(DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0) rate_ml_hr\nFROM raw_intervals r JOIN target t USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end>r.interval_start\n  AND r.interval_start>=t.intime AND r.interval_end<=t.landmark_time\n  AND DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0<=6.0+1e-12\nORDER BY r.stay_id,r.interval_start\n").df()
    print('Fixed6 interval rows:', len(intervals))
    print('Stays with fixed6 intervals:', intervals.stay_id.nunique())
    hour_rows = []
    for r in target[['stay_id', 'intime', 'landmark_time']].itertuples(index=False):
        intime = pd.Timestamp(r.intime)
        lm = pd.Timestamp(r.landmark_time)
        hs = intime.floor('h')
        cur = hs
        while cur + pd.Timedelta(hours=1) <= lm:
            hour_rows.append((int(r.stay_id), cur, cur + pd.Timedelta(hours=1)))
            cur += pd.Timedelta(hours=1)
    hours = pd.DataFrame(hour_rows, columns=['stay_id', 'hour_start', 'hour_end'])
    conh = duckdb.connect(database=':memory:')
    conh.register('hours_df', hours)
    conh.register('intervals_df', intervals)
    conh.execute('CREATE TEMP TABLE hours AS SELECT * FROM hours_df')
    conh.execute('CREATE TEMP TABLE intervals AS SELECT * FROM intervals_df')
    hourly = conh.execute("\nWITH ov AS (\n SELECT h.stay_id,h.hour_start,h.hour_end,\n        i.interval_start,i.interval_end,i.rate_ml_hr,\n        GREATEST(i.interval_start,h.hour_start) ov_start,\n        LEAST(i.interval_end,h.hour_end) ov_end\n FROM hours h\n LEFT JOIN intervals i\n   ON i.stay_id=h.stay_id\n  AND i.interval_end>h.hour_start\n  AND i.interval_start<h.hour_end\n),\ncalc AS (\n SELECT stay_id,hour_start,hour_end,\n   SUM(CASE WHEN ov_end>ov_start THEN\n       rate_ml_hr*DATE_DIFF('second',ov_start,ov_end)/3600.0 ELSE 0 END) numerator_ml,\n   DATE_DIFF(\n     'second',\n     GREATEST(hour_start,MIN(interval_start)),\n     LEAST(hour_end,MAX(interval_end))\n   )/3600.0 AS proportion_covered\n FROM ov\n GROUP BY stay_id,hour_start,hour_end\n)\nSELECT stay_id,hour_start,hour_end,proportion_covered,\n CASE WHEN proportion_covered>0.5\n      THEN numerator_ml/proportion_covered ELSE NULL END AS hourly_adjusted_ml_hr\nFROM calc\nORDER BY stay_id,hour_end\n").df()
    hourly = hourly.merge(base[['stay_id', 'ibw_kg']], on='stay_id', how='left', validate='m:1').merge(target[['stay_id', 'uo_unreliable_by_landmark']], on='stay_id', how='left', validate='m:1')
    hourly['hourly_adjusted_ml_kg_hr'] = hourly.hourly_adjusted_ml_hr / pd.to_numeric(hourly.ibw_kg, errors='coerce')
    hourly.loc[hourly.uo_unreliable_by_landmark.fillna(0).eq(1), 'hourly_adjusted_ml_kg_hr'] = np.nan
    coverage = pd.DataFrame([{'calendar_hour_rows': len(hourly), 'valid_gt_half_hour': int(hourly.hourly_adjusted_ml_kg_hr.notna().sum()), 'valid_pct': float(100 * hourly.hourly_adjusted_ml_kg_hr.notna().mean()), 'stays_with_any_valid_hour': int(hourly.loc[hourly.hourly_adjusted_ml_kg_hr.notna(), 'stay_id'].nunique())}])
    display_table(coverage)
    wcsv(coverage, OUT, 'urine_output_hourly_coverage.csv')
    def stages_for_stay(g):
        g = g.sort_values('hour_end').reset_index(drop=True)
        vals = pd.to_numeric(g.hourly_adjusted_ml_kg_hr, errors='coerce').to_numpy(float)
        ends = pd.to_datetime(g.hour_end).to_numpy()
        best_cons = 0
        best_mean = 0
        for i in range(len(g)):
            for W in (6, 12, 24):
                if i - W + 1 < 0:
                    continue
                x = vals[i - W + 1:i + 1]
                if not np.isfinite(x).all():
                    continue
                t = pd.to_datetime(g.hour_end.iloc[i - W + 1:i + 1])
                if len(t) > 1:
                    diffs = np.diff(t.values.astype('datetime64[s]').astype('int64'))
                    if not np.all(diffs == 3600):
                        continue
                mx = float(np.max(x))
                mean = float(np.mean(x))
                if W == 24:
                    if mx < 0.3:
                        best_cons = max(best_cons, 3)
                    if mean < 0.3:
                        best_mean = max(best_mean, 3)
                if W == 12:
                    if mx == 0:
                        best_cons = max(best_cons, 3)
                    if mean == 0:
                        best_mean = max(best_mean, 3)
                    if mx < 0.5:
                        best_cons = max(best_cons, 2)
                    if mean < 0.5:
                        best_mean = max(best_mean, 2)
                if W == 6:
                    if mx < 0.5:
                        best_cons = max(best_cons, 1)
                    if mean < 0.5:
                        best_mean = max(best_mean, 1)
        return pd.Series({'urine_output_stage_consecutive': float(best_cons) if best_cons >= 1 else np.nan, 'urine_output_stage_mean': float(best_mean) if best_mean >= 1 else np.nan, 'valid_hours': int(np.isfinite(vals).sum())})
    stages = hourly.groupby('stay_id', sort=False).apply(stages_for_stay).reset_index()
    stages = target[['stay_id']].merge(stages, on='stay_id', how='left', validate='1:1')
    stages['valid_hours'] = stages.valid_hours.fillna(0).astype(int)
    display_table(stages[['urine_output_stage_consecutive', 'urine_output_stage_mean', 'valid_hours']].describe())
    wcsv(stages, OUT, 'persistent_oliguria_stages.csv')
    def cohort_variant(stage_series, label):
        q = target[['subject_id', 'stay_id', 'intime', 'kdigo_creatinine_stage_strict_outpatient', 'active_rrt_any_by_landmark']].copy()
        q['uo_stage_variant'] = pd.to_numeric(stage_series, errors='coerce').to_numpy()
        cr = pd.to_numeric(q.kdigo_creatinine_stage_strict_outpatient, errors='coerce')
        uo = pd.to_numeric(q.uo_stage_variant, errors='coerce')
        rrt = q.active_rrt_any_by_landmark.fillna(0).eq(1)
        q['combined_positive'] = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        sel = first_positive(q)
        cmp = membership_compare(idx, sel, label)
        cmp['uo_positive_target_stays'] = int(uo.fillna(0).ge(1).sum())
        cmp['qualifying_target_stays'] = int(q.combined_positive.sum())
        return (sel, cmp)
    sbase = target[['stay_id', 'urine_output_stage_reliability_adjusted']].merge(stages, on='stay_id', how='left', validate='1:1')
    variants = [('urine_output_stage_reliability_adjusted', 'reference_urine_output'), ('urine_output_stage_consecutive', 'calendar_hour_consecutive_urine_output'), ('urine_output_stage_mean', 'calendar_hour_mean_urine_output')]
    rows = []
    details = []
    for col, label in variants:
        sel, cmp = cohort_variant(sbase[col], label)
        rows.append(cmp)
        d = idx[['subject_id', 'stay_id']].rename(columns={'stay_id': 'strict_outpatient_stay_id'}).merge(sel[['subject_id', 'stay_id']].rename(columns={'stay_id': 'alternative_stay_id'}), on='subject_id', how='outer', indicator=True)
        d['variant'] = label
        details.append(d)
    summary = pd.DataFrame(rows)
    reference_row = summary.loc[summary.variant.eq('reference_urine_output')].iloc[0]
    if int(reference_row.subject_added_vs_strict_outpatient) != 0 or int(reference_row.subject_dropped_vs_strict_outpatient) != 0 or int(reference_row.same_subject_replaced_stay) != 0:
        raise AssertionError('Strict-outpatient cohort failed self-reproduction.')
    display_table(summary)
    wcsv(summary, OUT, 'persistent_oliguria_cohort_effect.csv')
    wcsv(pd.concat(details, ignore_index=True), OUT, 'persistent_oliguria_membership_detail.csv')
    rc = sbase.copy()
    rc['reference_pos'] = pd.to_numeric(rc.urine_output_stage_reliability_adjusted, errors='coerce').fillna(0).ge(1)
    rc['consecutive_pos'] = pd.to_numeric(rc.urine_output_stage_consecutive, errors='coerce').fillna(0).ge(1)
    reclass = pd.DataFrame([{'reference_urine_output_positive_stays': int(rc.reference_pos.sum()), 'consecutive_urine_output_positive_stays': int(rc.consecutive_pos.sum()), 'reference_positive_consecutive_not': int((rc.reference_pos & ~rc.consecutive_pos).sum()), 'consecutive_positive_reference_not': int((rc.consecutive_pos & ~rc.reference_pos).sum())}])
    display_table(reclass)
    wcsv(reclass, OUT, 'urine_output_reclassification_summary.csv')
    manifest = {
        'status': 'PASS',
        'outcome_blind': True,
        'strict_outpatient_cohort_n': len(idx),
        'method': 'calendar-hour urine-output reconstruction with >0.5 h coverage and persistent/consecutive-hour criteria',
        'reference': 'Hasidim et al. Scientific Reports 2025;15:20009; DOI 10.1038/s41598-025-95535-4',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    wjson(manifest, OUT, 'persistent_oliguria_manifest.json')
    print('PERSISTENT OLIGURIA EVALUATION: PASS')


if __name__ == "__main__":
    main()
