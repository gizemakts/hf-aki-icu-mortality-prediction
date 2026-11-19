#!/usr/bin/env python3
"""Evaluate urine-output measurement-interval cap sensitivity variants."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT
from helpers._runtime import display_table

EXPECTED_PRIMARY_N = 3457
EXPECTED_PRIMARY_TARGET_N = 10083
EXPECTED_SOURCE_P95_N = 3365
EXPECTED_NO_INTERVAL_CAP_N = 3698


def main() -> None:
    """Reconstruct interval-cap variants from raw urine-output events."""
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))
    root = _REPOSITORY_ROOT
    raw_root = Path(os.environ["MIMIC_ROOT"]).resolve()
    cohort_dir = root / "outputs" / "01_cohort"
    phenotype_dir = root / "outputs" / "02_phenotype"
    population_dir = root / "outputs" / "06_sensitivity" / "populations"
    qc_dir = root / "quality_control" / "sensitivity" / "urine_output_interval_caps"
    population_dir.mkdir(parents=True, exist_ok=True); qc_dir.mkdir(parents=True, exist_ok=True)

    def req(path):
        path=Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    def qpath(path): return str(Path(path).resolve()).replace("'", "''")
    prohibited=("death","mortality","expire","outcome","deathtime","survival","auc","brier","calibration")
    def assert_preoutcome(df,label):
        bad=[c for c in df.columns if any(t in c.lower() for t in prohibited)]
        if bad: raise AssertionError(f"{label}: prohibited fields {bad}")
    def first_positive(df,positive_col):
        q=df.loc[df[positive_col].fillna(False).astype(bool)].copy(); q=q.sort_values(["subject_id","intime","stay_id"],kind="mergesort")
        return q.drop_duplicates("subject_id",keep="first").copy()

    P_TARGET=req(phenotype_dir/"aki_target_preoutcome.parquet")
    P_INDEX=req(cohort_dir/"primary_cohort_preoutcome.parquet")
    P_LM=req(cohort_dir/"landmark_population.parquet")
    OE=req(raw_root/"icu"/"outputevents.csv.gz")
    target=pd.read_parquet(P_TARGET,columns=["subject_id","hadm_id","stay_id","admittime","intime","landmark_time","creatinine_stage","creatinine_evaluable","ibw_kg","uo_unreliable_by_landmark","active_rrt_any_by_landmark","urine_output_stage","aki_positive","kdigo_stage"])
    index=pd.read_parquet(P_INDEX); lm=pd.read_parquet(P_LM,columns=["subject_id","hadm_id","stay_id","intime","landmark_time"])
    for df,label in [(target,"primary target"),(index,"primary cohort"),(lm,"landmark")]: assert_preoutcome(df,label)
    if len(target)!=EXPECTED_PRIMARY_TARGET_N or len(index)!=EXPECTED_PRIMARY_N: raise AssertionError(f"Primary source sizes changed: target={len(target)}, cohort={len(index)}")
    if target.stay_id.duplicated().any() or lm.stay_id.duplicated().any(): raise AssertionError("stay_id uniqueness failure")
    for col in ["admittime","intime","landmark_time"]:
        if col in target.columns: target[col]=pd.to_datetime(target[col],errors="raise")
    for col in ["intime","landmark_time"]: lm[col]=pd.to_datetime(lm[col],errors="raise")

    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    COMPARTMENT_CASE = "\nCASE\n  WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n  WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n  WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n  WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n  WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n  WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n  ELSE NULL\nEND\n"
    SOURCE_CASE = "\nCASE\n  WHEN oe.itemid=226559 THEN 'FOLEY'\n  WHEN oe.itemid=226560 THEN 'VOID'\n  WHEN oe.itemid=226561 THEN 'CONDOM_CATH'\n  WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n  WHEN oe.itemid=226563 THEN 'SUPRAPUBIC'\n  WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n  WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n  WHEN oe.itemid=226567 THEN 'STRAIGHT_CATH'\n  WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n  WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n  ELSE 'OTHER'\nEND\n"
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('lm_df', lm[['stay_id', 'intime', 'landmark_time']])
    con.execute('CREATE TEMP TABLE landmark AS SELECT * FROM lm_df')
    con.execute(f"\nCREATE VIEW oe AS\nSELECT stay_id,itemid,charttime,storetime,value\nFROM read_csv_auto('{qpath(OE)}', header=true, sample_size=200000)\n")
    con.execute(f"\nCREATE TEMP TABLE uo_events_all AS\nSELECT\n    l.stay_id,\n    {COMPARTMENT_CASE} AS compartment,\n    oe.charttime,\n    MAX(oe.storetime) AS storetime,\n    SUM(oe.value) AS volume_ml,\n    COUNT(DISTINCT oe.itemid) AS n_source_itemids,\n    CASE\n      WHEN COUNT(DISTINCT oe.itemid)=1 THEN MIN({SOURCE_CASE})\n      ELSE 'MIXED_' || ({COMPARTMENT_CASE})\n    END AS source_class\nFROM landmark l\nJOIN oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL\n  AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime>=l.intime\n  AND oe.charttime<=l.landmark_time\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime<=l.landmark_time\nGROUP BY l.stay_id,compartment,oe.charttime\nHAVING compartment IS NOT NULL\n")
    con.execute('\nCREATE TEMP TABLE uo_interval_raw_all AS\nSELECT\n    stay_id,compartment,source_class,\n    charttime AS interval_end,\n    volume_ml,\n    LAG(charttime) OVER(\n      PARTITION BY stay_id,compartment ORDER BY charttime\n    ) AS interval_start\nFROM uo_events_all\n')
    con.execute("\nCREATE TEMP TABLE uo_intervals_all AS\nSELECT\n    r.stay_id,r.compartment,r.source_class,\n    r.interval_start,r.interval_end,r.volume_ml,\n    DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0 AS duration_hours\nFROM uo_interval_raw_all r\nJOIN landmark l USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end>r.interval_start\n  AND r.interval_start>=l.intime\n  AND r.interval_end<=l.landmark_time\n")
    source_dist = con.execute('\nSELECT\n  source_class,\n  COUNT(*) AS intervals,\n  COUNT(DISTINCT stay_id) AS stays,\n  MEDIAN(duration_hours) AS median_h,\n  QUANTILE_CONT(duration_hours,0.90) AS p90_h,\n  QUANTILE_CONT(duration_hours,0.95) AS p95_h,\n  QUANTILE_CONT(duration_hours,0.975) AS p975_h,\n  QUANTILE_CONT(duration_hours,0.99) AS p99_h,\n  100.0*AVG(CASE WHEN duration_hours>6 THEN 1 ELSE 0 END) AS pct_gt6h\nFROM uo_intervals_all\nGROUP BY source_class\nORDER BY intervals DESC\n').df()
    if source_dist.empty or source_dist.p95_h.isna().any():
        raise AssertionError('Source-duration P95 derivation failed.')
    source_caps = dict(zip(source_dist.source_class.astype(str), source_dist.p95_h.astype(float)))
    display_table(source_dist)
    print('SOURCE P95 CAPS DERIVED: PASS')
    target_ids = target[['stay_id', 'intime', 'landmark_time']].copy()
    con.register('target_ids_df', target_ids)
    con.execute('CREATE TEMP TABLE target_ids AS SELECT * FROM target_ids_df')
    intervals = con.execute('\nSELECT\n    i.stay_id,i.compartment,i.source_class,i.interval_start,i.interval_end,i.volume_ml,\n    i.duration_hours,\n    i.volume_ml/i.duration_hours AS rate_ml_hr\nFROM uo_intervals_all i\nJOIN target_ids t USING(stay_id)\nORDER BY i.stay_id,i.interval_start,i.interval_end\n').df()
    hour_rows = []
    for r in target[['stay_id', 'intime', 'landmark_time']].itertuples(index=False):
        cur = pd.Timestamp(r.intime).floor('h')
        lm_time = pd.Timestamp(r.landmark_time)
        while cur + pd.Timedelta(hours=1) <= lm_time:
            hour_rows.append((int(r.stay_id), cur, cur + pd.Timedelta(hours=1)))
            cur += pd.Timedelta(hours=1)
    hours = pd.DataFrame(hour_rows, columns=['stay_id', 'hour_start', 'hour_end'])
    if hours.empty:
        raise AssertionError('Calendar-hour grid is empty.')
    print('Target raw intervals:', len(intervals))
    print('Calendar-hour rows  :', len(hours))
    def interval_mask(df, variant):
        dur = pd.to_numeric(df.duration_hours, errors='coerce').to_numpy(float)
        keep = np.isfinite(dur) & (dur > 0)
        if variant == 'fixed_6h':
            keep &= dur <= 6.0 + 1e-12
        elif variant == 'source_p95_cap':
            caps = df.source_class.astype(str).map(source_caps).to_numpy(float)
            keep &= np.isfinite(caps) & (dur <= caps + 1e-12)
        elif variant == 'no_interval_cap':
            pass
        else:
            raise ValueError(variant)
        return keep

    def build_hourly(variant):
        z = intervals.loc[interval_mask(intervals, variant), ['stay_id', 'interval_start', 'interval_end', 'rate_ml_hr']].copy()
        cx = duckdb.connect(database=':memory:')
        cx.execute('PRAGMA threads=4')
        cx.register('hours_df', hours)
        cx.register('intervals_df', z)
        cx.execute('CREATE TEMP TABLE hours AS SELECT * FROM hours_df')
        cx.execute('CREATE TEMP TABLE intervals AS SELECT * FROM intervals_df')
        h = cx.execute("\n    WITH ov AS (\n      SELECT\n        h.stay_id,h.hour_start,h.hour_end,\n        i.rate_ml_hr,\n        GREATEST(i.interval_start,h.hour_start) AS ov_start,\n        LEAST(i.interval_end,h.hour_end) AS ov_end\n      FROM hours h\n      JOIN intervals i\n        ON i.stay_id=h.stay_id\n       AND i.interval_end>h.hour_start\n       AND i.interval_start<h.hour_end\n      WHERE LEAST(i.interval_end,h.hour_end)>GREATEST(i.interval_start,h.hour_start)\n    ),\n    agg AS (\n      SELECT\n        stay_id,hour_start,hour_end,\n        SUM(rate_ml_hr*DATE_DIFF('second',ov_start,ov_end)/3600.0) AS numerator_ml,\n        DATE_DIFF('second',MIN(ov_start),MAX(ov_end))/3600.0 AS coverage_h\n      FROM ov\n      GROUP BY stay_id,hour_start,hour_end\n    )\n    SELECT\n      h.stay_id,h.hour_start,h.hour_end,\n      a.numerator_ml,a.coverage_h,\n      CASE WHEN a.coverage_h>0.5 THEN a.numerator_ml/a.coverage_h END AS hourly_ml_hr\n    FROM hours h\n    LEFT JOIN agg a USING(stay_id,hour_start,hour_end)\n    ORDER BY h.stay_id,h.hour_end\n    ").df()
        h = h.merge(target[['stay_id', 'intime', 'ibw_kg', 'uo_unreliable_by_landmark']], on='stay_id', how='left', validate='m:1')
        h['uo_ml_kg_hr'] = pd.to_numeric(h.hourly_ml_hr, errors='coerce') / pd.to_numeric(h.ibw_kg, errors='coerce')
        bad = pd.to_numeric(h.uo_unreliable_by_landmark, errors='coerce').fillna(0).eq(1)
        h.loc[bad, 'uo_ml_kg_hr'] = np.nan
        h['variant'] = variant
        return h
    hourly_by_variant = {v: build_hourly(v) for v in ['fixed_6h', 'source_p95_cap', 'no_interval_cap']}
    for v, h in hourly_by_variant.items():
        print(v, 'valid UO hours =', int(h.uo_ml_kg_hr.notna().sum()))
    def max_consecutive_valid(vals):
        best = cur = 0
        for ok in np.isfinite(vals):
            if ok:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def stage_uo(g):
        g = g.sort_values('hour_end').reset_index(drop=True)
        vals = pd.to_numeric(g.uo_ml_kg_hr, errors='coerce').to_numpy(float)
        intime = pd.Timestamp(g.intime.iloc[0])
        best = 0
        path_anuria = False
        path_olig24 = False
        for W, kind in [(6, 's1'), (12, 's2'), (12, 'anuria'), (24, 's3')]:
            for i in range(len(g)):
                j = i - W + 1
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
                actual_start = max(pd.Timestamp(g.hour_start.iloc[j]), intime)
                actual_end = pd.Timestamp(g.hour_end.iloc[i])
                elapsed_h = (actual_end - actual_start).total_seconds() / 3600.0
                if elapsed_h + 1e-12 < W:
                    continue
                stat = float(np.mean(x))
                if kind == 's1' and stat < 0.5:
                    best = max(best, 1)
                elif kind == 's2' and stat < 0.5:
                    best = max(best, 2)
                elif kind == 'anuria' and stat == 0:
                    best = max(best, 3)
                    path_anuria = True
                elif kind == 's3' and stat < 0.3:
                    best = max(best, 3)
                    path_olig24 = True
        return (float(best) if best >= 1 else np.nan, path_anuria, path_olig24)
    stage_tables = {}
    for variant, h in hourly_by_variant.items():
        rows = []
        for stay_id, g in h.groupby('stay_id', sort=False):
            vals = pd.to_numeric(g.uo_ml_kg_hr, errors='coerce').to_numpy(float)
            st, an, ol = stage_uo(g)
            rows.append({'stay_id': int(stay_id), f'uo_stage_{variant}': st, f'uo_valid_hours_{variant}': int(np.isfinite(vals).sum()), f'uo_max_consecutive_valid_hours_{variant}': max_consecutive_valid(vals), f'uo_stage3_anuria12_{variant}': int(an), f'uo_stage3_oliguria24_{variant}': int(ol)})
        stage_tables[variant] = pd.DataFrame(rows)
    fixed = target[['stay_id', 'urine_output_stage']].merge(stage_tables['fixed_6h'][['stay_id', 'uo_stage_fixed_6h']], on='stay_id', validate='1:1')
    a = pd.to_numeric(fixed.urine_output_stage, errors='coerce').fillna(-999).to_numpy(float)
    b = pd.to_numeric(fixed.uo_stage_fixed_6h, errors='coerce').fillna(-999).to_numpy(float)
    uo_stage_mismatch = int((~np.isclose(a, b, rtol=0, atol=1e-10)).sum())
    if uo_stage_mismatch:
        raise AssertionError(f'Fixed 6-hour reconstruction does not exactly reproduce primary urine-output stage; mismatch={uo_stage_mismatch}')
    print('FIXED 6-HOUR TARGET UO STAGE EXACT REPRODUCTION: PASS')
    def combine(cr, uo, rrt):
        cr = pd.to_numeric(cr, errors='coerce')
        uo = pd.to_numeric(uo, errors='coerce')
        rrt = pd.Series(rrt, index=cr.index).fillna(False).astype(bool)
        pos = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
        st = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
        st = st.where(st.ge(1), np.nan).mask(rrt, 3.0)
        return (pos, st)
    variant_selected = {}
    variant_target = {}
    for variant in ['fixed_6h', 'source_p95_cap', 'no_interval_cap']:
        z = target.copy().merge(stage_tables[variant], on='stay_id', how='left', validate='1:1')
        uocol = f'uo_stage_{variant}'
        z[f'combined_positive_{variant}'], z[f'combined_stage_{variant}'] = combine(z.creatinine_stage, z[uocol], pd.to_numeric(z.active_rrt_any_by_landmark, errors='coerce').fillna(0).eq(1))
        sel = first_positive(z, f'combined_positive_{variant}')
        variant_target[variant] = z
        variant_selected[variant] = sel
    fixed_sel = variant_selected['fixed_6h']
    if len(fixed_sel) != EXPECTED_PRIMARY_N:
        raise AssertionError(f'Fixed 6-hour primary cohort reproduction failed: N={len(fixed_sel)}')

    def compare(ref, alt, label):
        rs = set(ref.subject_id.astype(int))
        as_ = set(alt.subject_id.astype(int))
        common = rs & as_
        rstay = dict(zip(ref.subject_id.astype(int), ref.stay_id.astype(int)))
        astay = dict(zip(alt.subject_id.astype(int), alt.stay_id.astype(int)))
        replaced = sum((rstay[s] != astay[s] for s in common))
        return {'variant': label, 'subjects': len(alt), 'added_vs_fixed_6h': len(as_ - rs), 'dropped_vs_fixed_6h': len(rs - as_), 'common_subjects': len(common), 'replaced_stay_among_common': replaced, 'subject_jaccard': len(common) / len(rs | as_) if rs | as_ else np.nan}
    summary = pd.DataFrame([compare(fixed_sel, variant_selected[v], v) for v in ['fixed_6h', 'source_p95_cap', 'no_interval_cap']])
    stage_change_rows = []
    for v in ['source_p95_cap', 'no_interval_cap']:
        a = target[['stay_id', 'urine_output_stage']].merge(stage_tables[v][['stay_id', f'uo_stage_{v}']], on='stay_id', validate='1:1')
        x = pd.to_numeric(a.urine_output_stage, errors='coerce').fillna(-999).to_numpy(float)
        y = pd.to_numeric(a[f'uo_stage_{v}'], errors='coerce').fillna(-999).to_numpy(float)
        stage_change_rows.append({'variant': v, 'target_uo_stage_changed_stays': int((~np.isclose(x, y, rtol=0, atol=1e-10)).sum()), 'target_uo_positive_locked': int(pd.to_numeric(a.urine_output_stage, errors='coerce').fillna(0).ge(1).sum()), 'target_uo_positive_variant': int(pd.to_numeric(a[f'uo_stage_{v}'], errors='coerce').fillna(0).ge(1).sum())})
    stage_changes = pd.DataFrame(stage_change_rows)
    display_table(summary)
    display_table(stage_changes)
    print('FIXED 6-HOUR PRIMARY COHORT REPRODUCTION: PASS')
    fixed_reference = variant_selected["fixed_6h"]
    if len(fixed_reference) != EXPECTED_PRIMARY_N:
        raise AssertionError(f"Fixed 6-hour reconstruction N={len(fixed_reference)}, expected {EXPECTED_PRIMARY_N}.")
    if len(variant_selected["source_p95_cap"]) != EXPECTED_SOURCE_P95_N:
        raise AssertionError(f"Source-P95 interval-cap population N={len(variant_selected['source_p95_cap'])}, expected {EXPECTED_SOURCE_P95_N}.")
    if len(variant_selected["no_interval_cap"]) != EXPECTED_NO_INTERVAL_CAP_N:
        raise AssertionError(f"No-interval-cap population N={len(variant_selected['no_interval_cap'])}, expected {EXPECTED_NO_INTERVAL_CAP_N}.")

    # Persist only the two prespecified interval-cap sensitivity populations.
    filenames={"source_p95_cap":"11_uo_source_p95_cap.parquet","no_interval_cap":"12_uo_no_interval_cap.parquet"}
    for variant,filename in filenames.items():
        df=variant_selected[variant].copy()
        uocol=f"uo_stage_{variant}"; poscol=f"combined_positive_{variant}"; stagecol=f"combined_stage_{variant}"
        keep=[c for c in ["subject_id","hadm_id","stay_id","admittime","intime","landmark_time","creatinine_stage",uocol,"active_rrt_any_by_landmark",poscol,stagecol] if c in df.columns]
        saved=df[keep].copy().rename(columns={uocol:"urine_output_stage_variant",poscol:"combined_positive_variant",stagecol:"combined_stage_variant"})
        saved["variant"] = "uo_source_p95_cap" if variant=="source_p95_cap" else "uo_no_interval_cap"
        assert_preoutcome(saved, variant)
        saved.to_parquet(population_dir/filename,index=False)

    source_dist.to_csv(qc_dir/"source_interval_duration_distribution.csv",index=False)
    summary.to_csv(qc_dir/"population_membership_comparison.csv",index=False)
    stage_changes.to_csv(qc_dir/"target_stage_changes.csv",index=False)
    hard=pd.DataFrame([{
        "fixed_6h_target_stage_mismatch": uo_stage_mismatch,
        "fixed_6h_primary_n": len(fixed_reference),
        "source_p95_cap_n": len(variant_selected["source_p95_cap"]),
        "no_interval_cap_n": len(variant_selected["no_interval_cap"]),
        "outcome_used": False,
    }])
    if uo_stage_mismatch != 0: raise AssertionError("Fixed 6-hour reconstruction does not reproduce the primary urine-output stage.")
    hard.to_csv(qc_dir/"hard_guards.csv",index=False)
    manifest={
        "status":"PASS", "outcome_blind":True,
        "fixed_6h_primary_n":len(fixed_reference),
        "source_p95_cap_n":len(variant_selected["source_p95_cap"]),
        "no_interval_cap_n":len(variant_selected["no_interval_cap"]),
        "created_utc":datetime.now(timezone.utc).isoformat(),
    }
    (qc_dir/"urine_output_interval_cap_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("URINE-OUTPUT INTERVAL-CAP SENSITIVITY: PASS")
    display_table(hard)


if __name__ == "__main__":
    main()
