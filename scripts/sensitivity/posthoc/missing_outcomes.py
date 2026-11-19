#!/usr/bin/env python3
"""Hypothetical eICU outcome-availability sensitivity; dry-run by default."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from common import (PRIMARY_PREDICTORS, REPO, RESULTS, atomic_csv, atomic_json,
                    disclosure_check, external_baseline_concordance, locked_probability, manifest,
                    scalar_metrics, snapshot, verify_postflight, verify_protected)

FULL="outputs/08_external_transport/eicu_outcome_attachment/eicu_prediction_ready_with_outcome_availability.parquet"
UNAVAILABLE="outputs/08_external_transport/eicu_outcome_attachment/eicu_outcome_unavailable_rows.parquet"
COHORT="outputs/08_external_transport/eicu_outcome_attachment/eicu_final_cohort_with_outcome_availability.parquet"
COEF="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"
PERFORMANCE="outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json"


def smd(a: pd.Series, b: pd.Series) -> float:
    a=pd.to_numeric(a,errors="coerce"); b=pd.to_numeric(b,errors="coerce")
    den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    return float((a.mean()-b.mean())/den) if den>0 else float("nan")


def scenario_outcome(base_outcome, evaluable, assigned_value):
    """Return a copy with only unavailable outcomes assigned to 0 or 1."""
    if assigned_value not in (0,1): raise ValueError("Scenario assignment must be binary")
    y=np.asarray(base_outcome,float).copy(); available=np.asarray(evaluable).astype(bool)
    if np.isnan(y[available]).any(): raise RuntimeError("Observed/evaluable outcomes cannot be missing")
    y[~available]=assigned_value
    return y


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--execute",action="store_true"); args=ap.parse_args()
    hashes=verify_protected([FULL,UNAVAILABLE,COEF,PERFORMANCE]);before=snapshot([FULL,UNAVAILABLE,COEF,PERFORMANCE])
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(REPO/FULL)
    if pf.metadata.num_rows!=1918 or not set(PRIMARY_PREDICTORS+["outcome_evaluable","hospital_expire_flag","hospital_id"]).issubset(pf.schema_arrow.names):
        raise RuntimeError("Fixed full outcome-availability matrix schema/count mismatch")
    if not args.execute:
        print("DRY-RUN PASS: Analysis 4A inputs/hashes validated; no patient rows loaded."); return
    df=pd.read_parquet(REPO/FULL); unavailable=pd.read_parquet(REPO/UNAVAILABLE)
    if int(df.outcome_evaluable.eq(0).sum())!=21 or len(unavailable)!=21: raise RuntimeError("Unavailable outcome count mismatch")
    prob=locked_probability(df)
    baseline=external_baseline_concordance(df,prob)
    rows=[]
    for label,value in [("all_21_survive",0),("all_21_die",1)]:
        y=scenario_outcome(pd.to_numeric(df.hospital_expire_flag,errors="coerce"),df.outcome_evaluable.eq(1),value)
        rows.append({"scenario":label,"analysis_n":1918,"observed_outcomes":1897,"hypothetical_assignments":21,"hypothetical_deaths":int(value*21),"total_deaths_under_scenario":int(y.sum()),**scalar_metrics(y,prob)})
    available=df.outcome_evaluable.eq(1); missing=~available
    comparison=[]
    for col in PRIMARY_PREDICTORS:
        comparison.append({"predictor":col,"evaluable_n":int(df.loc[available,col].notna().sum()),
                           "unavailable_n":int(df.loc[missing,col].notna().sum()),
                           "evaluable_mean":float(df.loc[available,col].mean()),
                           "unavailable_mean":float(df.loc[missing,col].mean()),
                           "standardized_mean_difference_unavailable_minus_evaluable":smd(df.loc[missing,col],df.loc[available,col])})
    hosp=df.groupby("hospital_id").outcome_evaluable.agg(target_n="size",evaluable_n="sum").reset_index()
    hosp["unavailable_n"]=hosp.target_n-hosp.evaluable_n
    zero=int(hosp.evaluable_n.eq(0).sum())
    # Disclosure-safe: individual hospital rows are never exported when either cell is <11.
    dist=pd.DataFrame([{"target_hospitals":int(len(hosp)),"hospitals_with_unavailable":int(hosp.unavailable_n.gt(0).sum()),
                        "hospitals_with_zero_evaluable":zero,"unavailable_patients":int(hosp.unavailable_n.sum()),
                        "small_cell_threshold":11,"hospital_rows_exported":False}])
    out=RESULTS/"analysis4_missing_outcomes"; out.mkdir(parents=True,exist_ok=True)
    p1=out/"HYPOTHETICAL_SCENARIOS.csv"; p2=out/"OUTCOME_AVAILABILITY_CHARACTERISTICS.csv"; p3=out/"HOSPITAL_DISTRIBUTION_DISCLOSURE_SAFE.csv"
    atomic_csv(pd.DataFrame(rows),p1); atomic_csv(pd.DataFrame(comparison),p2); atomic_csv(dist,p3)
    p4=out/"LOCKED_BASELINE_CONCORDANCE.json";atomic_json(baseline,p4);post=verify_postflight(before);disc=disclosure_check([p1,p2,p3,p4],11)
    atomic_json(manifest("analysis4_missing_outcomes","COMPLETE",{"small_cell_threshold":11,"scenarios":[0,1],"reference_gate":"exact locked scalar reproduction"},hashes,[p1,p2,p3,p4],post,disc),out/"RESULT_MANIFEST.json")
    print(f"COMPLETE: {out}")

if __name__=="__main__": main()
