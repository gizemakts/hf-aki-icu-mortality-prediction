#!/usr/bin/env python3
"""Riley-framework sample-size reproduction and post hoc sensitivity."""
from __future__ import annotations
import argparse
import math
import numpy as np
import pandas as pd
from common import REPO, RESULTS, atomic_csv, atomic_json, disclosure_check, manifest, pmsampsize_binary, sha256, snapshot, verify_postflight, verify_protected

FIXED=["outputs/03_predictors/parameter_budget.csv","scripts/preprocessing/predictors/model_design.py","outputs/03_predictors/modeling_policy.json",
 "outputs/04_modeling/apparent_predictions.csv","outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_apparent_predictions.csv",
 "analyses/harmonized_internal_validation/INTERNAL_VALIDATION_SUMMARY.csv",
 "analyses/harmonized_internal_validation/RUN_MANIFEST.json"]


def apparent_cox_snell(y, p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-12,1-1e-12); n=len(y)
    prevalence=float(y.mean())
    ll_model=float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
    ll_null=float(np.sum(y*np.log(prevalence)+(1-y)*np.log(1-prevalence)))
    return float(1-np.exp(-2*(ll_model-ll_null)/n))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--execute",action="store_true"); args=ap.parse_args()
    budget=FIXED[0]; hashes=verify_protected([budget]);before=snapshot(FIXED)
    original=pd.read_csv(REPO/budget)
    reproduced=pd.DataFrame([{"model":"PRIMARY_TRANSPORTABLE_CORE",**pmsampsize_binary(.06,19,.20,.90,3457)},
                             {"model":"SECONDARY_BIOMARKER_AUGMENTED",**pmsampsize_binary(.06,21,.20,.90,3457)},
                             {"model":"ABSOLUTE_MAX_PARAMETER_GUARD_23",**pmsampsize_binary(.06,23,.20,.90,3457)}])
    checks={"criterion1_n_shrinkage":"criterion1_n_shrinkage","criterion2_n_r2_difference":"criterion2_n_r2_difference",
            "criterion3_n_overall_risk_precision":"criterion3_n_risk_precision","required_n":"required_n"}
    for old,new in checks.items():
        if not np.array_equal(original[old].to_numpy(int),reproduced[new].to_numpy(int)): raise RuntimeError(f"Planning reproduction mismatch: {old}")
    canonical_header=pd.read_csv(REPO/FIXED[3],nrows=0).columns.tolist();harmonized_header=pd.read_csv(REPO/FIXED[4],nrows=0).columns.tolist();summary_header=pd.read_csv(REPO/FIXED[5],nrows=0).columns.tolist()
    if not {"hospital_expire_flag","primary_apparent_probability_mi_mean"}.issubset(canonical_header):raise RuntimeError("Canonical apparent prediction schema mismatch")
    if not {"hospital_expire_flag","apparent_probability_mi_mean"}.issubset(harmonized_header):raise RuntimeError("Harmonized apparent prediction schema mismatch")
    if not {"metric","optimism_corrected"}.issubset(summary_header):raise RuntimeError("Harmonized validation summary schema mismatch")
    if not args.execute:
        print("DRY-RUN PASS: original canonical planning calculations reproduced exactly; no outcome/prediction rows loaded."); return
    rows=[]
    for cohort,n,events in [("canonical",3457,707),("harmonized",2098,537)]:
        prev=events/n
        for r2 in [.02,.04,.06,.08,.10,.12]: rows.append({"cohort":cohort,"r2_source":"plausible_assumption_grid",**pmsampsize_binary(r2,19,prev,.90,n)})
    canonical_path=REPO/FIXED[3]
    harmonized_path=REPO/FIXED[4]
    for cohort,path,probcol,n in [("canonical",canonical_path,"primary_apparent_probability_mi_mean",3457),
                                  ("harmonized",harmonized_path,"apparent_probability_mi_mean",2098)]:
        df=pd.read_csv(path,usecols=["hospital_expire_flag",probcol])
        r2=apparent_cox_snell(df.hospital_expire_flag,df[probcol])
        rows.append({"cohort":cohort,"r2_source":"post_hoc_apparent_MI_mean_prediction_proxy",**pmsampsize_binary(r2,19,float(df.hospital_expire_flag.mean()),.90,n)})
    comparison=pd.DataFrame(rows)
    slope_path=REPO/FIXED[5]
    slope_df=pd.read_csv(slope_path); slope=float(slope_df.loc[slope_df.metric.eq("calibration_slope"),"optimism_corrected"].iloc[0])
    interpretation=pd.DataFrame([{"cohort":"canonical","parameters":19,"n":3457,"events":707,"retrospective_assessment":True,
                                  "optimism_corrected_calibration_slope":np.nan,"note":"Planning reproduction is not prospective justification; assumed R2=0.06."},
                                 {"cohort":"harmonized","parameters":19,"n":2098,"events":537,"retrospective_assessment":True,
                                  "optimism_corrected_calibration_slope":slope,"note":"Slope 0.913878 is consistent with modest overfitting but is not a sample-size criterion or proof of adequacy."}])
    out=RESULTS/"analysis5"; out.mkdir(parents=True,exist_ok=True)
    p1=out/"ORIGINAL_PLANNING_REPRODUCTION.csv"; p2=out/"RILEY_SENSITIVITY_GRID.csv"; p3=out/"INTERPRETATION_CONTEXT.csv"
    atomic_csv(reproduced,p1); atomic_csv(comparison,p2); atomic_csv(interpretation,p3)
    inputs=before;post=verify_postflight(before);disc=disclosure_check([p1,p2,p3])
    atomic_json(manifest("analysis5_sample_size","COMPLETE",{"parameter_count":19,"target_shrinkage":.90,"r2_grid":[.02,.04,.06,.08,.10,.12]},inputs,[p1,p2,p3],post,disc),out/"RESULT_MANIFEST.json")
    print(f"COMPLETE: {out}")

if __name__=="__main__": main()
