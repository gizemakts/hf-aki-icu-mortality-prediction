#!/usr/bin/env python3
"""Outcome-blind post hoc FCE alternative-start stability analysis."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve();WORKSPACE=HERE.parents[1];REPO=WORKSPACE.parents[2]
sys.path.insert(0,str(WORKSPACE))
from common import PRIMARY_PREDICTORS,atomic_csv,atomic_json,locked_probability,sha256,snapshot,verify_postflight
from eicu_imputation_robustness import FCE_TARGETS,iterative_imputer,summarize_shifts

RAW="outputs/08_external_transport/eicu_final_preoutcome/eicu_predictor_matrix_preimputation.parquet"
FIXED="outputs/08_external_transport/eicu_final_preoutcome/eicu_prediction_ready_matrix_preoutcome.parquet"
DEV="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_development_matrix_with_outcome.parquet"
COEF="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"
FCE="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_pooled_conditional_coefficients.csv"
INIT="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_initialization_values.csv"
KEYS=["unique_pid","hstay_id","stay_id","hospital_id"]
OUT=WORKSPACE/"additional"/"results"/"fce_start_stability"

def main():
    inputs=[RAW,FIXED,DEV,COEF,FCE,INIT];before=snapshot(inputs)
    raw=pd.read_parquet(REPO/RAW).sort_values(KEYS).reset_index(drop=True)
    fixed=pd.read_parquet(REPO/FIXED).sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(raw[KEYS],fixed[KEYS],check_dtype=False,check_exact=True,check_like=False)
    dev=pd.read_parquet(REPO/DEV,columns=FCE_TARGETS)
    fixed_init=pd.read_csv(REPO/INIT).set_index("target_predictor").initialization_value.to_dict()
    starts={"fixed_mean":fixed_init,
      "development_median":{t:float(dev[t].median()) for t in FCE_TARGETS},
      "development_p10":{t:float(dev[t].quantile(.10)) for t in FCE_TARGETS},
      "development_p90":{t:float(dev[t].quantile(.90)) for t in FCE_TARGETS}}
    reference=locked_probability(fixed);traces=[];shift=[];differences=[]
    for label,start in starts.items():
        completed,trace,prob=iterative_imputer(raw,start,feedback=True)
        trace.insert(0,"initialization",label);traces.append(trace)
        shift.append(summarize_shifts(reference,prob,label))
        for target in FCE_TARGETS:
            mask=raw[target].isna().to_numpy();delta=completed.loc[mask,target].to_numpy(float)-fixed.loc[mask,target].to_numpy(float)
            differences.append({"initialization":label,"target":target,"imputed_n":int(mask.sum()),
              "mean_difference":float(delta.mean()) if len(delta) else 0.,"mean_absolute_difference":float(np.abs(delta).mean()) if len(delta) else 0.,
              "maximum_absolute_difference":float(np.abs(delta).max()) if len(delta) else 0.})
        if label=="fixed_mean":
            if not np.allclose(prob,reference,rtol=0,atol=1e-12):raise RuntimeError("Fixed-start probability reproduction failed")
            if not np.allclose(completed[PRIMARY_PREDICTORS],fixed[PRIMARY_PREDICTORS],rtol=0,atol=1e-10):raise RuntimeError("Fixed-start matrix reproduction failed")
    OUT.mkdir(parents=True,exist_ok=True);p1=OUT/"CONVERGENCE_TRACE.csv";p2=OUT/"PREDICTION_SHIFT_SUMMARY.csv";p3=OUT/"IMPUTED_VALUE_SHIFT_SUMMARY.csv"
    atomic_csv(pd.concat(traces,ignore_index=True),p1);atomic_csv(pd.DataFrame(shift),p2);atomic_csv(pd.DataFrame(differences),p3)
    post=verify_postflight(before)
    manifest={"status":"COMPLETE","post_hoc_exploratory":True,"outcome_accessed":False,"alternative_start_stability_only":True,
      "does_not_establish_fixed_point_uniqueness":True,"does_not_resolve_missing_data_uncertainty":True,"code_sha256":sha256(HERE),
      "input_hashes":before,"postflight":post,"privacy":{"aggregate_only":True,"patient_rows_exported":False,"hospital_ids_exported":False},
      "outputs":{p.name:sha256(p) for p in [p1,p2,p3]}}
    atomic_json(manifest,OUT/"RESULT_MANIFEST.json");print(OUT)

if __name__=="__main__":main()
