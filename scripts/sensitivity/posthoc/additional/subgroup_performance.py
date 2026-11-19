#!/usr/bin/env python3
"""Post hoc exploratory locked-risk subgroup performance with cluster uncertainty."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,roc_auc_score

HERE=Path(__file__).resolve(); WORKSPACE=HERE.parents[1]; REPO=WORKSPACE.parents[2]
sys.path.insert(0,str(WORKSPACE))
from common import (atomic_csv,atomic_json,external_baseline_concordance,locked_probability,
                    safe_logit,sha256,snapshot,verify_postflight)
import statsmodels.api as sm

FULL="outputs/08_external_transport/eicu_outcome_attachment/eicu_prediction_ready_with_outcome_availability.parquet"
EVAL="outputs/08_external_transport/eicu_outcome_attachment/eicu_external_validation_evaluable.parquet"
COEF="outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"
PERF="outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json"
RECEIPT="work/posthoc_jicm/checkpoints/EXTERNAL_BASELINE_CONCORDANCE.json"
FEAS="work/posthoc_jicm/additional_validation/results/feasibility/SUBGROUP_FEASIBILITY.csv"
OUT=WORKSPACE/"additional"/"results"/"subgroups"
SEED=2026102904; REPLICATES=2000

def metrics(y,p,allow_slope):
    y=np.asarray(y,float);p=np.asarray(p,float);eta=safe_logit(p)
    intercept=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=eta).fit(maxiter=200,tol=1e-10)
    out={"auroc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),
         "observed_over_expected":float(y.sum()/p.sum()),"calibration_intercept":float(intercept.params[0])}
    if allow_slope:
        slope=sm.GLM(y,sm.add_constant(eta,has_constant="add"),family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
        out["calibration_slope"]=float(slope.params[1])
    return out

def cluster_intervals(frame,allow_slope):
    hospitals=frame.hospital_id.unique();rows=[]
    for rep in range(1,REPLICATES+1):
        rng=np.random.default_rng(np.random.SeedSequence([SEED,rep]))
        draw=rng.choice(hospitals,len(hospitals),replace=True)
        sampled=pd.concat([frame.loc[frame.hospital_id.eq(h)] for h in draw],ignore_index=True)
        if sampled.hospital_expire_flag.nunique()!=2:continue
        try: rows.append(metrics(sampled.hospital_expire_flag,sampled.locked_probability,allow_slope))
        except Exception: continue
    if len(rows)<1900:raise RuntimeError(f"Insufficient valid cluster replicates: {len(rows)}")
    boot=pd.DataFrame(rows);result={}
    for col in boot.columns:
        lo,hi=np.quantile(boot[col],[.025,.975]);result[col]={"p025":float(lo),"p975":float(hi),"valid_replicates":len(boot)}
    return result

def main():
    inputs=[FULL,EVAL,COEF,PERF,RECEIPT,FEAS];before=snapshot(inputs)
    feas=pd.read_csv(REPO/FEAS);eligible=feas.loc[feas.performance_gate.eq("GO")]
    full=pd.read_parquet(REPO/FULL);baseline=external_baseline_concordance(full,locked_probability(full))
    receipt=json.loads((REPO/RECEIPT).read_text(encoding="utf-8"))
    if receipt.get("status")!="PASS" or baseline["status"]!="PASS":raise RuntimeError("Locked external baseline gate failed")
    d=pd.read_parquet(REPO/EVAL);d["locked_probability"]=locked_probability(d)
    d["age_group"]=pd.cut(d.age_per10y,[-np.inf,6.5,7.5,np.inf],right=False,labels=["age_lt65","age_65_74","age_ge75"])
    d["sex_group"]=d.sex_female.map({1.:"female",0.:"male"})
    point=[];interval=[]
    for _,gate in eligible.iterrows():
        variable=gate.variable;label=gate.group;column="age_group" if variable=="age" else "sex_group"
        g=d.loc[d[column].eq(label)].copy();allow=gate.calibration_slope_gate=="GO"
        observed=metrics(g.hospital_expire_flag,g.locked_probability,allow)
        point.append({"variable":variable,"group":label,"n":len(g),"deaths":int(g.hospital_expire_flag.sum()),
                      "survivors":int(len(g)-g.hospital_expire_flag.sum()),"contributing_hospitals":g.hospital_id.nunique(),
                      **observed,"calibration_slope_status":"ESTIMATED" if allow else "INSUFFICIENT_EVENTS",
                      "interpretation":"post hoc exploratory subgroup performance; not fairness evidence"})
        ci=cluster_intervals(g,allow)
        for metric,values in ci.items():interval.append({"variable":variable,"group":label,"metric":metric,**values,
          "method":"hospital-cluster percentile bootstrap; conditional on locked model/FCE"})
    OUT.mkdir(parents=True,exist_ok=True);p1=OUT/"SUBGROUP_PERFORMANCE.csv";p2=OUT/"SUBGROUP_CLUSTER_INTERVALS.csv"
    atomic_csv(pd.DataFrame(point),p1);atomic_csv(pd.DataFrame(interval),p2)
    post=verify_postflight(before)
    manifest={"status":"COMPLETE","post_hoc_exploratory":True,"not_fairness_evidence":True,"no_model_update_or_recalibration":True,
      "bootstrap":{"replicates":REPLICATES,"seed":SEED,"cluster":"hospital"},"code_sha256":sha256(HERE),"input_hashes":before,
      "postflight":post,"privacy":{"patient_rows_exported":False,"hospital_ids_exported":False,"small_cell_threshold":11},
      "outputs":{p1.name:sha256(p1),p2.name:sha256(p2)}}
    atomic_json(manifest,OUT/"RESULT_MANIFEST.json");print(OUT)

if __name__=="__main__":main()
