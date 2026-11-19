#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow

EXPECTED_PREOUTCOME_LOCK_SHA="1d06e8d1788945a2468bfe164ae6150e94ed1d8910be4fa3387c21d0f67b8714"
EXPECTED_PREOUTCOME_VERIFICATION_SHA="0a530bd30fe1ff131cb205c073a9f3d99f5e5d3f6585e1f957121de24397a604"
EXPECTED_PATIENT_RAW_SHA="2acf25f75144c5fc8a2a5b68c6741edac2dc7ca51021127f73e12f7428dc392b"
EXPECTED_FULL_N=1918
EXPECTED_UNAVAILABLE_N=21
EXPECTED_EVALUABLE_N=1897
EXPECTED_HOSPITALS=125
EXPECTED_COHORT_SHA="13d71e26ac0861dee20889b9f4a121748d12a0bbd2cde17d1dcb968c2c058ea8"
EXPECTED_PREIMPUTATION_SHA="43faf1beced67ea3e6cdd4ec3733160d9fd61b14932aba885fd7998aa14a168e"
EXPECTED_READY_SHA="9269e4c8eb858ea3a4e7c559d533827ea72788e44d30060a5502433b55591e66"
EXPECTED_IDENTITY={
"final_uniquePid_set_sha256_sorted_newline":"77a431ab239d2514576c9dc8732c1d708a3ff7b0109592f0ed09fff52bf0a570",
"final_hstay_id_set_sha256_sorted_newline":"7df2c7e79e33471ba0045a07dfd5615db379d905856d350b775bb4aa80fd6cb3",
"final_stay_id_set_sha256_sorted_newline":"df5c8f0ebad9cee7439ac4a07c03c54a255a8c2397ed7836399e688fc55ab92f",
}
PRIMARY_PREDICTORS=["age_per10y","sex_female","heart_rate_max_per10","map_min_per10","resp_rate_max_per5","temperature_max_abs_deviation_from_37c","spo2_deficit_per5","gcs_deficit_from15","hemoglobin_min_gdl","wbc_max_log2","platelet_min_log2","bun_max_log2","bicarbonate_min_per5","sodium_max_abs_deviation_from140_per5","potassium_max_mmol_l","glucose_max_log2","vasoactive_any_by_landmark","kdigo_stage_2_indicator","kdigo_stage_3_indicator"]
RAW_STEMS={"patient.csv.gz":"patient","pastHistory.csv.gz":"pastHistory","nurseCharting.csv.gz":"nurseCharting","lab.csv.gz":"lab","infusionDrug.csv.gz":"infusionDrug","treatment.csv.gz":"treatment","intakeOutput.csv.gz":"intakeOutput","diagnosis.csv.gz":"diagnosis"}
OUTCOME="hospital_expire_flag"

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def verify_hash(path, expected, label):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"{label} missing: {path}")
    got=sha256_file(path)
    if got.lower()!=expected.lower(): raise AssertionError(f"{label} SHA mismatch: {got} != {expected}")
    return got

def rel(project,path): return str(Path(path).resolve().relative_to(project.resolve())).replace("\\","/")

def discover_raw(root,stem):
    wanted={f"{stem}.csv".lower(),f"{stem}.csv.gz".lower()}
    matches=sorted([p.resolve() for p in root.rglob("*") if p.is_file() and p.name.lower() in wanted],key=lambda p:p.as_posix().lower())
    if len(matches)!=1: raise AssertionError(f"{stem}: expected exactly one raw file, found {len(matches)}")
    return matches[0]

def sha_strings(values):
    vals=sorted({str(v).strip() for v in values if pd.notna(v)})
    return hashlib.sha256(("\n".join(vals)+"\n").encode()).hexdigest()

def sha_ints(values):
    vals=sorted({int(v) for v in values if pd.notna(v)})
    return hashlib.sha256(("\n".join(str(v) for v in vals)+"\n").encode()).hexdigest()

def ordered_sha(df,cols):
    rows=["\t".join("" if pd.isna(v) else str(v) for v in row) for row in df[cols].itertuples(index=False,name=None)]
    return hashlib.sha256(("\n".join(rows)+"\n").encode()).hexdigest()

def exact_equal(a,b,cols):
    if len(a)!=len(b): return False
    for c in cols:
        x,y=a[c],b[c]
        if pd.api.types.is_numeric_dtype(x) or pd.api.types.is_numeric_dtype(y):
            if not np.array_equal(pd.to_numeric(x,errors="coerce").to_numpy(float),pd.to_numeric(y,errors="coerce").to_numpy(float),equal_nan=True): return False
        else:
            if not x.astype("string").fillna("<NA>").equals(y.astype("string").fillna("<NA>")): return False
    return True

def patient_columns(path):
    need={"patientunitstayid","patienthealthsystemstayid","uniquepid","hospitalid","hospitaldischargestatus"}
    header=pd.read_csv(path,nrows=0).columns.tolist()
    m={str(c).strip().lower():c for c in header}
    miss=sorted(need-set(m))
    if miss: raise AssertionError(f"patient table missing required columns: {miss}")
    return [m[x] for x in sorted(need)],m

def smd(a,b):
    a=pd.to_numeric(a,errors="coerce").dropna().to_numpy(float); b=pd.to_numeric(b,errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2:return float("nan")
    pooled=math.sqrt(max((np.var(a,ddof=1)+np.var(b,ddof=1))/2,0))
    if pooled==0:return 0.0 if np.mean(a)==np.mean(b) else float("inf")
    return float((np.mean(a)-np.mean(b))/pooled)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--eicu-root",required=True)
    ap.add_argument("--preoutcome-lock-manifest",type=Path,required=True,
                    help="Original restricted pre-outcome lock manifest")
    ap.add_argument("--preoutcome-verification",type=Path,required=True,
                    help="Original restricted pre-outcome verification record")
    args=ap.parse_args()
    project=Path(args.project_root).resolve(); root=Path(args.eicu_root).resolve()
    lock=Path(args.preoutcome_lock_manifest).expanduser().resolve()
    verification=Path(args.preoutcome_verification).expanduser().resolve()
    verify_hash(lock,EXPECTED_PREOUTCOME_LOCK_SHA,"Pre-outcome lock manifest")
    verify_hash(verification,EXPECTED_PREOUTCOME_VERIFICATION_SHA,"Pre-outcome verification record")
    pre=project/"outputs"/"08_external_transport"/"eicu_final_preoutcome"
    preqc=project/"quality_control"/"eicu_external_validation"/"eicu_final_preoutcome"
    out=project/"outputs"/"08_external_transport"/"eicu_outcome_attachment"
    qc=project/"quality_control"/"eicu_external_validation"/"eicu_outcome_attachment"
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"{out} already non-empty")
    if qc.exists() and any(qc.iterdir()): raise FileExistsError(f"{qc} already non-empty")
    out.mkdir(parents=True,exist_ok=True); qc.mkdir(parents=True,exist_ok=True)

    cohort_path=pre/"eicu_final_cohort_preoutcome.parquet"; pre_path=pre/"eicu_predictor_matrix_preimputation.parquet"; ready_path=pre/"eicu_prediction_ready_matrix_preoutcome.parquet"
    verify_hash(cohort_path,EXPECTED_COHORT_SHA,"Fixed cohort"); verify_hash(pre_path,EXPECTED_PREIMPUTATION_SHA,"Pre-imputation"); verify_hash(ready_path,EXPECTED_READY_SHA,"Ready")
    cohort=pd.read_parquet(cohort_path); ready=pd.read_parquet(ready_path)
    if len(cohort)!=EXPECTED_FULL_N or len(ready)!=EXPECTED_FULL_N: raise AssertionError("Fixed N drift")
    for c in ("stay_id","hstay_id","hospital_id"):
        cohort[c]=pd.to_numeric(cohort[c],errors="raise").astype("int64"); ready[c]=pd.to_numeric(ready[c],errors="raise").astype("int64")
    cohort["unique_pid"]=cohort["unique_pid"].astype("string").str.strip(); ready["unique_pid"]=ready["unique_pid"].astype("string").str.strip()
    if cohort["stay_id"].duplicated().any() or cohort["unique_pid"].duplicated().any() or ready["stay_id"].duplicated().any(): raise AssertionError("Fixed uniqueness drift")
    if not cohort["stay_id"].eq(ready["stay_id"]).all(): raise AssertionError("Fixed cohort/ready order mismatch")
    if ready[PRIMARY_PREDICTORS].isna().any().any() or not np.isfinite(ready[PRIMARY_PREDICTORS].to_numpy(float)).all(): raise AssertionError("Fixed ready matrix invalid")
    actual={"final_uniquePid_set_sha256_sorted_newline":sha_strings(cohort["unique_pid"]),"final_hstay_id_set_sha256_sorted_newline":sha_ints(cohort["hstay_id"]),"final_stay_id_set_sha256_sorted_newline":sha_ints(cohort["stay_id"])}
    if actual!=EXPECTED_IDENTITY: raise AssertionError("Fixed identity drift")
    order_pre=ordered_sha(cohort,["unique_pid","hstay_id","stay_id","hospital_id"])

    fm=json.loads(lock.read_text(encoding="utf-8"))
    fingerprints=fm.get("raw_eicu_input_fingerprints")
    if not isinstance(fingerprints,dict) or not set(RAW_STEMS).issubset(fingerprints):
        raise AssertionError("Lock manifest lacks required original eICU fingerprints")
    for fn in RAW_STEMS:
        entry=fingerprints[fn]
        if not isinstance(entry,dict) or not {"sha256","size_bytes"}.issubset(entry):
            raise AssertionError(f"Lock manifest fingerprint invalid: {fn}")
    rawpaths={}; rawrows=[]
    print("="*96); print("eICU OUTCOME-EVALUABLE ATTACHMENT BUILD"); print("="*96)
    print("Performance   : NOT COMPUTED"); print("Recalibration : NOT PERFORMED"); print("Re-hashing raw eICU inputs...")
    for fn,stem in RAW_STEMS.items():
        p=discover_raw(root,stem); exp=fm["raw_eicu_input_fingerprints"][fn]; got=sha256_file(p); size=p.stat().st_size
        if got!=exp["sha256"] or int(size)!=int(exp["size_bytes"]): raise AssertionError(f"Raw input drift: {fn}")
        rawpaths[fn]=p; rawrows.append({"filename":fn,"size_bytes":int(size),"expected_sha256":exp["sha256"],"observed_sha256":got,"status":"PASS"}); print("  PASS",fn)
    if sha256_file(rawpaths["patient.csv.gz"])!=EXPECTED_PATIENT_RAW_SHA: raise AssertionError("patient.csv.gz drift")

    patient_path=rawpaths["patient.csv.gz"]; usecols,m=patient_columns(patient_path)
    patient=pd.read_csv(patient_path,usecols=usecols,low_memory=False).rename(columns={m["patientunitstayid"]:"stay_id",m["patienthealthsystemstayid"]:"hstay_id",m["uniquepid"]:"unique_pid",m["hospitalid"]:"hospital_id",m["hospitaldischargestatus"]:"hospital_discharge_status"})
    for c in ("stay_id","hstay_id","hospital_id"): patient[c]=pd.to_numeric(patient[c],errors="raise").astype("int64")
    patient["unique_pid"]=patient["unique_pid"].astype("string").str.strip()
    if patient["stay_id"].duplicated().any(): raise AssertionError("Raw patient duplicate stay")
    patient["_status_norm"]=patient["hospital_discharge_status"].astype("string").str.strip().str.casefold()
    patient["_valid"]=patient["_status_norm"].isin({"alive","expired"})
    hrows=patient.loc[patient["hstay_id"].isin(set(cohort["hstay_id"]))].copy()
    ident=hrows.groupby("hstay_id").agg(unique_pid_n=("unique_pid","nunique"),hospital_id_n=("hospital_id","nunique"))
    if len(ident)!=EXPECTED_FULL_N or not ident["unique_pid_n"].eq(1).all() or not ident["hospital_id_n"].eq(1).all(): raise AssertionError("Hospital-stay identity integrity failure")

    outcomes={}; rr=[]
    for h,g in hrows.groupby("hstay_id",sort=True):
        valid=sorted(set(g.loc[g["_valid"],"_status_norm"].astype(str)))
        if len(valid)==1:
            oe=1; y=0 if valid[0]=="alive" else 1; state="OUTCOME_EVALUABLE"; outcomes[int(h)]=y
        elif len(valid)==0:
            oe=0; y=None; state="OUTCOME_UNAVAILABLE_SOURCE_DATABASE"
        else:
            raise AssertionError(f"Conflicting outcome values in hstay_id={h}: {valid}")
        rr.append({"hstay_id":int(h),"icu_rows_in_hospital_stay":int(len(g)),"valid_status_rows_n":int(g["_valid"].sum()),"invalid_or_missing_status_rows_n":int((~g["_valid"]).sum()),"unique_valid_status_n":len(valid),"outcome_evaluable":oe,"resolution_state":state})
    resolution=pd.DataFrame(rr)
    n_eval=int(resolution["outcome_evaluable"].sum()); n_un=int((resolution["outcome_evaluable"]==0).sum())
    if n_eval!=EXPECTED_EVALUABLE_N or n_un!=EXPECTED_UNAVAILABLE_N: raise AssertionError(f"Evaluability drift: evaluable={n_eval} unavailable={n_un}")

    evalmap=dict(zip(resolution["hstay_id"].astype(int),resolution["outcome_evaluable"].astype(int)))
    full=cohort.copy(); full["outcome_evaluable"]=full["hstay_id"].map(evalmap).astype("int8")
    full[OUTCOME]=pd.Series([outcomes.get(int(h)) for h in full["hstay_id"]],dtype="Int8")
    if int(full[OUTCOME].isna().sum())!=EXPECTED_UNAVAILABLE_N: raise AssertionError("Nullable outcome count mismatch")
    if full.loc[full["outcome_evaluable"]==1,OUTCOME].isna().any(): raise AssertionError("Evaluable row missing outcome")
    if not full.loc[full["outcome_evaluable"]==0,OUTCOME].isna().all(): raise AssertionError("Unavailable row has outcome")
    ready_full=ready.copy(); ready_full["outcome_evaluable"]=full["outcome_evaluable"].to_numpy(); ready_full[OUTCOME]=full[OUTCOME].to_numpy()
    if not exact_equal(cohort,full,cohort.columns.tolist()) or not exact_equal(ready,ready_full,ready.columns.tolist()): raise AssertionError("Fixed values changed")
    if ordered_sha(full,["unique_pid","hstay_id","stay_id","hospital_id"])!=order_pre: raise AssertionError("Full membership/order changed")

    mask=full["outcome_evaluable"]==1
    evaluable=ready_full.loc[mask].reset_index(drop=True); unavailable=full.loc[~mask,["unique_pid","hstay_id","stay_id","hospital_id","eicu_cr_rrt_stage","outcome_evaluable",OUTCOME]].reset_index(drop=True)
    if len(evaluable)!=EXPECTED_EVALUABLE_N or len(unavailable)!=EXPECTED_UNAVAILABLE_N: raise AssertionError("Subset N mismatch")
    y=pd.to_numeric(evaluable[OUTCOME],errors="raise").astype(int)
    if not set(y.unique()).issubset({0,1}): raise AssertionError("Outcome nonbinary")

    hosp=full.groupby("hospital_id",as_index=False).agg(fixed_target_n=("outcome_evaluable","size"),outcome_evaluable_n=("outcome_evaluable","sum")); hosp["outcome_unavailable_n"]=hosp["fixed_target_n"]-hosp["outcome_evaluable_n"]; hosp["outcome_unavailable_pct"]=100*hosp["outcome_unavailable_n"]/hosp["fixed_target_n"]
    stage=full.groupby("eicu_cr_rrt_stage",as_index=False).agg(fixed_target_n=("outcome_evaluable","size"),outcome_evaluable_n=("outcome_evaluable","sum")); stage["outcome_unavailable_n"]=stage["fixed_target_n"]-stage["outcome_evaluable_n"]; stage["outcome_unavailable_pct"]=100*stage["outcome_unavailable_n"]/stage["fixed_target_n"]
    estays=set(evaluable["stay_id"].astype(int)); re=ready[ready["stay_id"].isin(estays)]; ru=ready[~ready["stay_id"].isin(estays)]
    comp=[]
    for p in PRIMARY_PREDICTORS:
        a=pd.to_numeric(re[p],errors="raise"); b=pd.to_numeric(ru[p],errors="raise")
        comp.append({"predictor":p,"evaluable_n":len(a),"unavailable_n":len(b),"evaluable_mean":float(a.mean()),"unavailable_mean":float(b.mean()),"evaluable_sd":float(a.std(ddof=1)),"unavailable_sd":float(b.std(ddof=1)),"evaluable_median":float(a.median()),"unavailable_median":float(b.median()),"smd_evaluable_minus_unavailable":smd(a,b)})
    comp=pd.DataFrame(comp)

    p_full=out/"eicu_final_cohort_with_outcome_availability.parquet"; p_ready=out/"eicu_prediction_ready_with_outcome_availability.parquet"; p_eval=out/"eicu_external_validation_evaluable.parquet"; p_un=out/"eicu_outcome_unavailable_rows.parquet"
    full.to_parquet(p_full,index=False); ready_full.to_parquet(p_ready,index=False); evaluable.to_parquet(p_eval,index=False); unavailable.to_parquet(p_un,index=False)
    p_hosp=qc/"OUTCOME_EVALUABILITY_BY_HOSPITAL.csv"; hosp.to_csv(p_hosp,index=False)
    p_stage=qc/"OUTCOME_EVALUABILITY_BY_KDIGO_STAGE.csv"; stage.to_csv(p_stage,index=False)
    p_comp=qc/"OUTCOME_EVALUABILITY_PREDICTOR_COMPARISON.csv"; comp.to_csv(p_comp,index=False)
    p_raw=qc/"RAW_EICU_HASH_RECHECK_BEFORE_OUTCOME.csv"; pd.DataFrame(rawrows).to_csv(p_raw,index=False)

    nh_eval=int((hosp["outcome_evaluable_n"]>0).sum()); nh_un=int((hosp["outcome_unavailable_n"]>0).sum()); deaths=int(y.sum()); alive=len(y)-deaths; rate=float(y.mean())
    flow=pd.DataFrame([
        {"metric":"fixed_target_cohort_n","value":EXPECTED_FULL_N},{"metric":"outcome_unavailable_n","value":EXPECTED_UNAVAILABLE_N},{"metric":"outcome_unavailable_pct","value":100*EXPECTED_UNAVAILABLE_N/EXPECTED_FULL_N},{"metric":"outcome_evaluable_n","value":EXPECTED_EVALUABLE_N},{"metric":"fixed_target_hospitals_n","value":EXPECTED_HOSPITALS},{"metric":"outcome_evaluable_hospitals_n","value":nh_eval},{"metric":"hospitals_with_outcome_unavailable_n","value":nh_un},{"metric":"outcome_evaluable_alive_n","value":alive},{"metric":"outcome_evaluable_expired_n","value":deaths},{"metric":"outcome_evaluable_event_rate","value":rate},{"metric":"performance_metrics_computed","value":False},{"metric":"recalibration_performed","value":False}
    ]); p_flow=qc/"OUTCOME_EVALUABILITY_FLOW.csv"; flow.to_csv(p_flow,index=False)

    ids={"full_uniquePid_set_sha256_sorted_newline":sha_strings(full["unique_pid"]),"full_hstay_id_set_sha256_sorted_newline":sha_ints(full["hstay_id"]),"full_stay_id_set_sha256_sorted_newline":sha_ints(full["stay_id"]),"evaluable_uniquePid_set_sha256_sorted_newline":sha_strings(evaluable["unique_pid"]),"evaluable_hstay_id_set_sha256_sorted_newline":sha_ints(evaluable["hstay_id"]),"evaluable_stay_id_set_sha256_sorted_newline":sha_ints(evaluable["stay_id"]),"unavailable_hstay_id_set_sha256_sorted_newline":sha_ints(unavailable["hstay_id"]),"full_ordered_identity_sha256":order_pre}
    p_ids=qc/"OUTCOME_EVALUABILITY_IDENTITY_HASHES.json"; p_ids.write_text(json.dumps(ids,indent=2)+"\n",encoding="utf-8")
    status=pd.DataFrame({"status":["Alive","Expired","Unavailable"],"n":[alive,deaths,EXPECTED_UNAVAILABLE_N],"denominator":[EXPECTED_EVALUABLE_N,EXPECTED_EVALUABLE_N,EXPECTED_FULL_N]}); status["pct"]=100*status["n"]/status["denominator"]; p_status=qc/"OUTCOME_STATUS_AND_AVAILABILITY_DISTRIBUTION.csv"; status.to_csv(p_status,index=False)
    access=pd.DataFrame([{"source_file":patient_path.name,"raw_column":m["patienthealthsystemstayid"],"purpose":"hospital-stay outcome availability key","outcome_field":False},{"source_file":patient_path.name,"raw_column":m["hospitaldischargestatus"],"purpose":"database-native in-hospital mortality outcome","outcome_field":True}]); p_access=qc/"OUTCOME_FIELD_ACCESS_LEDGER.csv"; access.to_csv(p_access,index=False)
    generated=[p_full,p_ready,p_eval,p_un,p_hosp,p_stage,p_comp,p_raw,p_flow,p_ids,p_status,p_access]
    hashes={rel(project,p):sha256_file(p) for p in generated}
    src=Path(__file__).resolve(); srcsha=sha256_file(src)
    manifest={"status":"PASS","gate":"eICU outcome-evaluable attachment","created_utc":datetime.now(timezone.utc).isoformat(),"analysis_scope":"outcome_attachment","fixed_target_cohort_n":EXPECTED_FULL_N,"outcome_unavailable_n":EXPECTED_UNAVAILABLE_N,"outcome_evaluable_n":EXPECTED_EVALUABLE_N,"outcome_evaluable_hospitals_n":nh_eval,"hospitals_with_outcome_unavailable_n":nh_un,"outcome_evaluable_event_summary":{"alive_n":alive,"expired_n":deaths,"event_rate":rate},"outcome_definition":{"source":"patient.hospitalDischargeStatus","mapping":{"Alive":0,"Expired":1},"missing_outcome_imputed":False},"target_cohort_membership_changed":False,"predictor_values_changed":False,"performance_metrics_computed":False,"recalibration_performed":False,"preoutcome_anchor":{"lock_manifest_sha256":EXPECTED_PREOUTCOME_LOCK_SHA,"lock_verification_sha256":EXPECTED_PREOUTCOME_VERIFICATION_SHA,"cohort_sha256":EXPECTED_COHORT_SHA,"preimputation_sha256":EXPECTED_PREIMPUTATION_SHA,"prediction_ready_sha256":EXPECTED_READY_SHA},"identity_hashes":ids,"generated_artifact_hashes":hashes,"source_script_path":rel(project,src),"source_script_sha256":srcsha,"runtime_versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"pyarrow":pyarrow.__version__},"performance_computation_authorized":False,"recalibration_authorized":False,"next_action":"independent review and governed attachment/evaluability lock"}
    p_manifest=qc/"EICU_OUTCOME_EVALUABILITY_ATTACHMENT_MANIFEST.json"; p_manifest.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); msha=sha256_file(p_manifest)
    receipt={"status":"PASS","gate":manifest["gate"],"manifest_sha256":msha,"source_script_sha256":srcsha,"full_target_n":EXPECTED_FULL_N,"outcome_unavailable_n":EXPECTED_UNAVAILABLE_N,"outcome_evaluable_n":EXPECTED_EVALUABLE_N,"full_membership_invariance_pass":True,"predictor_invariance_pass":True,"outcome_imputation_performed":False,"performance_metrics_computed":False,"recalibration_performed":False,"analysis_scope":"outcome_attachment","performance_computation_authorized":False,"verified_utc":datetime.now(timezone.utc).isoformat()}
    p_receipt=qc/"EICU_OUTCOME_EVALUABILITY_ATTACHMENT_RECEIPT.json"; p_receipt.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

    print(); print("="*96); print("eICU OUTCOME-EVALUABLE ATTACHMENT: PASS"); print("="*96)
    print("Fixed target cohort N             : 1,918"); print("Outcome unavailable N              : 21"); print("Outcome-evaluable N                : 1,897"); print("Outcome-evaluable hospitals        :",nh_eval); print("Hospitals with unavailable outcome :",nh_un); print("Outcome-evaluable Alive            :",alive); print("Outcome-evaluable Expired          :",deaths); print("Outcome-evaluable mortality rate   :",f"{100*rate:.4f}%"); print("Full target membership changed     : NO"); print("Fixed predictor values changed    : NO"); print("Outcome imputation performed       : NO"); print("Performance metrics computed       : NO"); print("Recalibration performed            : NO"); print("Performance authorized             : NO"); print("Attachment manifest SHA-256        :",msha); print("Attachment receipt SHA-256         :",sha256_file(p_receipt)); print("="*96)

if __name__=="__main__": main()
