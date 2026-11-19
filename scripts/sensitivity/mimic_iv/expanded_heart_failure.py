#!/usr/bin/env python3
"""Evaluate the prespecified expanded prior-heart-failure sensitivity definition."""

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
EXPECTED_POST_TRANSPLANT_PRIOR_HF_OVERRIDE_N = 111
EXPECTED_CONSECUTIVE_UO_N = 2972
EXPECTED_FIRST_ELIGIBLE_SUBJECTS = 6129


def main() -> None:
    """Construct the outcome-blind expanded prior-heart-failure sensitivity cohort."""
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))
    PROJECT_ROOT = _REPOSITORY_ROOT
    MIMIC_ROOT = Path(os.environ["MIMIC_ROOT"]).resolve()
    COHORT_DIR = PROJECT_ROOT / "outputs" / "01_cohort"
    PHENOTYPE_DIR = PROJECT_ROOT / "outputs" / "02_phenotype"
    OUT = PROJECT_ROOT / "outputs" / "06_sensitivity" / "expanded_heart_failure"
    QC_DIR = PROJECT_ROOT / "quality_control" / "sensitivity" / "expanded_heart_failure"
    OUT.mkdir(parents=True, exist_ok=True); QC_DIR.mkdir(parents=True, exist_ok=True)

    def req(path):
        path = Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        return path
    def qpath(path): return str(Path(path).resolve()).replace("'", "''")
    def assert_unique(df,key,label):
        if df[key].isna().any(): raise AssertionError(f"{label}: missing {key}")
        if df[key].duplicated().any(): raise AssertionError(f"{label}: duplicate {key}")
    prohibited_tokens=("death","mortality","expire","outcome","deathtime","survival","brier","auc","calibration")
    def outcome_like(cols): return [c for c in cols if any(t in c.lower() for t in prohibited_tokens)]
    def assert_preoutcome(df,label):
        bad=outcome_like(df.columns)
        if bad: raise AssertionError(f"{label} contains prohibited outcome/model fields: {bad}")
    def stage_equal(a,b):
        aa=pd.to_numeric(a,errors="coerce").fillna(-999).to_numpy(float); bb=pd.to_numeric(b,errors="coerce").fillna(-999).to_numpy(float)
        return np.isclose(aa,bb,rtol=0,atol=1e-10)
    def rowmax(a,b):
        aa=pd.to_numeric(a,errors="coerce"); bb=pd.to_numeric(b,errors="coerce"); out=pd.concat([aa,bb],axis=1).max(axis=1,skipna=True)
        return out.where(aa.notna()|bb.notna())
    def first_positive(df,positive_col="combined_positive"):
        q=df.loc[df[positive_col].fillna(False).astype(bool)].copy(); q=q.sort_values(["subject_id","intime","stay_id"],kind="mergesort")
        return q.drop_duplicates("subject_id",keep="first").copy()
    def combine_stage(cr,uo,rrt):
        cr=pd.to_numeric(cr,errors="coerce"); uo=pd.to_numeric(uo,errors="coerce"); rrt=pd.Series(rrt,index=cr.index).fillna(False).astype(bool)
        pos=cr.fillna(0).ge(1)|uo.fillna(0).ge(1)|rrt; st=pd.concat([cr.fillna(-1),uo.fillna(-1)],axis=1).max(axis=1); st=st.where(st.ge(1),np.nan).mask(rrt,3.0)
        return pos,st
    def comparison_row(ref,alt,alt_name,ref_name="primary"):
        a=ref[["subject_id","stay_id"]].rename(columns={"stay_id":"ref_stay"}); b=alt[["subject_id","stay_id"]].rename(columns={"stay_id":"alt_stay"}); m=a.merge(b,on="subject_id",how="outer",indicator=True)
        common=m["_merge"].eq("both"); same=common&m.ref_stay.eq(m.alt_stay); rs=set(a.subject_id.astype(int)); als=set(b.subject_id.astype(int)); rst=set(a.ref_stay.astype(int)); ast=set(b.alt_stay.astype(int))
        return {"reference":ref_name,"variant":alt_name,"reference_subjects":len(rs),"variant_subjects":len(als),"subject_added_vs_reference":len(als-rs),"subject_dropped_vs_reference":len(rs-als),"common_subjects":len(rs&als),"common_subject_same_stay":int(same.sum()),"common_subject_replaced_stay":int((common&~same).sum()),"subject_jaccard":len(rs&als)/len(rs|als) if rs|als else 1.0,"stay_jaccard":len(rst&ast)/len(rst|ast) if rst|ast else 1.0}
    def stage_distribution(df,col="combined_stage"):
        s=pd.to_numeric(df[col],errors="coerce"); return {str(int(k)):int(v) for k,v in s.dropna().value_counts().sort_index().items()}

    P_PRIMARY_TARGET=req(PHENOTYPE_DIR/"aki_target_preoutcome.parquet")
    P_PRIMARY_INDEX=req(COHORT_DIR/"primary_cohort_preoutcome.parquet")
    P_PRIMARY_MANIFEST=req(COHORT_DIR/"primary_cohort_manifest.json")
    P_PH=req(COHORT_DIR/"heart_failure_krt_phenotypes.parquet")
    P_LM=req(COHORT_DIR/"landmark_population.parquet")
    P_CRROLL=req(PHENOTYPE_DIR/"creatinine"/"kdigo_creatinine.parquet")
    P_RRT=req(PHENOTYPE_DIR/"rrt"/"active_rrt.parquet")
    P_UOBASE=req(PHENOTYPE_DIR/"urine_output"/"urine_output_event_time_base.parquet")
    P_OVERRIDE=req(PHENOTYPE_DIR/"post_transplant"/"post_transplant_stay_reconstruction.csv")
    manifest=json.loads(P_PRIMARY_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status")!="PASS" or int(manifest.get("primary_n",-1))!=EXPECTED_PRIMARY_N: raise AssertionError("Primary cohort manifest did not pass.")
    lm=pd.read_parquet(P_LM,columns=["subject_id","hadm_id","stay_id","admittime","intime","landmark_time"])
    ph=pd.read_parquet(P_PH,columns=["stay_id","prior_documented_hf","current_hf_sensitivity","hf_wide_sensitivity","prior_expanded_explicit_hf","current_hf_expanded_sensitivity","hf_expanded_sensitivity","chronic_krt_prior_evidence","current_chronic_krt_state_code","chronic_krt_definite","current_eskd_code","current_dialysis_dependence_code","current_dialysis_encounter_code"])
    crroll=pd.read_parquet(P_CRROLL,columns=["stay_id","kdigo_creatinine_stage","creatinine_evaluable"]); rrt=pd.read_parquet(P_RRT,columns=["stay_id","active_rrt_any_by_landmark"]); uobase=pd.read_parquet(P_UOBASE,columns=["stay_id","admission_weight_kg","ibw_kg","height_cm","gender"])
    canon_target=pd.read_parquet(P_PRIMARY_TARGET); canon_index=pd.read_parquet(P_PRIMARY_INDEX); override_locked=pd.read_csv(P_OVERRIDE,usecols=["stay_id","post_transplant_temporal_allow_acute_rrt"])
    for df,label in [(lm,"landmark"),(ph,"HF/KRT phenotype"),(crroll,"rolling creatinine"),(rrt,"RRT"),(uobase,"UO base"),(canon_target,"primary target"),(canon_index,"primary cohort"),(override_locked,"post-transplant override")]: assert_preoutcome(df,label)
    for df,key,label in [(lm,"stay_id","landmark"),(ph,"stay_id","phenotype"),(crroll,"stay_id","creatinine"),(rrt,"stay_id","RRT"),(uobase,"stay_id","UO base"),(canon_target,"stay_id","primary target"),(canon_index,"subject_id","primary cohort subject"),(canon_index,"stay_id","primary cohort stay")]: assert_unique(df,key,label)
    if len(canon_target)!=EXPECTED_PRIMARY_TARGET_N or len(canon_index)!=EXPECTED_PRIMARY_N: raise AssertionError("Primary source sizes changed.")
    for col in ["prior_documented_hf","current_hf_sensitivity","hf_wide_sensitivity","prior_expanded_explicit_hf","current_hf_expanded_sensitivity","hf_expanded_sensitivity","chronic_krt_prior_evidence","current_chronic_krt_state_code","chronic_krt_definite"]: ph[col]=pd.to_numeric(ph[col],errors="coerce").fillna(0).astype("int8")
    rrt["active_rrt_any_by_landmark"]=pd.to_numeric(rrt["active_rrt_any_by_landmark"],errors="coerce").fillna(0).astype("int8")
    base=lm.merge(ph,on="stay_id",how="left",validate="1:1").merge(crroll,on="stay_id",how="left",validate="1:1").merge(rrt,on="stay_id",how="left",validate="1:1").merge(uobase,on="stay_id",how="left",validate="1:1")
    for col in ["admittime","intime","landmark_time"]: base[col]=pd.to_datetime(base[col],errors="raise")
    assert_preoutcome(base,"expanded heart-failure sensitivity base")

    ADM = req(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    DX = req(MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz')
    PROC = req(MIMIC_ROOT / 'hosp' / 'procedures_icd.csv.gz')
    adm = pd.read_csv(ADM, usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'edregtime', 'edouttime'])
    for c in ['admittime', 'dischtime', 'edregtime', 'edouttime']:
        adm[c] = pd.to_datetime(adm[c], errors='coerce')
    assert_preoutcome(adm, 'admissions allowlist')
    con_codes = duckdb.connect(database=':memory:')
    con_codes.execute('PRAGMA threads=4')
    con_codes.register('lm_df', lm)
    con_codes.register('adm_df', adm)
    con_codes.execute('CREATE TEMP TABLE lm AS SELECT * FROM lm_df')
    con_codes.execute('CREATE TEMP TABLE admissions AS SELECT * FROM adm_df')
    con_codes.execute(f"\nCREATE TEMP TABLE dx_hadm AS\nSELECT\n  subject_id,hadm_id,\n  MAX(CASE WHEN (icd_version=9 AND code LIKE '428%') OR\n                (icd_version=10 AND code LIKE 'I50%') THEN 1 ELSE 0 END) AS direct_hf,\n  MAX(CASE WHEN\n        (icd_version=9 AND (\n            code LIKE '428%' OR\n            code IN ('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493')\n        ))\n        OR\n        (icd_version=10 AND (\n            code LIKE 'I50%' OR\n            code IN ('I110','I130','I132')\n        ))\n      THEN 1 ELSE 0 END) AS expanded_hf,\n  MAX(CASE WHEN (icd_version=9 AND code='5856') OR\n                (icd_version=10 AND code='N186') THEN 1 ELSE 0 END) AS kf,\n  MAX(CASE WHEN (icd_version=9 AND code='V4511') OR\n                (icd_version=10 AND code='Z992') THEN 1 ELSE 0 END) AS dialysis_dep,\n  MAX(CASE WHEN (icd_version=9 AND code='V420') OR\n                (icd_version=10 AND code='Z940') THEN 1 ELSE 0 END) AS tx_status,\n  MAX(CASE WHEN icd_version=10 AND code='T8612' THEN 1 ELSE 0 END) AS tx_failure\nFROM (\n  SELECT subject_id,hadm_id,icd_version,\n         UPPER(REPLACE(TRIM(icd_code),'.','')) AS code\n  FROM read_csv_auto('{qpath(DX)}',header=true,sample_size=200000)\n)\nGROUP BY subject_id,hadm_id\n")
    hf_prior = con_codes.execute('\nSELECT\n  l.stay_id,\n  MAX(CASE WHEN d.direct_hf=1 THEN 1 ELSE 0 END) AS prior_hf_recomputed,\n  MAX(CASE WHEN d.expanded_hf=1 THEN 1 ELSE 0 END) AS prior_expanded_hf_recomputed,\n  MAX(CASE WHEN d.direct_hf=1 THEN a.dischtime END) AS latest_prior_hf_dischtime,\n  MAX(CASE WHEN d.expanded_hf=1 THEN a.dischtime END) AS latest_prior_expanded_hf_dischtime,\n  COUNT(DISTINCT CASE WHEN d.direct_hf=1 THEN a.hadm_id END) AS prior_hf_admission_count,\n  COUNT(DISTINCT CASE WHEN d.expanded_hf=1 THEN a.hadm_id END) AS prior_expanded_hf_admission_count\nFROM lm l\nLEFT JOIN admissions a\n  ON a.subject_id=l.subject_id\n AND a.hadm_id<>l.hadm_id\n AND a.dischtime IS NOT NULL\n AND a.dischtime<=l.admittime\nLEFT JOIN dx_hadm d\n  ON d.subject_id=a.subject_id AND d.hadm_id=a.hadm_id\nGROUP BY l.stay_id\n').df()
    hf_current = con_codes.execute('\nSELECT\n  l.stay_id,\n  COALESCE(d.direct_hf,0) AS current_hf_recomputed,\n  COALESCE(d.expanded_hf,0) AS current_expanded_hf_recomputed\nFROM lm l\nLEFT JOIN dx_hadm d\n  ON d.subject_id=l.subject_id AND d.hadm_id=l.hadm_id\n').df()
    hf_timing = hf_prior.merge(hf_current, on='stay_id', validate='1:1')
    hf_timing['prior_hf_recomputed'] = pd.to_numeric(hf_timing.prior_hf_recomputed, errors='coerce').fillna(0).astype('int8')
    hf_timing['current_hf_recomputed'] = pd.to_numeric(hf_timing.current_hf_recomputed, errors='coerce').fillna(0).astype('int8')
    hf_timing['prior_expanded_hf_recomputed'] = pd.to_numeric(hf_timing.prior_expanded_hf_recomputed, errors='coerce').fillna(0).astype('int8')
    hf_timing['current_expanded_hf_recomputed'] = pd.to_numeric(hf_timing.current_expanded_hf_recomputed, errors='coerce').fillna(0).astype('int8')
    hf_timing['hf_wide_recomputed'] = (hf_timing.prior_hf_recomputed.eq(1) | hf_timing.current_hf_recomputed.eq(1)).astype('int8')
    hf_timing['hf_expanded_recomputed'] = (hf_timing.prior_expanded_hf_recomputed.eq(1) | hf_timing.current_expanded_hf_recomputed.eq(1)).astype('int8')
    hf_timing['latest_prior_hf_dischtime'] = pd.to_datetime(hf_timing.latest_prior_hf_dischtime, errors='coerce')
    hf_timing['latest_prior_expanded_hf_dischtime'] = pd.to_datetime(hf_timing.latest_prior_expanded_hf_dischtime, errors='coerce')
    hfcheck = ph[['stay_id', 'prior_documented_hf', 'current_hf_sensitivity', 'hf_wide_sensitivity', 'prior_expanded_explicit_hf', 'current_hf_expanded_sensitivity', 'hf_expanded_sensitivity']].merge(hf_timing[['stay_id', 'prior_hf_recomputed', 'current_hf_recomputed', 'hf_wide_recomputed', 'prior_expanded_hf_recomputed', 'current_expanded_hf_recomputed', 'hf_expanded_recomputed']], on='stay_id', validate='1:1')
    prior_mismatch = int((hfcheck.prior_documented_hf != hfcheck.prior_hf_recomputed).sum())
    current_mismatch = int((hfcheck.current_hf_sensitivity != hfcheck.current_hf_recomputed).sum())
    wide_mismatch = int((hfcheck.hf_wide_sensitivity != hfcheck.hf_wide_recomputed).sum())
    prior_expanded_mismatch = int((hfcheck.prior_expanded_explicit_hf != hfcheck.prior_expanded_hf_recomputed).sum())
    current_expanded_mismatch = int((hfcheck.current_hf_expanded_sensitivity != hfcheck.current_expanded_hf_recomputed).sum())
    expanded_wide_mismatch = int((hfcheck.hf_expanded_sensitivity != hfcheck.hf_expanded_recomputed).sum())
    if any([prior_mismatch, current_mismatch, wide_mismatch, prior_expanded_mismatch, current_expanded_mismatch, expanded_wide_mismatch]):
        raise AssertionError(f'HF reconstruction mismatch: prior={prior_mismatch}, current={current_mismatch}, wide={wide_mismatch}, prior_expanded={prior_expanded_mismatch}, current_expanded={current_expanded_mismatch}, expanded_wide={expanded_wide_mismatch}')
    base = base.merge(hf_timing, on='stay_id', how='left', validate='1:1')
    base['hf_recency_days'] = (base.admittime - base.latest_prior_hf_dischtime).dt.total_seconds() / 86400.0
    base['hf_prior_within_3y'] = (base.prior_documented_hf.eq(1) & base.hf_recency_days.ge(0) & base.hf_recency_days.le(365.25 * 3)).astype('int8')
    base['hf_prior_within_1y'] = (base.prior_documented_hf.eq(1) & base.hf_recency_days.ge(0) & base.hf_recency_days.le(365.25)).astype('int8')
    hf_qc = pd.DataFrame([{'landmark_stays': len(base), 'prior_direct_hf': int(base.prior_documented_hf.sum()), 'current_direct_hf': int(base.current_hf_sensitivity.sum()), 'hf_wide_prior_or_current': int(base.hf_wide_sensitivity.sum()), 'prior_expanded_hf': int(base.prior_expanded_explicit_hf.sum()), 'prior_expanded_only_stays': int((base.prior_expanded_explicit_hf.eq(1) & base.prior_documented_hf.eq(0)).sum()), 'current_expanded_hf': int(base.current_hf_expanded_sensitivity.sum()), 'hf_expanded_prior_or_current': int(base.hf_expanded_sensitivity.sum()), 'prior_recompute_mismatch': prior_mismatch, 'current_recompute_mismatch': current_mismatch, 'wide_recompute_mismatch': wide_mismatch, 'prior_expanded_recompute_mismatch': prior_expanded_mismatch, 'current_expanded_recompute_mismatch': current_expanded_mismatch, 'expanded_wide_recompute_mismatch': expanded_wide_mismatch, 'prior_hf_within_3y_stays': int(base.hf_prior_within_3y.sum()), 'prior_hf_within_1y_stays': int(base.hf_prior_within_1y.sum())}])
    display_table(hf_qc)
    print('HF TIMING/RECENCY RECONSTRUCTION: PASS')
    override_pool = base.loc[(base.hf_wide_sensitivity.eq(1) | base.prior_expanded_explicit_hf.eq(1)) & base.chronic_krt_prior_evidence.fillna(0).eq(1), ['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time', 'prior_documented_hf', 'current_hf_sensitivity', 'hf_wide_sensitivity', 'current_chronic_krt_state_code', 'active_rrt_any_by_landmark']].copy()
    con_codes.register('override_pool_df', override_pool[['subject_id', 'hadm_id', 'stay_id', 'admittime']])
    con_codes.execute('CREATE TEMP TABLE override_pool AS SELECT * FROM override_pool_df')
    hist_dx = con_codes.execute('\nSELECT\n  p.stay_id,\n  MAX(CASE WHEN d.kf=1 OR d.dialysis_dep=1 THEN a.dischtime END) AS latest_prior_krt_state_time,\n  MAX(CASE WHEN d.tx_status=1 THEN a.dischtime END) AS latest_prior_tx_status_time,\n  MAX(CASE WHEN d.tx_failure=1 THEN a.dischtime END) AS latest_prior_tx_failure_time\nFROM override_pool p\nJOIN admissions a\n  ON a.subject_id=p.subject_id\nJOIN dx_hadm d\n  ON d.subject_id=a.subject_id AND d.hadm_id=a.hadm_id\nWHERE a.hadm_id<>p.hadm_id\n  AND a.dischtime IS NOT NULL\n  AND a.dischtime<=p.admittime\nGROUP BY p.stay_id\n').df()
    con_codes.execute(f"\nCREATE TEMP TABLE proc_norm AS\nSELECT subject_id,hadm_id,icd_version,chartdate,\n       UPPER(REPLACE(TRIM(icd_code),'.','')) AS code\nFROM read_csv_auto('{qpath(PROC)}',header=true,sample_size=200000)\n")
    hist_proc = con_codes.execute("\nSELECT\n  p.stay_id,\n  MAX(CAST(pr.chartdate AS TIMESTAMP)) AS latest_prior_tx_proc_time\nFROM override_pool p\nJOIN proc_norm pr ON pr.subject_id=p.subject_id\nJOIN admissions a ON a.subject_id=pr.subject_id AND a.hadm_id=pr.hadm_id\nWHERE pr.hadm_id<>p.hadm_id\n  AND pr.chartdate IS NOT NULL\n  AND CAST(pr.chartdate AS TIMESTAMP)<p.admittime\n  AND a.dischtime IS NOT NULL AND a.dischtime<=p.admittime\n  AND ((pr.icd_version=9 AND pr.code='5569') OR\n       (pr.icd_version=10 AND pr.code LIKE '0TY%'))\nGROUP BY p.stay_id\n").df()
    current_tx = con_codes.execute('\nSELECT\n  p.stay_id,\n  COALESCE(d.tx_status,0) AS current_tx_status,\n  COALESCE(d.tx_failure,0) AS current_tx_failure\nFROM override_pool p\nLEFT JOIN dx_hadm d ON d.subject_id=p.subject_id AND d.hadm_id=p.hadm_id\n').df()
    tx = override_pool.merge(hist_dx, on='stay_id', how='left', validate='1:1').merge(hist_proc, on='stay_id', how='left', validate='1:1').merge(current_tx, on='stay_id', how='left', validate='1:1')
    tx['latest_prior_tx_time'] = pd.concat([pd.to_datetime(tx.latest_prior_tx_status_time, errors='coerce'), pd.to_datetime(tx.latest_prior_tx_proc_time, errors='coerce')], axis=1).max(axis=1)
    tx['tx_after_krt'] = tx.latest_prior_tx_time.notna() & tx.latest_prior_krt_state_time.notna() & (pd.to_datetime(tx.latest_prior_tx_time) > pd.to_datetime(tx.latest_prior_krt_state_time))
    tx['failure_after_tx'] = tx.latest_prior_tx_failure_time.notna() & tx.latest_prior_tx_time.notna() & (pd.to_datetime(tx.latest_prior_tx_failure_time) >= pd.to_datetime(tx.latest_prior_tx_time))
    tx['current_tx_failure'] = pd.to_numeric(tx.current_tx_failure, errors='coerce').fillna(0).astype('int8')
    tx['post_transplant_override'] = tx.tx_after_krt & ~tx.failure_after_tx.fillna(False) & tx.current_chronic_krt_state_code.fillna(0).eq(0) & tx.current_tx_failure.eq(0)
    post_transplant_state = base[['stay_id']].merge(tx[['stay_id', 'post_transplant_override', 'latest_prior_krt_state_time', 'latest_prior_tx_time', 'latest_prior_tx_failure_time', 'current_tx_failure']], on='stay_id', how='left', validate='1:1')
    post_transplant_state['post_transplant_override'] = post_transplant_state.post_transplant_override.fillna(False).astype(bool)
    base = base.merge(post_transplant_state, on='stay_id', how='left', validate='1:1')
    base['post_transplant_override'] = base.post_transplant_override.fillna(False).astype(bool)
    reference_override_set = set(pd.to_numeric(override_locked.loc[override_locked.post_transplant_temporal_allow_acute_rrt.fillna(False).astype(bool), 'stay_id'], errors='raise').astype(int))
    recomputed_primary_set = set(base.loc[base.prior_documented_hf.eq(1) & base.post_transplant_override, 'stay_id'].astype(int))
    if len(reference_override_set) != EXPECTED_POST_TRANSPLANT_PRIOR_HF_OVERRIDE_N:
        raise AssertionError(f'Reference post-transplant override set N={len(reference_override_set)}, expected 111.')
    if recomputed_primary_set != reference_override_set:
        raise AssertionError(f'Generalized post-transplant logic does not reproduce reference prior-HF set. added={len(recomputed_primary_set - reference_override_set)}, dropped={len(reference_override_set - recomputed_primary_set)}')
    base['full_krt_eligible'] = base.chronic_krt_definite.fillna(0).eq(0) | base.post_transplant_override
    base['prior_only_krt_eligible_sensitivity'] = base.chronic_krt_prior_evidence.fillna(0).eq(0) | base.post_transplant_override
    primary_target_mask = base.prior_documented_hf.eq(1) & base.full_krt_eligible
    primary_target_stays = set(base.loc[primary_target_mask, 'stay_id'].astype(int))
    canon_target_stays = set(canon_target.stay_id.astype(int))
    if primary_target_stays != canon_target_stays or len(primary_target_stays) != EXPECTED_PRIMARY_TARGET_N:
        raise AssertionError(
            "Failed to reproduce the reference target cohort with prior HF after KRT adjudication "
            f"(n={len(primary_target_stays)}, added={len(primary_target_stays - canon_target_stays)}, "
            f"dropped={len(canon_target_stays - primary_target_stays)})."
        )
    krt_qc = pd.DataFrame([{'hf_wide_prior_krt_pool_stays': len(override_pool), 'generalized_post_transplant_state_override_all_hfwide': int(base.post_transplant_override.sum()), 'reference_prior_hf_post_transplant_override': len(reference_override_set), 'recomputed_prior_hf_post_transplant_override': len(recomputed_primary_set), 'primary_krt_eligible_target': int(primary_target_mask.sum()), 'prior_only_krt_eligible_priorHF_stays': int((base.prior_documented_hf.eq(1) & base.prior_only_krt_eligible_sensitivity).sum())}])
    display_table(krt_qc)
    print('POST-TRANSPLANT / KRT ELIGIBILITY RECONSTRUCTION: PASS')
    pool = base.loc[(base.hf_wide_sensitivity.eq(1) | base.prior_expanded_explicit_hf.eq(1)) & base.prior_only_krt_eligible_sensitivity,].copy()
    assert_unique(pool, 'stay_id', 'sensitivity union pool')
    assert_preoutcome(pool, 'sensitivity union pool')
    OE = req(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    con_rel = duckdb.connect(database=':memory:')
    con_rel.execute('PRAGMA threads=4')
    con_rel.register('pool_df', pool[['stay_id', 'intime', 'landmark_time']])
    con_rel.execute('CREATE TEMP TABLE pool AS SELECT * FROM pool_df')
    rel = con_rel.execute(f"\nSELECT\n  p.stay_id,\n  MAX(CASE WHEN oe.itemid IN (226557,226558) THEN 1 ELSE 0 END) AS has_ureteral_stent_by_landmark,\n  MAX(CASE WHEN oe.itemid IN (227488,227489) THEN 1 ELSE 0 END) AS has_gu_irrigation_by_landmark\nFROM pool p\nJOIN read_csv_auto('{qpath(OE)}',header=true,sample_size=200000) oe USING(stay_id)\nWHERE oe.itemid IN (226557,226558,227488,227489)\n  AND oe.charttime>=p.intime AND oe.charttime<=p.landmark_time\n  AND oe.storetime IS NOT NULL AND oe.storetime<=p.landmark_time\nGROUP BY p.stay_id\n").df()
    rel_full = pool[['stay_id']].merge(rel, on='stay_id', how='left', validate='1:1')
    for c in ['has_ureteral_stent_by_landmark', 'has_gu_irrigation_by_landmark']:
        rel_full[c] = pd.to_numeric(rel_full[c], errors='coerce').fillna(0).astype('int8')
    rel_full['uo_unreliable_recomputed'] = (rel_full.has_ureteral_stent_by_landmark.eq(1) | rel_full.has_gu_irrigation_by_landmark.eq(1)).astype('int8')
    pool = pool.merge(rel_full, on='stay_id', how='left', validate='1:1')
    relcheck = canon_target[['stay_id', 'uo_unreliable_by_landmark']].merge(pool[['stay_id', 'uo_unreliable_recomputed']], on='stay_id', validate='1:1')
    rel_mismatch = int((pd.to_numeric(relcheck.uo_unreliable_by_landmark, errors='coerce').fillna(0).astype(int) != pd.to_numeric(relcheck.uo_unreliable_recomputed, errors='coerce').fillna(0).astype(int)).sum())
    if rel_mismatch:
        raise AssertionError(f'UO reliability reconstruction mismatch on reference target: {rel_mismatch}')
    pool_qc = pd.DataFrame([{'sensitivity_union_pool_stays': len(pool), 'priorHF_fullKRT_target_stays': int((pool.prior_documented_hf.eq(1) & pool.full_krt_eligible).sum()), 'HFwide_fullKRT_target_stays': int((pool.hf_wide_sensitivity.eq(1) & pool.full_krt_eligible).sum()), 'HFexpandedPrior_fullKRT_target_stays': int((pool.prior_expanded_explicit_hf.eq(1) & pool.full_krt_eligible).sum()), 'HFexpandedPrior_only_added_target_stays': int((pool.prior_expanded_explicit_hf.eq(1) & pool.prior_documented_hf.eq(0) & pool.full_krt_eligible).sum()), 'uo_unreliable_stays_union': int(pool.uo_unreliable_recomputed.sum()), 'uo_reliability_mismatch_vs_primary': rel_mismatch}])
    display_table(pool_qc)
    print('SENSITIVITY UNION POOL / UO RELIABILITY: PASS')
    LAB = req(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    TR = req(MIMIC_ROOT / 'hosp' / 'transfers.csv.gz')
    tr = pd.read_csv(TR, usecols=['subject_id', 'hadm_id', 'eventtype', 'careunit', 'intime', 'outtime'])
    for c in ['intime', 'outtime']:
        tr[c] = pd.to_datetime(tr[c], errors='coerce')
    assert_preoutcome(tr, 'transfers allowlist')
    conc = duckdb.connect(database=':memory:')
    conc.execute('PRAGMA threads=4')
    conc.register('pool_df', pool[['subject_id', 'hadm_id', 'stay_id', 'admittime', 'intime', 'landmark_time']])
    conc.register('adm_df', adm)
    conc.register('tr_df', tr)
    conc.execute('CREATE TEMP TABLE target_lm AS SELECT * FROM pool_df')
    conc.execute('CREATE TEMP TABLE admissions AS SELECT * FROM adm_df')
    conc.execute('CREATE TEMP TABLE transfers_all AS SELECT * FROM tr_df')
    conc.execute(f"\nCREATE TEMP TABLE creat_all AS\nSELECT subject_id,hadm_id,charttime,storetime,valuenum AS creatinine\nFROM read_csv_auto('{qpath(LAB)}',header=true,sample_size=200000)\nWHERE itemid=50912\n  AND valuenum>0 AND valuenum<=150\n  AND charttime IS NOT NULL AND storetime IS NOT NULL\n")
    conc.execute("\nCREATE TEMP TABLE valid_physical_transfers AS\nSELECT subject_id,hadm_id,LOWER(COALESCE(eventtype,'')) AS eventtype,\n       careunit,intime,outtime\nFROM transfers_all\nWHERE intime IS NOT NULL AND outtime IS NOT NULL AND outtime>intime\n  AND careunit IS NOT NULL\n  AND LOWER(COALESCE(eventtype,'')) IN ('ed','admit','transfer')\n")
    conc.execute("\nCREATE TEMP TABLE valid_ed_transfers AS\nSELECT * FROM valid_physical_transfers\nWHERE eventtype='ed' OR LOWER(careunit) LIKE '%emergency%'\n")
    conc.execute("\nCREATE TEMP TABLE hist_same_time AS\nSELECT\n  t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime,\n  MAX(c.storetime) AS storetime,\n  AVG(c.creatinine) AS creatinine,\n  MAX(CASE WHEN c.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS has_labevent_hadm_id\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.admittime-INTERVAL '365 days'\n  AND c.charttime<t.admittime-INTERVAL '7 days'\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.subject_id,t.admittime,t.landmark_time,c.charttime\n")
    conc.execute('\nCREATE TEMP TABLE hist_strict AS\nSELECT h.*\nFROM hist_same_time h\nWHERE has_labevent_hadm_id=0\n  AND NOT EXISTS (\n    SELECT 1 FROM admissions a\n    WHERE a.subject_id=h.subject_id\n      AND a.admittime IS NOT NULL AND a.dischtime IS NOT NULL\n      AND h.charttime>=a.admittime AND h.charttime<=a.dischtime\n  )\n  AND NOT EXISTS (\n    SELECT 1 FROM admissions a\n    WHERE a.subject_id=h.subject_id\n      AND a.edregtime IS NOT NULL AND a.edouttime IS NOT NULL\n      AND a.edouttime>a.edregtime\n      AND h.charttime>=a.edregtime AND h.charttime<=a.edouttime\n  )\n  AND NOT EXISTS (\n    SELECT 1 FROM valid_ed_transfers e\n    WHERE e.subject_id=h.subject_id\n      AND h.charttime>=e.intime AND h.charttime<=e.outtime\n  )\n  AND NOT EXISTS (\n    SELECT 1 FROM valid_physical_transfers tr\n    WHERE tr.subject_id=h.subject_id\n      AND h.charttime>=tr.intime AND h.charttime<=tr.outtime\n  )\n')
    strict_baseline = conc.execute('\nSELECT\n  t.stay_id,\n  COUNT(h.charttime) AS hist_strict_n,\n  MEDIAN(h.creatinine) AS hist_strict_median,\n  AVG(h.creatinine) AS hist_strict_mean_sensitivity,\n  ARG_MAX(h.creatinine,h.charttime) AS hist_strict_most_recent_sensitivity,\n  MIN(h.charttime) AS hist_strict_first_time,\n  MAX(h.charttime) AS hist_strict_last_time\nFROM target_lm t\nLEFT JOIN hist_strict h USING(stay_id)\nGROUP BY t.stay_id\n').df()
    assert_unique(strict_baseline, 'stay_id', 'strict historical baseline')
    conc.execute("\nCREATE TEMP TABLE current_same_time AS\nSELECT\n  t.stay_id,t.admittime,t.intime,t.landmark_time,c.charttime,\n  AVG(c.creatinine) AS creatinine\nFROM target_lm t\nJOIN creat_all c ON c.subject_id=t.subject_id\nWHERE c.charttime>=t.intime-INTERVAL '7 days'\n  AND c.charttime<=t.landmark_time\n  AND c.storetime<=t.landmark_time\nGROUP BY t.stay_id,t.admittime,t.intime,t.landmark_time,c.charttime\n")
    conc.execute("\nCREATE TEMP TABLE cr_refs AS\nSELECT *,\n  MIN(creatinine) OVER(\n    PARTITION BY stay_id ORDER BY charttime\n    RANGE BETWEEN INTERVAL '48 hours' PRECEDING\n    AND INTERVAL '1 microsecond' PRECEDING\n  ) AS low48\nFROM current_same_time\n")
    conc.register('strict_baseline_df', strict_baseline)
    conc.execute('CREATE TEMP TABLE strict_baseline_tbl AS SELECT * FROM strict_baseline_df')
    events = conc.execute('\nSELECT\n  r.stay_id,r.charttime,r.creatinine,r.low48,\n  b.hist_strict_median,b.hist_strict_mean_sensitivity,\n  b.hist_strict_most_recent_sensitivity\nFROM cr_refs r\nJOIN target_lm t USING(stay_id)\nLEFT JOIN strict_baseline_tbl b USING(stay_id)\nWHERE r.charttime>=t.intime AND r.charttime<=t.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()

    def historical_stage(df, baseline_col):
        cr = pd.to_numeric(df.creatinine, errors='coerce')
        b = pd.to_numeric(df[baseline_col], errors='coerce')
        low48 = pd.to_numeric(df.low48, errors='coerce')
        has = b.notna()
        ratio15 = has & cr.ge(1.5 * b)
        ratio20 = has & cr.ge(2.0 * b)
        ratio30 = has & cr.ge(3.0 * b)
        delta03 = low48.notna() & cr.ge(low48 + 0.3)
        out = np.select([ratio30, cr.ge(4.0) & (ratio15 | delta03), ratio20, ratio15], [3.0, 3.0, 2.0, 1.0], default=0.0)
        out = pd.Series(out, index=df.index, dtype='float64')
        out.loc[~has] = np.nan
        return out
    events['stage_hist_strict_median'] = historical_stage(events, 'hist_strict_median')
    hist_stages = events.groupby('stay_id', as_index=False)[['stage_hist_strict_median']].max()
    crtab = pool[['stay_id', 'kdigo_creatinine_stage', 'creatinine_evaluable']].merge(strict_baseline, on='stay_id', how='left', validate='1:1').merge(hist_stages, on='stay_id', how='left', validate='1:1')
    crtab = crtab.rename(columns={'kdigo_creatinine_stage': 'cr_stage_rolling_only', 'creatinine_evaluable': 'cr_evaluable_rolling'})
    crtab['primary_creatinine_stage'] = rowmax(crtab.cr_stage_rolling_only, crtab.stage_hist_strict_median)
    event_stays = set(events.stay_id.astype(int))
    hist_eval = crtab.hist_strict_median.notna() & crtab.stay_id.astype(int).isin(event_stays)
    crtab['primary_creatinine_evaluable'] = (pd.to_numeric(crtab.cr_evaluable_rolling, errors='coerce').fillna(0).eq(1) | hist_eval).astype('int8')
    crtab['history_only_positive'] = (pd.to_numeric(crtab.cr_stage_rolling_only, errors='coerce').fillna(0).lt(1) & pd.to_numeric(crtab.primary_creatinine_stage, errors='coerce').fillna(0).ge(1)).astype('int8')
    cur = conc.execute('\nSELECT\n  stay_id,\n  ARG_MIN(creatinine,charttime) FILTER(\n    WHERE charttime>=admittime AND charttime<=landmark_time\n  ) AS admission_first_cr,\n  MIN(creatinine) FILTER(\n    WHERE charttime>=admittime AND charttime<=landmark_time\n  ) AS admission_min_cr\nFROM current_same_time\nGROUP BY stay_id\n').df()
    refs = pool[['stay_id']].merge(strict_baseline[['stay_id', 'hist_strict_median']], on='stay_id', how='left', validate='1:1').merge(cur, on='stay_id', how='left', validate='1:1')
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
    hier_events = conc.execute('\nSELECT r.*,x.hier_ref_cr\nFROM cr_refs r\nJOIN target_lm t USING(stay_id)\nLEFT JOIN refs_tbl x USING(stay_id)\nWHERE r.charttime>=t.intime AND r.charttime<=t.landmark_time\nORDER BY r.stay_id,r.charttime\n').df()
    crv = pd.to_numeric(hier_events.creatinine, errors='coerce')
    b = pd.to_numeric(hier_events.hier_ref_cr, errors='coerce')
    low48 = pd.to_numeric(hier_events.low48, errors='coerce')
    ratio15 = b.notna() & crv.ge(1.5 * b)
    ratio20 = b.notna() & crv.ge(2.0 * b)
    ratio30 = b.notna() & crv.ge(3.0 * b)
    delta = low48.notna() & crv.ge(low48 + 0.3)
    hier_events['hier_stage_event'] = np.select([ratio30, crv.ge(4.0) & (ratio15 | delta), ratio20, ratio15 | delta], [3.0, 3.0, 2.0, 1.0], default=0.0)
    hier_stage = hier_events.groupby('stay_id', as_index=False).hier_stage_event.max().rename(columns={'hier_stage_event': 'cr_stage_hierarchical_sensitivity'})
    crtab = crtab.merge(hier_stage, on='stay_id', how='left', validate='1:1').merge(refs[['stay_id', 'hier_ref_source', 'hier_ref_cr']], on='stay_id', how='left', validate='1:1')
    pool = pool.merge(crtab, on='stay_id', how='left', validate='1:1')
    assert_preoutcome(pool, 'pool after creatinine reconstruction')
    cr_qc = pd.DataFrame([{'union_pool_stays': len(pool), 'strict_historical_baseline_available': int(pool.hist_strict_median.notna().sum()), 'strict_historical_single_measurement': int(pd.to_numeric(pool.hist_strict_n, errors='coerce').fillna(0).eq(1).sum()), 'rolling_positive': int(pd.to_numeric(pool.cr_stage_rolling_only, errors='coerce').fillna(0).ge(1).sum()), 'primary_creatinine_positive': int(pd.to_numeric(pool.primary_creatinine_stage, errors='coerce').fillna(0).ge(1).sum()), 'history_only_positive': int(pool.history_only_positive.sum()), 'hierarchy_positive': int(pd.to_numeric(pool.cr_stage_hierarchical_sensitivity, errors='coerce').fillna(0).ge(1).sum())}])
    display_table(cr_qc)
    print('CREATININE RECONSTRUCTION COMPLETE')
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558)
    COMPARTMENT_CASE = "\nCASE\n WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n WHEN oe.itemid=226557 THEN 'URETERAL_STENT_R'\n WHEN oe.itemid=226558 THEN 'URETERAL_STENT_L'\n ELSE NULL\nEND\n"
    conuo = duckdb.connect(database=':memory:')
    conuo.execute('PRAGMA threads=4')
    conuo.register('target_df', pool[['stay_id', 'intime', 'landmark_time']])
    conuo.execute('CREATE TEMP TABLE target AS SELECT * FROM target_df')
    conuo.execute(f"\nCREATE TEMP TABLE oe AS\nSELECT stay_id,itemid,charttime,storetime,value\nFROM read_csv_auto('{qpath(OE)}',header=true,sample_size=200000)\n")
    conuo.execute(f'\nCREATE TEMP TABLE uo_events AS\nSELECT\n  t.stay_id,\n  {COMPARTMENT_CASE} AS compartment,\n  oe.charttime,\n  MAX(oe.storetime) AS storetime,\n  SUM(oe.value) AS volume_ml\nFROM target t\nJOIN oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY}\n  AND oe.value IS NOT NULL\n  AND oe.value BETWEEN 0 AND 5000\n  AND oe.charttime>=t.intime\n  AND oe.charttime<=t.landmark_time\n  AND oe.storetime IS NOT NULL\n  AND oe.storetime<=t.landmark_time\nGROUP BY t.stay_id,compartment,oe.charttime\nHAVING compartment IS NOT NULL\n')
    conuo.execute('\nCREATE TEMP TABLE raw_intervals AS\nSELECT\n  stay_id,compartment,\n  LAG(charttime) OVER(PARTITION BY stay_id,compartment ORDER BY charttime) AS interval_start,\n  charttime AS interval_end,\n  volume_ml\nFROM uo_events\n')
    intervals = conuo.execute("\nSELECT\n  r.stay_id,r.compartment,r.interval_start,r.interval_end,r.volume_ml,\n  r.volume_ml/(DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0) AS rate_ml_hr\nFROM raw_intervals r\nJOIN target t USING(stay_id)\nWHERE r.interval_start IS NOT NULL\n  AND r.interval_end>r.interval_start\n  AND r.interval_start>=t.intime\n  AND r.interval_end<=t.landmark_time\n  AND DATE_DIFF('second',r.interval_start,r.interval_end)/3600.0<=6.0+1e-12\nORDER BY r.stay_id,r.interval_start\n").df()
    hour_rows = []
    for r in pool[['stay_id', 'intime', 'landmark_time']].itertuples(index=False):
        cur = pd.Timestamp(r.intime).floor('h')
        lm_ = pd.Timestamp(r.landmark_time)
        while cur + pd.Timedelta(hours=1) <= lm_:
            hour_rows.append((int(r.stay_id), cur, cur + pd.Timedelta(hours=1)))
            cur += pd.Timedelta(hours=1)
    hours = pd.DataFrame(hour_rows, columns=['stay_id', 'hour_start', 'hour_end'])
    conh = duckdb.connect(database=':memory:')
    conh.execute('PRAGMA threads=4')
    conh.register('hours_df', hours)
    conh.register('intervals_df', intervals)
    conh.execute('CREATE TEMP TABLE hours AS SELECT * FROM hours_df')
    conh.execute('CREATE TEMP TABLE intervals AS SELECT * FROM intervals_df')
    hourly = conh.execute("\nWITH ov AS (\n SELECT\n   h.stay_id,h.hour_start,h.hour_end,\n   i.rate_ml_hr,\n   GREATEST(i.interval_start,h.hour_start) AS ov_start,\n   LEAST(i.interval_end,h.hour_end) AS ov_end\n FROM hours h\n JOIN intervals i\n   ON i.stay_id=h.stay_id\n  AND i.interval_end>h.hour_start\n  AND i.interval_start<h.hour_end\n WHERE LEAST(i.interval_end,h.hour_end)>GREATEST(i.interval_start,h.hour_start)\n),\nagg AS (\n SELECT\n   stay_id,hour_start,hour_end,\n   SUM(rate_ml_hr*DATE_DIFF('second',ov_start,ov_end)/3600.0) AS numerator_ml,\n   DATE_DIFF('second',MIN(ov_start),MAX(ov_end))/3600.0 AS coverage_h\n FROM ov\n GROUP BY stay_id,hour_start,hour_end\n)\nSELECT\n  h.stay_id,h.hour_start,h.hour_end,\n  a.numerator_ml,a.coverage_h,\n  CASE WHEN a.coverage_h>0.5 THEN a.numerator_ml/a.coverage_h END AS hourly_ml_hr\nFROM hours h\nLEFT JOIN agg a USING(stay_id,hour_start,hour_end)\nORDER BY h.stay_id,h.hour_end\n").df()
    hourly = hourly.merge(pool[['stay_id', 'intime', 'ibw_kg', 'admission_weight_kg', 'uo_unreliable_recomputed']], on='stay_id', how='left', validate='m:1')
    hourly['uo_ibw'] = pd.to_numeric(hourly.hourly_ml_hr, errors='coerce') / pd.to_numeric(hourly.ibw_kg, errors='coerce')
    hourly['uo_abw'] = pd.to_numeric(hourly.hourly_ml_hr, errors='coerce') / pd.to_numeric(hourly.admission_weight_kg, errors='coerce')
    bad = hourly.uo_unreliable_recomputed.fillna(0).eq(1)
    hourly.loc[bad, ['uo_ibw', 'uo_abw']] = np.nan
    valid_abw = pd.to_numeric(hourly.admission_weight_kg, errors='coerce').ge(25) & pd.to_numeric(hourly.admission_weight_kg, errors='coerce').le(300)
    hourly['uo_abw_25_300'] = hourly.uo_abw.where(valid_abw)

    def max_consecutive_valid(vals):
        best = cur = 0
        for ok in np.isfinite(vals):
            if ok:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def stage_and_paths(g, value_col, mode='mean'):
        g = g.sort_values('hour_end').reset_index(drop=True)
        vals = pd.to_numeric(g[value_col], errors='coerce').to_numpy(float)
        intime = pd.Timestamp(g.intime.iloc[0])
        best = 0
        any_s1_6 = False
        any_s2_12 = False
        any_anuria12 = False
        any_oliguria24 = False
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
                stat = float(np.mean(x)) if mode == 'mean' else float(np.max(x))
                if kind == 's1' and stat < 0.5:
                    best = max(best, 1)
                    any_s1_6 = True
                elif kind == 's2' and stat < 0.5:
                    best = max(best, 2)
                    any_s2_12 = True
                elif kind == 'anuria' and stat == 0:
                    best = max(best, 3)
                    any_anuria12 = True
                elif kind == 's3' and stat < 0.3:
                    best = max(best, 3)
                    any_oliguria24 = True
        return {'stage': float(best) if best >= 1 else np.nan, 'valid_hours': int(np.isfinite(vals).sum()), 'max_consecutive_valid_hours': max_consecutive_valid(vals), 'path_s1_6h_lt05': int(any_s1_6), 'path_s2_12h_lt05': int(any_s2_12), 'path_stage3_anuria_12h': int(any_anuria12), 'path_stage3_oliguria_24h': int(any_oliguria24)}
    rows = []
    for stay_id, g in hourly.groupby('stay_id', sort=False):
        mi = stage_and_paths(g, 'uo_ibw', 'mean')
        ci = stage_and_paths(g, 'uo_ibw', 'cons')
        ma = stage_and_paths(g, 'uo_abw', 'mean')
        mab = stage_and_paths(g, 'uo_abw_25_300', 'mean')
        rows.append({'stay_id': int(stay_id), 'uo_stage_mean_ibw_elapsed': mi['stage'], 'uo_stage_cons_ibw_elapsed': ci['stage'], 'uo_stage_mean_abw_elapsed': ma['stage'], 'uo_stage_mean_abw_25_300_elapsed': mab['stage'], 'uo_valid_hours_ibw': mi['valid_hours'], 'uo_max_consecutive_valid_hours_ibw': mi['max_consecutive_valid_hours'], 'uo_valid_hours_abw': ma['valid_hours'], 'uo_max_consecutive_valid_hours_abw': ma['max_consecutive_valid_hours'], 'uo_path_s1_6h_lt05': mi['path_s1_6h_lt05'], 'uo_path_s2_12h_lt05': mi['path_s2_12h_lt05'], 'uo_path_stage3_anuria_12h': mi['path_stage3_anuria_12h'], 'uo_path_stage3_oliguria_24h': mi['path_stage3_oliguria_24h']})
    uostage = pd.DataFrame(rows)
    assert_unique(uostage, 'stay_id', 'urine-output stage table')
    pool = pool.merge(uostage, on='stay_id', how='left', validate='1:1')
    pool['urine_output_ascertainable_recomputed'] = pd.to_numeric(pool.ibw_kg, errors='coerce').gt(0) & pd.to_numeric(pool.uo_max_consecutive_valid_hours_ibw, errors='coerce').fillna(0).ge(6)
    assert_preoutcome(pool, 'pool after UO reconstruction')
    uo_qc = pd.DataFrame([{'union_pool_stays': len(pool), 'IBW_available': int(pd.to_numeric(pool.ibw_kg, errors='coerce').gt(0).sum()), 'ABW_available': int(pd.to_numeric(pool.admission_weight_kg, errors='coerce').gt(0).sum()), 'mean-UO_IBW_positive_stays': int(pd.to_numeric(pool.uo_stage_mean_ibw_elapsed, errors='coerce').fillna(0).ge(1).sum()), 'consecutive-UO_IBW_positive_stays': int(pd.to_numeric(pool.uo_stage_cons_ibw_elapsed, errors='coerce').fillna(0).ge(1).sum()), 'UO_stage3_anuria12_path_stays': int(pool.uo_path_stage3_anuria_12h.sum()), 'UO_stage3_oliguria24_path_stays': int(pool.uo_path_stage3_oliguria_24h.sum())}])
    display_table(uo_qc)
    print('ELAPSED-DURATION URINE-OUTPUT RECONSTRUCTION COMPLETE')
    prim_target = pool.loc[pool.prior_documented_hf.eq(1) & pool.full_krt_eligible].copy()
    if len(prim_target) != EXPECTED_PRIMARY_TARGET_N:
        raise AssertionError(f'Recomputed primary target N={len(prim_target)}, expected 10083.')
    check = canon_target[['stay_id', 'creatinine_stage', 'creatinine_evaluable', 'urine_output_stage', 'urine_output_stage_consecutive_sensitivity', 'urine_output_max_consecutive_valid_hours', 'urine_output_ascertainable', 'uo_unreliable_by_landmark', 'active_rrt_any_by_landmark', 'aki_positive', 'kdigo_stage']].merge(prim_target[['stay_id', 'primary_creatinine_stage', 'primary_creatinine_evaluable', 'uo_stage_mean_ibw_elapsed', 'uo_stage_cons_ibw_elapsed', 'uo_max_consecutive_valid_hours_ibw', 'urine_output_ascertainable_recomputed', 'uo_unreliable_recomputed', 'active_rrt_any_by_landmark']], on='stay_id', validate='1:1', suffixes=('_canon', '_re'))
    prim_target['combined_positive'], prim_target['combined_stage'] = combine_stage(prim_target.primary_creatinine_stage, prim_target.uo_stage_mean_ibw_elapsed, prim_target.active_rrt_any_by_landmark.eq(1))
    prim_target['combined_positive_consecutive_uo'], prim_target['combined_stage_consecutive_uo'] = combine_stage(prim_target.primary_creatinine_stage, prim_target.uo_stage_cons_ibw_elapsed, prim_target.active_rrt_any_by_landmark.eq(1))
    check = check.merge(prim_target[['stay_id', 'combined_positive', 'combined_stage']], on='stay_id', validate='1:1')
    repro_metrics = {'cr_stage_mismatch': int((~stage_equal(check.creatinine_stage, check.primary_creatinine_stage)).sum()), 'cr_evaluable_mismatch': int((pd.to_numeric(check.creatinine_evaluable, errors='coerce').fillna(0).astype(int) != pd.to_numeric(check.primary_creatinine_evaluable, errors='coerce').fillna(0).astype(int)).sum()), 'uo_mean_stage_mismatch': int((~stage_equal(check.urine_output_stage, check.uo_stage_mean_ibw_elapsed)).sum()), 'uo_cons_stage_mismatch': int((~stage_equal(check.urine_output_stage_consecutive_sensitivity, check.uo_stage_cons_ibw_elapsed)).sum()), 'uo_max_consecutive_mismatch': int((pd.to_numeric(check.urine_output_max_consecutive_valid_hours, errors='coerce').fillna(-1).astype(int) != pd.to_numeric(check.uo_max_consecutive_valid_hours_ibw, errors='coerce').fillna(-1).astype(int)).sum()), 'uo_ascertainable_mismatch': int((check.urine_output_ascertainable.fillna(False).astype(bool) != check.urine_output_ascertainable_recomputed.fillna(False).astype(bool)).sum()), 'uo_unreliable_mismatch': int((pd.to_numeric(check.uo_unreliable_by_landmark, errors='coerce').fillna(0).astype(int) != pd.to_numeric(check.uo_unreliable_recomputed, errors='coerce').fillna(0).astype(int)).sum()), 'rrt_mismatch': int((pd.to_numeric(check.active_rrt_any_by_landmark_canon, errors='coerce').fillna(0).astype(int) != pd.to_numeric(check.active_rrt_any_by_landmark_re, errors='coerce').fillna(0).astype(int)).sum()), 'combined_positive_mismatch': int((check.aki_positive.fillna(False).astype(bool) != check.combined_positive.fillna(False).astype(bool)).sum()), 'combined_stage_mismatch': int((~stage_equal(check.kdigo_stage, check.combined_stage)).sum())}
    if any((v != 0 for v in repro_metrics.values())):
        raise AssertionError(
            f"Failed to reproduce the reference phenotype measures for the primary target cohort ({repro_metrics})."
        )
    primary_sel = first_positive(prim_target, 'combined_positive')
    if len(primary_sel) != EXPECTED_PRIMARY_N:
        raise AssertionError(f'Recomputed primary N={len(primary_sel)}, expected {EXPECTED_PRIMARY_N}.')
    consecutive_uo_sel = first_positive(prim_target, 'combined_positive_consecutive_uo')
    if len(consecutive_uo_sel) != EXPECTED_CONSECUTIVE_UO_N:
        raise AssertionError(f'Recomputed consecutive-UO N={len(consecutive_uo_sel)}, expected 2972.')
    repro = pd.DataFrame([{'primary_target_n': len(prim_target), 'primary_selected_n': len(primary_sel), 'uo_consecutive_selected_n': len(consecutive_uo_sel), **repro_metrics}])
    display_table(repro)
    print('EXACT  PRIMARY + CONSECUTIVE UO REPRODUCTION: PASS')
    pool['aki_positive_recomputed'], pool['kdigo_stage_recomputed'] = combine_stage(pool.primary_creatinine_stage, pool.uo_stage_mean_ibw_elapsed, pool.active_rrt_any_by_landmark.eq(1))
    pool['combined_positive_rolling_only'], pool['combined_stage_rolling_only'] = combine_stage(pool.cr_stage_rolling_only, pool.uo_stage_mean_ibw_elapsed, pool.active_rrt_any_by_landmark.eq(1))
    pool['combined_positive_hierarchy'], pool['combined_stage_hierarchy'] = combine_stage(pool.cr_stage_hierarchical_sensitivity, pool.uo_stage_mean_ibw_elapsed, pool.active_rrt_any_by_landmark.eq(1))
    pool['combined_positive_abw'], pool['combined_stage_abw'] = combine_stage(pool.primary_creatinine_stage, pool.uo_stage_mean_abw_elapsed, pool.active_rrt_any_by_landmark.eq(1))
    pool['combined_positive_abw_bounded'], pool['combined_stage_abw_bounded'] = combine_stage(pool.primary_creatinine_stage, pool.uo_stage_mean_abw_25_300_elapsed, pool.active_rrt_any_by_landmark.eq(1))
    mask_primary = pool.prior_documented_hf.eq(1) & pool.full_krt_eligible
    mask_hfwide = pool.hf_wide_sensitivity.eq(1) & pool.full_krt_eligible
    mask_hfexpanded_prior = pool.prior_expanded_explicit_hf.eq(1) & pool.full_krt_eligible
    mask_prioronly = pool.prior_documented_hf.eq(1) & pool.prior_only_krt_eligible_sensitivity
    mask_3y = mask_primary & pool.hf_prior_within_3y.eq(1)
    mask_1y = mask_primary & pool.hf_prior_within_1y.eq(1)

    def select_variant(mask, positive_col, stage_col, name):
        q = pool.loc[mask].copy()
        q['variant'] = name
        q['combined_positive_variant'] = q[positive_col].fillna(False).astype(bool)
        q['combined_stage_variant'] = pd.to_numeric(q[stage_col], errors='coerce')
        sel = first_positive(q, 'combined_positive_variant')
        assert_unique(sel, 'subject_id', name + ' subject')
        assert_unique(sel, 'stay_id', name + ' stay')
        assert_preoutcome(sel, name)
        return sel
    variants = {}
    variants['primary'] = select_variant(mask_primary, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'primary')
    variants['uo_consecutive_ibw_elapsed'] = consecutive_uo_sel.assign(variant='uo_consecutive_ibw_elapsed', combined_positive_variant=lambda x: x.combined_positive_consecutive_uo.astype(bool), combined_stage_variant=lambda x: pd.to_numeric(x.combined_stage_consecutive_uo, errors='coerce'))
    variants['hf_wide'] = select_variant(mask_hfwide, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'hf_wide')
    variants['expanded_prior_heart_failure'] = select_variant(mask_hfexpanded_prior, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'expanded_prior_heart_failure')
    variants['prior_only_krt'] = select_variant(mask_prioronly, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'prior_only_krt')
    variants['rolling_only_creatinine'] = select_variant(mask_primary, 'combined_positive_rolling_only', 'combined_stage_rolling_only', 'rolling_only_creatinine')
    variants['hierarchical_creatinine'] = select_variant(mask_primary, 'combined_positive_hierarchy', 'combined_stage_hierarchy', 'hierarchical_creatinine')
    variants['hf_recency_3y'] = select_variant(mask_3y, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'hf_recency_3y')
    variants['hf_recency_1y'] = select_variant(mask_1y, 'aki_positive_recomputed', 'kdigo_stage_recomputed', 'hf_recency_1y')
    variants['uo_mean_abw_elapsed'] = select_variant(mask_primary, 'combined_positive_abw', 'combined_stage_abw', 'uo_mean_abw_elapsed')
    variants['uo_mean_abw_25_300_elapsed'] = select_variant(mask_primary, 'combined_positive_abw_bounded', 'combined_stage_abw_bounded', 'uo_mean_abw_25_300_elapsed')
    fixed_all = pool.loc[mask_primary].copy().sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort').drop_duplicates('subject_id', keep='first')
    if len(fixed_all) != EXPECTED_FIRST_ELIGIBLE_SUBJECTS:
        raise AssertionError(f'Fixed-first eligible subjects={len(fixed_all)}, expected {EXPECTED_FIRST_ELIGIBLE_SUBJECTS}.')
    fixed_all['fixed_first_aki_positive'] = fixed_all.aki_positive_recomputed.fillna(False).astype(bool)
    fixed_all['fixed_first_kdigo_stage'] = pd.to_numeric(fixed_all.kdigo_stage_recomputed, errors='coerce')
    fixed_pos = fixed_all.loc[fixed_all.fixed_first_aki_positive].copy()
    fixed_pos['variant'] = 'fixed_first_eligible'
    fixed_pos['combined_positive_variant'] = True
    fixed_pos['combined_stage_variant'] = fixed_pos.fixed_first_kdigo_stage
    variants['fixed_first_eligible'] = fixed_pos
    summary_rows = []
    for name, df in variants.items():
        stagecol = 'combined_stage_variant' if 'combined_stage_variant' in df.columns else 'kdigo_stage_recomputed'
        summary_rows.append({'variant': name, 'subjects': len(df), 'stage_distribution': json.dumps(stage_distribution(df, stagecol), sort_keys=True)})
    variant_summary = pd.DataFrame(summary_rows).sort_values('variant').reset_index(drop=True)
    comparisons = pd.DataFrame([comparison_row(variants['primary'], df, name) for name, df in variants.items() if name != 'primary'])
    display_table(variant_summary)
    display_table(comparisons)
    print('EXPANDED HEART-FAILURE SENSITIVITY POPULATION CONSTRUCTED')
    def uo_path_row(df, label):
        uostage3 = pd.to_numeric(df.uo_stage_mean_ibw_elapsed, errors='coerce').eq(3)
        an = df.uo_path_stage3_anuria_12h.fillna(0).eq(1)
        ol = df.uo_path_stage3_oliguria_24h.fillna(0).eq(1)
        return {'population': label, 'n': len(df), 'uo_kdigo_stage_3_n': int(uostage3.sum()), 'kdigo_stage_3_anuria_12h_path_n': int(an.sum()), 'kdigo_stage_3_oliguria_24h_path_n': int(ol.sum()), 'both_kdigo_stage_3_uo_paths_n': int((an & ol).sum()), 'uo_kdigo_stage_3_without_anuria_12h_n': int((uostage3 & ~an).sum()), 'uo_kdigo_stage_3_without_oliguria_24h_n': int((uostage3 & ~ol).sum())}
    uo_path_qc = pd.DataFrame([uo_path_row(pool.loc[mask_primary], 'primary_eligible_target_stays'), uo_path_row(variants['primary'], 'primary_selected_cohort'), uo_path_row(pool.loc[mask_hfwide], 'hfwide_eligible_target_stays'), uo_path_row(variants['hf_wide'], 'hfwide_selected_cohort')])
    primary_target = pool.loc[mask_primary].copy()
    history_qc = pd.DataFrame([{'primary_target_stays': len(primary_target), 'strict_hist_baseline_available': int(primary_target.hist_strict_median.notna().sum()), 'strict_hist_baseline_available_pct': float(100 * primary_target.hist_strict_median.notna().mean()), 'strict_hist_single_measurement': int(pd.to_numeric(primary_target.hist_strict_n, errors='coerce').fillna(0).eq(1).sum()), 'history_only_creatinine_positive_stays': int(primary_target.history_only_positive.sum()), 'rolling_creatinine_positive_stays': int(pd.to_numeric(primary_target.cr_stage_rolling_only, errors='coerce').fillna(0).ge(1).sum()), 'primary_creatinine_positive_stays': int(pd.to_numeric(primary_target.primary_creatinine_stage, errors='coerce').fillna(0).ge(1).sum())}])
    hf_limit_qc = pd.DataFrame([{'primary_selected_n': len(variants['primary']), 'primary_selected_latest_prior_hf_gt3y_n': int((variants['primary'].hf_recency_days > 365.25 * 3).sum()), 'primary_selected_latest_prior_hf_gt5y_n': int((variants['primary'].hf_recency_days > 365.25 * 5).sum()), 'HF_recency_3y_selected_n': len(variants['hf_recency_3y']), 'HF_recency_1y_selected_n': len(variants['hf_recency_1y']), 'HFwide_selected_n': len(variants['hf_wide']), 'HFexpanded_prior_selected_n': len(variants['expanded_prior_heart_failure']), 'HFexpanded_prior_added_subjects_vs_primary': int(len(set(variants['expanded_prior_heart_failure'].subject_id.astype(int)) - set(variants['primary'].subject_id.astype(int))))}])
    fixed_qc = pd.DataFrame([{'subjects_with_first_eligible_stay': len(fixed_all), 'AKI_positive_at_fixed_first_eligible_stay': int(fixed_all.fixed_first_aki_positive.sum()), 'not_positive_at_fixed_first_eligible_stay': int((~fixed_all.fixed_first_aki_positive).sum()), 'primary_first_positive_subjects': len(variants['primary'])}])
    p = variants['primary'].copy()
    crpos = pd.to_numeric(p.primary_creatinine_stage, errors='coerce').fillna(0).ge(1)
    uopos = pd.to_numeric(p.uo_stage_mean_ibw_elapsed, errors='coerce').fillna(0).ge(1)
    rrtpos = p.active_rrt_any_by_landmark.fillna(0).eq(1)
    p['entry_pattern'] = 'Cr' + crpos.astype(int).astype(str) + '_UO' + uopos.astype(int).astype(str) + '_RRT' + rrtpos.astype(int).astype(str)
    entry_qc = p.entry_pattern.value_counts().rename_axis('entry_pattern').reset_index(name='n')
    for x in [uo_path_qc, history_qc, hf_limit_qc, fixed_qc, entry_qc]:
        display_table(x)
    print('REPORTING QC COMPLETE')
    expanded = variants["expanded_prior_heart_failure"].copy()
    if len(variants["primary"]) != EXPECTED_PRIMARY_N or len(variants["uo_consecutive_ibw_elapsed"]) != EXPECTED_CONSECUTIVE_UO_N:
        raise AssertionError("Reference populations changed while evaluating expanded heart failure.")
    expanded.to_parquet(OUT / "expanded_prior_heart_failure.parquet", index=False)
    comparison = comparison_row(variants["primary"], expanded, "expanded_prior_heart_failure")
    pd.DataFrame([comparison]).to_csv(QC_DIR / "expanded_heart_failure_membership_comparison.csv", index=False)
    hf_qc.to_csv(QC_DIR / "heart_failure_reconstruction.csv", index=False)
    hf_limit_qc.to_csv(QC_DIR / "expanded_heart_failure_summary.csv", index=False)
    hard_guards = {
        "status": "PASS",
        "outcome_blind": True,
        "primary_n": len(variants["primary"]),
        "expanded_prior_heart_failure_n": len(expanded),
        "expanded_prior_heart_failure_added_subjects": comparison["subject_added_vs_reference"],
        "hf_reconstruction_mismatches": int(prior_mismatch + current_mismatch + wide_mismatch + prior_expanded_mismatch + current_expanded_mismatch + expanded_wide_mismatch),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if hard_guards["hf_reconstruction_mismatches"] != 0:
        raise AssertionError("Expanded heart-failure reconstruction mismatch.")
    (OUT / "expanded_heart_failure_manifest.json").write_text(json.dumps(hard_guards, indent=2), encoding="utf-8")
    print("EXPANDED PRIOR-HEART-FAILURE SENSITIVITY: PASS")
    display_table(pd.DataFrame([comparison]))


if __name__ == "__main__":
    main()
