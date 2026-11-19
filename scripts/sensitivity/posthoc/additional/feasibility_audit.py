#!/usr/bin/env python3
"""Disclosure-safe feasibility audit for subgroup and hospital analyses."""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve()
WORKSPACE=HERE.parents[1]
REPO=WORKSPACE.parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(WORKSPACE))
from common import atomic_csv, atomic_json, sha256, snapshot, verify_postflight
from helpers._study_records import OUTCOME_MANIFEST, OUTCOME_MANIFEST_SHA256, verify_study_file

EVAL="outputs/08_external_transport/eicu_outcome_attachment/eicu_external_validation_evaluable.parquet"
FULL="outputs/08_external_transport/eicu_outcome_attachment/eicu_prediction_ready_with_outcome_availability.parquet"
FIXED="outputs/08_external_transport/eicu_final_preoutcome/eicu_prediction_ready_matrix_preoutcome.parquet"
RAW="data/external/eicu-collaborative-research-database-2.0/patient.csv.gz"
RECEIPT="work/posthoc_jicm/checkpoints/EXTERNAL_BASELINE_CONCORDANCE.json"
KEYS=["unique_pid","hstay_id","stay_id","hospital_id"]
OUT=WORKSPACE/"additional"/"results"/"feasibility"
MIN_N=200; MIN_EVENTS=50; MIN_SURVIVORS=50; MIN_HOSPITALS=20
CAL_MIN_EVENTS=100; CAL_MIN_SURVIVORS=100


def safe_count(value: int):
    return int(value) if int(value)>=11 else "SUPPRESSED_LT11"


def group_row(label: str, series: pd.Series, y: pd.Series, hospital: pd.Series) -> dict:
    mask=series.eq(label); n=int(mask.sum()); events=int(y[mask].sum()); survivors=n-events; sites=int(hospital[mask].nunique())
    eligible=n>=MIN_N and events>=MIN_EVENTS and survivors>=MIN_SURVIVORS and sites>=MIN_HOSPITALS
    slope=eligible and events>=CAL_MIN_EVENTS and survivors>=CAL_MIN_SURVIVORS
    return {"group":label,"n":safe_count(n),"deaths":safe_count(events),"survivors":safe_count(survivors),
            "contributing_hospitals":safe_count(sites),"performance_gate":"GO" if eligible else "NO_GO_INSUFFICIENT_SUPPORT",
            "calibration_slope_gate":"GO" if slope else "NO_GO_INSUFFICIENT_EVENTS_OR_SURVIVORS"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-manifest", type=Path, default=OUTCOME_MANIFEST)
    args = parser.parse_args(argv)
    manifest_path = verify_study_file(REPO / args.outcome_manifest, OUTCOME_MANIFEST_SHA256)
    inputs=[EVAL,FULL,FIXED,RAW,RECEIPT,manifest_path.as_posix()]
    before=snapshot(inputs)
    receipt=json.loads((REPO/RECEIPT).read_text(encoding="utf-8"))
    if receipt.get("status")!="PASS" or receipt.get("evaluable_n")!=1897 or receipt.get("events")!=433 or receipt.get("hospitals")!=123:
        raise RuntimeError("Current external baseline concordance receipt is not an eligible PASS")
    evaluable=pd.read_parquet(REPO/EVAL)
    if len(evaluable)!=1897 or int(evaluable.hospital_expire_flag.sum())!=433 or evaluable.hospital_id.nunique()!=123:
        raise RuntimeError("Fixed evaluable cohort guard failed")
    if evaluable[KEYS].isna().any().any() or evaluable[KEYS].duplicated().any():raise RuntimeError("Fixed evaluable identity invalid")
    if evaluable[["age_per10y","sex_female","hospital_expire_flag"]].isna().any().any():raise RuntimeError("Age/sex/outcome missing in evaluable artifact")
    if not set(evaluable.sex_female.unique()).issubset({0.,1.}):raise RuntimeError("Fixed sex indicator is nonbinary")
    age=pd.cut(evaluable.age_per10y,bins=[-np.inf,6.5,7.5,np.inf],right=False,labels=["age_lt65","age_65_74","age_ge75"])
    sex=evaluable.sex_female.map({1.:"female",0.:"male"})
    y=evaluable.hospital_expire_flag.astype(int); hospital=evaluable.hospital_id
    rows=[]
    for label in ["age_lt65","age_65_74","age_ge75"]:rows.append({"variable":"age","missing_n":0,**group_row(label,age,y,hospital)})
    for label in ["female","male"]:rows.append({"variable":"recorded_sex","missing_n":0,**group_row(label,sex,y,hospital)})
    group_table=pd.DataFrame(rows)

    site=evaluable.groupby("hospital_id",sort=False).hospital_expire_flag.agg(n="size",deaths="sum").reset_index(drop=True)
    site["survivors"]=site.n-site.deaths
    site["n_bin"]=pd.cut(site.n,[0,9,19,49,np.inf],labels=["1_9","10_19","20_49","50_plus"])
    site["event_bin"]=pd.cut(site.deaths,[-1,0,4,9,np.inf],labels=["0","1_4","5_9","10_plus"])
    site_bins=[]
    for variable in ["n_bin","event_bin"]:
        for label,count in site[variable].value_counts(sort=False).items():
            site_bins.append({"distribution":variable,"bin":str(label),"hospital_count":safe_count(int(count))})
    adequate=(site.n.ge(20)&site.deaths.ge(5)&site.survivors.ge(5))
    adequate_n=int(adequate.sum())
    hospital_summary={"hospitals":123,"patients":1897,"deaths":433,
      "zero_event_hospitals":safe_count(int(site.deaths.eq(0).sum())),"zero_survivor_hospitals":safe_count(int(site.survivors.eq(0).sum())),
      "hospitals_n_ge20_events_ge5_survivors_ge5":safe_count(adequate_n),
      "hierarchical_intercept_gate":"GO" if adequate_n>=30 else "NO_GO_INSUFFICIENT_SITE_SUPPORT",
      "independent_hospital_slopes":"PROHIBITED"}

    raw=pd.read_csv(REPO/RAW,usecols=["patientunitstayid","gender","age","ethnicity"],low_memory=False)
    if raw.patientunitstayid.duplicated().any():raise RuntimeError("Raw eICU patientunitstayid is not unique")
    linked=evaluable[["stay_id"]].merge(raw,left_on="stay_id",right_on="patientunitstayid",how="left",validate="1:1")
    if len(linked)!=1897 or linked.patientunitstayid.isna().any():raise RuntimeError("Raw demographic linkage incomplete")
    ethnicity=linked.ethnicity.astype("string").str.strip()
    missing=int(ethnicity.isna().sum()); unknown=int(ethnicity.str.lower().isin(["unknown","other/unknown","other","not specified"]).sum())
    counts=ethnicity.fillna("<MISSING>").value_counts()
    demographic_audit={"raw_linkage_n":1897,"raw_linkage_unmatched_n":0,"ethnicity_distinct_recorded_categories":int(counts.size),
      "ethnicity_missing_n":safe_count(missing),"ethnicity_unknown_or_other_n":safe_count(unknown),
      "ethnicity_categories_below_11_n":int((counts<11).sum()),
      "ethnicity_performance_gate":"NO_GO_UNHARMONIZED_SITE_STRUCTURED_FIELD",
      "other_demographic_performance":"NO_GO_NO_VALIDATED_FIXED_FIELDS"}

    OUT.mkdir(parents=True,exist_ok=True)
    atomic_csv(group_table,OUT/"SUBGROUP_FEASIBILITY.csv")
    atomic_csv(pd.DataFrame(site_bins),OUT/"HOSPITAL_SPARSITY_BINS.csv")
    atomic_json(hospital_summary,OUT/"HOSPITAL_FEASIBILITY.json")
    atomic_json(demographic_audit,OUT/"OTHER_DEMOGRAPHICS_AUDIT.json")
    post=verify_postflight(before)
    manifest={"status":"COMPLETE","post_hoc_exploratory":True,"code_sha256":sha256(HERE),"input_hashes":before,
      "postflight":post,"thresholds":{"minimum_n":MIN_N,"minimum_events":MIN_EVENTS,"minimum_survivors":MIN_SURVIVORS,
      "minimum_hospitals":MIN_HOSPITALS,"calibration_slope_minimum_events":CAL_MIN_EVENTS,"calibration_slope_minimum_survivors":CAL_MIN_SURVIVORS},
      "privacy":{"patient_rows_exported":False,"hospital_rows_or_ids_exported":False,"small_cell_threshold":11},
      "outputs":{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}}
    atomic_json(manifest,OUT/"RESULT_MANIFEST.json")
    print(OUT)


if __name__=="__main__":main()
