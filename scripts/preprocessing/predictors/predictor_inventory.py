#!/usr/bin/env python3
"""01 Predictor Inventory.

Build outcome-blind predictor inventory and matrices.

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
from helpers._runtime import logical_source_path


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
        raise ImportError('duckdb is required. Use the same command-line environment used for the phenotype programs.') from e
    CLEAN_ROOT = Path(os.environ['HF_AKI_PROJECT_ROOT']).resolve()
    if not CLEAN_ROOT.is_dir():
        raise FileNotFoundError(CLEAN_ROOT)
    WORK_OUT = CLEAN_ROOT / 'outputs' / '03_predictors'
    WORK_OUT.mkdir(parents=True, exist_ok=True)
    EXPECTED_PRIMARY_N = 3457
    EXPECTED_TARGET_N = 10083
    PROHIBITED_COLUMN_TOKENS = ('hospital_expire', 'expire_flag', 'mortality', 'death', 'deathtime', 'survival', 'outcome', 'dod')

    def assert_outcome_blind(df: pd.DataFrame, label: str):
        bad = [c for c in df.columns if any((tok in c.lower() for tok in PROHIBITED_COLUMN_TOKENS))]
        if bad:
            raise AssertionError(f'{label}: prohibited outcome-like columns detected: {bad}')

    def qpath(p: Path) -> str:
        return str(Path(p).resolve()).replace("'", "''")

    P_INDEX = CLEAN_ROOT / 'outputs' / '01_cohort' / 'primary_cohort_preoutcome.parquet'
    P_TARGET = CLEAN_ROOT / 'outputs' / '02_phenotype' / 'aki_target_preoutcome.parquet'
    for path in (P_INDEX, P_TARGET):
        if not path.is_file():
            raise FileNotFoundError(path)

    idx = pd.read_parquet(P_INDEX, columns=['subject_id', 'stay_id'])
    target_ids = pd.read_parquet(P_TARGET, columns=['stay_id'])
    assert_outcome_blind(idx, 'primary cohort identifiers')
    if len(idx) != EXPECTED_PRIMARY_N:
        raise AssertionError(f'Primary cohort N={len(idx)}, expected {EXPECTED_PRIMARY_N}.')
    if idx.subject_id.duplicated().any() or idx.stay_id.duplicated().any():
        raise AssertionError('Primary cohort identifiers are not unique.')
    if len(target_ids) != EXPECTED_TARGET_N or target_ids.stay_id.duplicated().any():
        raise AssertionError('AKI target count/uniqueness guard failed.')

    hard_guards = pd.DataFrame([
        {'check': 'primary_cohort_n', 'expected': EXPECTED_PRIMARY_N, 'observed': len(idx), 'pass': len(idx) == EXPECTED_PRIMARY_N},
        {'check': 'aki_target_stays', 'expected': EXPECTED_TARGET_N, 'observed': len(target_ids), 'pass': len(target_ids) == EXPECTED_TARGET_N},
        {'check': 'outcome_blind_identifiers', 'expected': True, 'observed': True, 'pass': True},
    ])
    hard_guards.to_csv(WORK_OUT / 'predictor_inventory_hard_guards.csv', index=False)
    display_table(hard_guards)
    if not bool(hard_guards['pass'].all()):
        raise AssertionError('Predictor inventory hard guard failed.')
    print('PREDICTOR INVENTORY HARD GUARDS: PASS')
    MIMIC_ROOT_OPTIONS = [Path(os.environ.get('MIMIC_ROOT', _RAW_DATA_ROOT))]

    def looks_like_mimic(p: Path) -> bool:
        p = Path(p)
        return all([(p / 'hosp' / 'admissions.csv.gz').is_file(), (p / 'hosp' / 'patients.csv.gz').is_file(), (p / 'hosp' / 'diagnoses_icd.csv.gz').is_file(), (p / 'hosp' / 'labevents.csv.gz').is_file(), (p / 'hosp' / 'd_labitems.csv.gz').is_file(), (p / 'icu' / 'icustays.csv.gz').is_file(), (p / 'icu' / 'chartevents.csv.gz').is_file(), (p / 'icu' / 'inputevents.csv.gz').is_file(), (p / 'icu' / 'procedureevents.csv.gz').is_file(), (p / 'icu' / 'd_items.csv.gz').is_file()])
    MIMIC_ROOT = next((p.resolve() for p in MIMIC_ROOT_OPTIONS if looks_like_mimic(p)), None)
    if MIMIC_ROOT is None:
        for f in Path('.').rglob('admissions.csv.gz'):
            if f.parent.name != 'hosp':
                continue
            cand = f.parent.parent
            if looks_like_mimic(cand):
                MIMIC_ROOT = cand.resolve()
                break
    if MIMIC_ROOT is None:
        raise FileNotFoundError('Could not locate a valid MIMIC-IV root.')
    FILES = {'admissions': MIMIC_ROOT / 'hosp' / 'admissions.csv.gz', 'patients': MIMIC_ROOT / 'hosp' / 'patients.csv.gz', 'diagnoses_icd': MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz', 'labevents': MIMIC_ROOT / 'hosp' / 'labevents.csv.gz', 'd_labitems': MIMIC_ROOT / 'hosp' / 'd_labitems.csv.gz', 'icustays': MIMIC_ROOT / 'icu' / 'icustays.csv.gz', 'chartevents': MIMIC_ROOT / 'icu' / 'chartevents.csv.gz', 'inputevents': MIMIC_ROOT / 'icu' / 'inputevents.csv.gz', 'procedureevents': MIMIC_ROOT / 'icu' / 'procedureevents.csv.gz', 'd_items': MIMIC_ROOT / 'icu' / 'd_items.csv.gz'}
    for k, p in FILES.items():
        if not p.is_file():
            raise FileNotFoundError(f'{k}: {p}')
    print('MIMIC_ROOT:', MIMIC_ROOT)
    for k, p in FILES.items():
        print(f'{k:16s} {p}')
    con = duckdb.connect(database=':memory:')
    con.execute('PRAGMA threads=4')
    con.register('idx_ids', idx)
    base = con.execute(f"\nWITH icu AS (\n    SELECT subject_id, hadm_id, stay_id, intime, first_careunit\n    FROM read_csv_auto('{qpath(FILES['icustays'])}', header=true, sample_size=200000)\n),\nadm AS (\n    SELECT subject_id, hadm_id, admittime, admission_type\n    FROM read_csv_auto('{qpath(FILES['admissions'])}', header=true, sample_size=200000)\n),\npat AS (\n    SELECT subject_id, gender, anchor_age, anchor_year\n    FROM read_csv_auto('{qpath(FILES['patients'])}', header=true, sample_size=200000)\n)\nSELECT\n    i.subject_id,\n    icu.hadm_id,\n    i.stay_id,\n    icu.intime,\n    icu.intime + INTERVAL 24 HOUR AS landmark_time,\n    adm.admittime,\n    adm.admission_type,\n    icu.first_careunit,\n    pat.gender,\n    CAST(pat.anchor_age + EXTRACT(YEAR FROM adm.admittime) - pat.anchor_year AS DOUBLE) AS age,\n    DATE_DIFF('minute', adm.admittime, icu.intime) / 60.0 AS hospital_to_icu_hours\nFROM idx_ids i\nJOIN icu ON i.subject_id=icu.subject_id AND i.stay_id=icu.stay_id\nJOIN adm ON icu.subject_id=adm.subject_id AND icu.hadm_id=adm.hadm_id\nJOIN pat ON i.subject_id=pat.subject_id\n").df()
    assert_outcome_blind(base, 'safe base')
    if len(base) != EXPECTED_PRIMARY_N or base.stay_id.nunique() != EXPECTED_PRIMARY_N:
        raise AssertionError('Base reconstruction does not preserve N=3457 / unique stay.')
    base['sex_female'] = base['gender'].astype('string').str.upper().eq('F').astype('int8')
    base_df = base.copy()
    con.register('base_df', base_df)
    con.execute('CREATE OR REPLACE TEMP TABLE base AS SELECT * FROM base_df')
    base_summary = pd.DataFrame([{'n': len(base), 'unique_subjects': base.subject_id.nunique(), 'unique_stays': base.stay_id.nunique(), 'landmark_offset_hours_min': ((base.landmark_time - base.intime).dt.total_seconds() / 3600).min(), 'landmark_offset_hours_max': ((base.landmark_time - base.intime).dt.total_seconds() / 3600).max(), 'age_missing_n': int(base.age.isna().sum()), 'sex_missing_n': int(base.gender.isna().sum()), 'hospital_to_icu_hours_missing_n': int(base.hospital_to_icu_hours.isna().sum())}])
    base_summary.to_csv(WORK_OUT / 'safe_base_summary.csv', index=False)
    display_table(base_summary)
    d_lab = con.execute(f"\nSELECT itemid, label, fluid, category\nFROM read_csv_auto('{qpath(FILES['d_labitems'])}', header=true, sample_size=-1)\n").df()
    d_items = con.execute(f"\nSELECT itemid, label, abbreviation, linksto, category, unitname\nFROM read_csv_auto('{qpath(FILES['d_items'])}', header=true, sample_size=-1)\n").df()
    LAB_IDS = {'albumin': 50862, 'bicarbonate': 50882, 'bun': 51006, 'glucose': 50931, 'sodium': 50983, 'potassium': 50971, 'creatinine': 50912, 'hemoglobin': 51222, 'platelet': 51265, 'wbc': 51301, 'lactate': 50813, 'bilirubin_total': 50885, 'inr': 51237}
    VITAL_IDS = {'heart_rate': [220045], 'map': [220052, 220181, 225312], 'resp_rate': [220210, 224690], 'temperature_f': [223761], 'temperature_c': [223762], 'spo2': [220277]}
    GCS_IDS = [223900, 223901, 220739]
    O2_DEVICE_ID = 226732
    VENT_PROC_IDS = [224385, 225792]
    lab_map = d_lab[d_lab.itemid.isin(LAB_IDS.values())].copy()
    lab_map['expected_feature'] = lab_map.itemid.map({v: k for k, v in LAB_IDS.items()})
    item_map = d_items[d_items.itemid.isin(sorted(set(sum(VITAL_IDS.values(), []) + GCS_IDS + [O2_DEVICE_ID] + VENT_PROC_IDS)))].copy()
    drug_names = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine', 'dobutamine', 'milrinone']
    di = d_items.copy()
    di['label_norm'] = di.label.astype('string').str.strip().str.lower()
    drug_map = di[di.linksto.astype('string').str.lower().eq('inputevents') & di.label_norm.isin(drug_names)][['itemid', 'label', 'label_norm', 'linksto', 'category', 'unitname']].copy()
    lab_map.to_csv(WORK_OUT / 'lab_itemid_verification.csv', index=False)
    item_map.to_csv(WORK_OUT / 'icu_itemid_verification.csv', index=False)
    drug_map.to_csv(WORK_OUT / 'vasoactive_itemid_discovery.csv', index=False)
    missing_lab = sorted(set(LAB_IDS.values()) - set(pd.to_numeric(lab_map.itemid, errors='coerce').dropna().astype(int)))
    if missing_lab:
        raise AssertionError(f'Required lab itemids missing from local d_labitems: {missing_lab}')
    missing_drugs = sorted(set(drug_names) - set(drug_map.label_norm.tolist()))
    if missing_drugs:
        raise AssertionError('Required vasoactive/inotrope labels were not uniquely discoverable in local d_items: ' + ', '.join(missing_drugs))
    critical_expected = {224385: 'intubation', 225792: 'invasive ventilation', 226732: 'o2 delivery device(s)'}
    critical_actual = {int(r.itemid): str(r.label).strip().lower() for r in item_map[item_map.itemid.isin(critical_expected)].itertuples(index=False)}
    bad_critical = {k: {'expected': v, 'observed': critical_actual.get(k)} for k, v in critical_expected.items() if critical_actual.get(k) != v}
    if bad_critical:
        raise AssertionError('Critical respiratory item label guard failed: ' + json.dumps(bad_critical))
    display_table(lab_map.sort_values('itemid'))
    display_table(drug_map.sort_values('label_norm'))
    ce_itemids = sorted(set(sum(VITAL_IDS.values(), []) + GCS_IDS + [O2_DEVICE_ID]))
    ce_ids_sql = ','.join(map(str, ce_itemids))
    con.execute(f"\nCREATE OR REPLACE TEMP TABLE ce_landmark AS\nSELECT\n    ce.subject_id, ce.hadm_id, ce.stay_id,\n    ce.charttime, ce.storetime, ce.itemid,\n    ce.value, ce.valuenum, ce.valueuom, ce.warning\nFROM read_csv_auto(\n    '{qpath(FILES['chartevents'])}',\n    header=true,\n    sample_size=200000\n) ce\nJOIN base b ON ce.stay_id=b.stay_id\nWHERE ce.itemid IN ({ce_ids_sql})\n  AND ce.charttime >= b.intime\n  AND ce.charttime <= b.landmark_time\n  AND ce.storetime IS NOT NULL\n  AND ce.storetime <= b.landmark_time\n")
    vitals = con.execute('\nSELECT\n    stay_id,\n    MAX(CASE WHEN itemid=220045 AND valuenum>0 AND valuenum<300 THEN valuenum END) AS heart_rate_max_0_24h,\n    MIN(CASE WHEN itemid IN (220052,220181,225312) AND valuenum>0 AND valuenum<300 THEN valuenum END) AS map_min_0_24h,\n    MAX(CASE WHEN itemid IN (220210,224690) AND valuenum>0 AND valuenum<70 THEN valuenum END) AS resp_rate_max_0_24h,\n    MIN(CASE\n        WHEN itemid=223761 AND valuenum>70 AND valuenum<120 THEN (valuenum-32.0)/1.8\n        WHEN itemid=223762 AND valuenum>10 AND valuenum<50 THEN valuenum\n    END) AS temperature_min_0_24h,\n    MAX(CASE\n        WHEN itemid=223761 AND valuenum>70 AND valuenum<120 THEN (valuenum-32.0)/1.8\n        WHEN itemid=223762 AND valuenum>10 AND valuenum<50 THEN valuenum\n    END) AS temperature_max_0_24h,\n    MIN(CASE WHEN itemid=220277 AND valuenum>0 AND valuenum<=100 THEN valuenum END) AS spo2_min_0_24h\nFROM ce_landmark\nGROUP BY stay_id\n').df()
    # Adapted GCS component/ETT logic (MIT-LCP mimic-code, MIT License).
    # Copyright (c) 2019 MIT Laboratory for Computational Physiology.
    # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/gcs.sql
    # Original license text: THIRD_PARTY_NOTICES.md.
    con.execute("\nCREATE OR REPLACE TEMP TABLE gcs_base AS\nWITH p AS (\n    SELECT\n        subject_id, stay_id, charttime,\n        MAX(CASE WHEN itemid=223901 THEN valuenum END) AS gcsmotor,\n        MAX(CASE\n            WHEN itemid=223900 AND value='No Response-ETT' THEN 0\n            WHEN itemid=223900 THEN valuenum\n        END) AS gcsverbal,\n        MAX(CASE WHEN itemid=220739 THEN valuenum END) AS gcseyes,\n        MAX(CASE WHEN itemid=223900 AND value='No Response-ETT' THEN 1 ELSE 0 END) AS endotrachflag\n    FROM ce_landmark\n    WHERE itemid IN (223900,223901,220739)\n    GROUP BY subject_id, stay_id, charttime\n)\nSELECT\n    p.*,\n    ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime ASC) AS rn\nFROM p\n")
    gcs = con.execute('\nWITH g AS (\n    SELECT\n        b.subject_id, b.stay_id, b.charttime,\n        CASE\n            WHEN b.gcsverbal=0 THEN 15\n            WHEN b.gcsverbal IS NULL AND b2.gcsverbal=0 THEN 15\n            WHEN b2.gcsverbal=0 THEN\n                COALESCE(b.gcsmotor,6)+COALESCE(b.gcsverbal,5)+COALESCE(b.gcseyes,4)\n            ELSE\n                COALESCE(b.gcsmotor,COALESCE(b2.gcsmotor,6))\n                + COALESCE(b.gcsverbal,COALESCE(b2.gcsverbal,5))\n                + COALESCE(b.gcseyes,COALESCE(b2.gcseyes,4))\n        END AS gcs\n    FROM gcs_base b\n    LEFT JOIN gcs_base b2\n      ON b.stay_id=b2.stay_id\n     AND b.rn=b2.rn+1\n     AND b2.charttime > b.charttime - INTERVAL 6 HOUR\n)\nSELECT stay_id, MIN(gcs) AS gcs_min_0_24h\nFROM g\nGROUP BY stay_id\n').df()
    vent_device = con.execute("\nSELECT\n    stay_id,\n    MAX(\n        CASE WHEN itemid=226732\n          AND LOWER(TRIM(value)) IN ('endotracheal tube','tracheostomy tube')\n        THEN 1 ELSE 0 END\n    ) AS invasive_device_charted_by_landmark\nFROM ce_landmark\nGROUP BY stay_id\n").df()
    for df, label in [(vitals, 'vitals'), (gcs, 'gcs'), (vent_device, 'vent_device')]:
        assert_outcome_blind(df, label)
    print('chartevents scan complete')
    lab_ids_sql = ','.join(map(str, sorted(LAB_IDS.values())))
    labs = con.execute(f"\nWITH x AS (\n    SELECT\n        b.stay_id,\n        le.itemid,\n        CASE\n            WHEN le.itemid=50862 AND le.valuenum>0 AND le.valuenum<=10 THEN le.valuenum\n            WHEN le.itemid=50882 AND le.valuenum>0 AND le.valuenum<=10000 THEN le.valuenum\n            WHEN le.itemid=51006 AND le.valuenum>0 AND le.valuenum<=300 THEN le.valuenum\n            WHEN le.itemid=50931 AND le.valuenum>0 AND le.valuenum<=10000 THEN le.valuenum\n            WHEN le.itemid=50983 AND le.valuenum>0 AND le.valuenum<=200 THEN le.valuenum\n            WHEN le.itemid=50971 AND le.valuenum>0 AND le.valuenum<=30 THEN le.valuenum\n            WHEN le.itemid=50912 AND le.valuenum>0 AND le.valuenum<=150 THEN le.valuenum\n            WHEN le.itemid=51222 AND le.valuenum>0 AND le.valuenum<=50 THEN le.valuenum\n            WHEN le.itemid=51265 AND le.valuenum>0 AND le.valuenum<=10000 THEN le.valuenum\n            WHEN le.itemid=51301 AND le.valuenum>0 AND le.valuenum<=1000 THEN le.valuenum\n            WHEN le.itemid=50813 AND le.valuenum>0 AND le.valuenum<=50 THEN le.valuenum\n            WHEN le.itemid=50885 AND le.valuenum>0 AND le.valuenum<=150 THEN le.valuenum\n            WHEN le.itemid=51237 AND le.valuenum>0 AND le.valuenum<=50 THEN le.valuenum\n            ELSE NULL\n        END AS v\n    FROM read_csv_auto(\n        '{qpath(FILES['labevents'])}',\n        header=true,\n        sample_size=200000\n    ) le\n    JOIN base b\n      ON le.subject_id=b.subject_id\n     AND le.hadm_id=b.hadm_id\n    WHERE le.itemid IN ({lab_ids_sql})\n      AND le.charttime IS NOT NULL\n      AND le.charttime >= b.intime\n      AND le.charttime <= b.landmark_time\n      AND le.storetime IS NOT NULL\n      AND le.storetime <= b.landmark_time\n      AND le.valuenum IS NOT NULL\n)\nSELECT\n    stay_id,\n    MIN(CASE WHEN itemid=51222 THEN v END) AS hemoglobin_min_0_24h,\n    MAX(CASE WHEN itemid=51301 THEN v END) AS wbc_max_0_24h,\n    MIN(CASE WHEN itemid=51265 THEN v END) AS platelet_min_0_24h,\n    MIN(CASE WHEN itemid=50862 THEN v END) AS albumin_min_0_24h,\n    MAX(CASE WHEN itemid=51006 THEN v END) AS bun_max_0_24h,\n    MIN(CASE WHEN itemid=50882 THEN v END) AS bicarbonate_min_0_24h,\n    MIN(CASE WHEN itemid=50983 THEN v END) AS sodium_min_0_24h,\n    MAX(CASE WHEN itemid=50983 THEN v END) AS sodium_max_0_24h,\n    MAX(CASE WHEN itemid=50971 THEN v END) AS potassium_max_0_24h,\n    MAX(CASE WHEN itemid=50931 THEN v END) AS glucose_max_0_24h,\n    MAX(CASE WHEN itemid=50813 THEN v END) AS lactate_max_0_24h,\n    MAX(CASE WHEN itemid=50885 THEN v END) AS bilirubin_total_max_0_24h,\n    MAX(CASE WHEN itemid=51237 THEN v END) AS inr_max_0_24h,\n    MAX(CASE WHEN itemid=50912 THEN v END) AS creatinine_max_0_24h\nFROM x\nGROUP BY stay_id\n").df()
    assert_outcome_blind(labs, 'labs')
    print('labevents scan complete:', len(labs), 'stays with >=1 selected lab')
    vent_proc = con.execute(f"\nSELECT\n    b.stay_id,\n    MAX(CASE WHEN pe.itemid IN (224385,225792) THEN 1 ELSE 0 END) AS invasive_vent_procedure_by_landmark\nFROM base b\nLEFT JOIN read_csv_auto(\n    '{qpath(FILES['procedureevents'])}',\n    header=true,\n    sample_size=200000\n) pe\n  ON pe.stay_id=b.stay_id\n AND pe.itemid IN (224385,225792)\n AND pe.starttime <= b.landmark_time\n AND pe.storetime IS NOT NULL\n AND pe.storetime <= b.landmark_time\nGROUP BY b.stay_id\n").df()
    drug_id_by_name = {r.label_norm: int(r.itemid) for r in drug_map.itertuples(index=False)}
    VASOPRESSOR_NAMES = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine']
    INOTROPE_NAMES = ['dobutamine', 'milrinone']
    vaso_ids = [drug_id_by_name[x] for x in VASOPRESSOR_NAMES]
    ino_ids = [drug_id_by_name[x] for x in INOTROPE_NAMES]
    all_drug_ids = sorted(set(vaso_ids + ino_ids))
    support = con.execute(f"\nSELECT\n    b.stay_id,\n    MAX(CASE WHEN ie.itemid IN ({','.join(map(str, vaso_ids))}) THEN 1 ELSE 0 END) AS vasopressor_any_by_landmark,\n    MAX(CASE WHEN ie.itemid IN ({','.join(map(str, ino_ids))}) THEN 1 ELSE 0 END) AS inotrope_any_by_landmark\nFROM base b\nLEFT JOIN read_csv_auto(\n    '{qpath(FILES['inputevents'])}',\n    header=true,\n    sample_size=200000\n) ie\n  ON ie.stay_id=b.stay_id\n AND ie.itemid IN ({','.join(map(str, all_drug_ids))})\n AND ie.starttime <= b.landmark_time\n AND ie.endtime >= b.intime\n AND ie.storetime IS NOT NULL\n AND ie.storetime <= b.landmark_time\n AND COALESCE(LOWER(ie.statusdescription),'') <> 'rewritten'\nGROUP BY b.stay_id\n").df()
    vent = base[['stay_id']].merge(vent_device, on='stay_id', how='left').merge(vent_proc, on='stay_id', how='left')
    vent[['invasive_device_charted_by_landmark', 'invasive_vent_procedure_by_landmark']] = vent[['invasive_device_charted_by_landmark', 'invasive_vent_procedure_by_landmark']].fillna(0).astype('int8')
    vent['invasive_vent_any_by_landmark'] = (vent.invasive_device_charted_by_landmark.eq(1) | vent.invasive_vent_procedure_by_landmark.eq(1)).astype('int8')
    assert_outcome_blind(vent, 'ventilation support')
    assert_outcome_blind(support, 'vasoactive support')
    display_table(pd.DataFrame([{'invasive_vent_any_n': int(vent.invasive_vent_any_by_landmark.sum()), 'vasopressor_any_n': int(support.vasopressor_any_by_landmark.fillna(0).sum()), 'inotrope_any_n': int(support.inotrope_any_by_landmark.fillna(0).sum())}]))
    prior_comorb = con.execute(f"\nWITH prior_codes AS (\n    SELECT\n        b.subject_id,\n        b.stay_id,\n        d.icd_version,\n        UPPER(REPLACE(TRIM(d.icd_code),'.','')) AS code\n    FROM base b\n    JOIN read_csv_auto(\n        '{qpath(FILES['admissions'])}',\n        header=true,\n        sample_size=200000\n    ) pa\n      ON pa.subject_id=b.subject_id\n     AND pa.hadm_id<>b.hadm_id\n     AND pa.dischtime IS NOT NULL\n     AND pa.dischtime < b.admittime\n    JOIN read_csv_auto(\n        '{qpath(FILES['diagnoses_icd'])}',\n        header=true,\n        sample_size=200000\n    ) d\n      ON d.subject_id=pa.subject_id AND d.hadm_id=pa.hadm_id\n)\nSELECT\n    stay_id,\n    MAX(CASE WHEN\n        (icd_version=9 AND code LIKE '585%')\n        OR (icd_version=10 AND code LIKE 'N18%')\n    THEN 1 ELSE 0 END) AS prior_ckd,\n    MAX(CASE WHEN\n        (icd_version=9 AND code LIKE '250%')\n        OR (icd_version=10 AND (\n            code LIKE 'E08%' OR code LIKE 'E09%' OR code LIKE 'E10%'\n            OR code LIKE 'E11%' OR code LIKE 'E13%'\n        ))\n    THEN 1 ELSE 0 END) AS prior_diabetes,\n    MAX(CASE WHEN\n        (icd_version=9 AND (code LIKE '42731%' OR code LIKE '42732%'))\n        OR (icd_version=10 AND code LIKE 'I48%')\n    THEN 1 ELSE 0 END) AS prior_af_or_flutter,\n    MAX(CASE WHEN\n        (icd_version=9 AND (\n            code LIKE '410%' OR code LIKE '411%' OR code LIKE '412%'\n            OR code LIKE '413%' OR code LIKE '414%'\n        ))\n        OR (icd_version=10 AND (\n            code LIKE 'I20%' OR code LIKE 'I21%' OR code LIKE 'I22%'\n            OR code LIKE 'I23%' OR code LIKE 'I24%' OR code LIKE 'I25%'\n        ))\n    THEN 1 ELSE 0 END) AS prior_ischemic_heart_disease,\n    MAX(CASE WHEN\n        (icd_version=9 AND (\n            code LIKE '490%' OR code LIKE '491%' OR code LIKE '492%'\n            OR code LIKE '493%' OR code LIKE '494%' OR code LIKE '495%'\n            OR code LIKE '496%'\n        ))\n        OR (icd_version=10 AND (\n            code LIKE 'J40%' OR code LIKE 'J41%' OR code LIKE 'J42%'\n            OR code LIKE 'J43%' OR code LIKE 'J44%' OR code LIKE 'J45%'\n            OR code LIKE 'J46%' OR code LIKE 'J47%'\n        ))\n    THEN 1 ELSE 0 END) AS prior_copd_or_chronic_airway_disease\nFROM prior_codes\nGROUP BY stay_id\n").df()
    for c in ['prior_ckd', 'prior_diabetes', 'prior_af_or_flutter', 'prior_ischemic_heart_disease', 'prior_copd_or_chronic_airway_disease']:
        prior_comorb[c] = pd.to_numeric(prior_comorb[c], errors='coerce').fillna(0).astype('int8')
    assert_outcome_blind(prior_comorb, 'prior comorbidity inventory')
    display_table(prior_comorb.drop(columns='stay_id').sum().to_frame('positive_n'))
    target_cols = ['subject_id', 'stay_id', 'creatinine_stage', 'urine_output_stage', 'active_rrt_any_by_landmark', 'kdigo_stage']
    target = pd.read_parquet(P_TARGET, columns=target_cols)
    assert_outcome_blind(target, ' renal target')
    renal = idx.merge(target, on=['subject_id', 'stay_id'], how='left', validate='1:1')
    if renal.kdigo_stage.isna().any():
        raise AssertionError('Selected  primary cohort has missing kdigo_stage.')
    renal = renal.rename(columns={'active_rrt_any_by_landmark': 'rrt_any_by_landmark'})
    display_table(renal[['kdigo_stage', 'creatinine_stage', 'urine_output_stage', 'rrt_any_by_landmark']].describe(include='all'))
    matrix = base[['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time', 'age', 'sex_female', 'hospital_to_icu_hours', 'admission_type', 'first_careunit']].copy()
    for df in [vitals, gcs, labs, vent[['stay_id', 'invasive_vent_any_by_landmark']], support, prior_comorb, renal.drop(columns=['subject_id'])]:
        matrix = matrix.merge(df, on='stay_id', how='left', validate='1:1')
    for c in ['invasive_vent_any_by_landmark', 'vasopressor_any_by_landmark', 'inotrope_any_by_landmark']:
        matrix[c] = pd.to_numeric(matrix[c], errors='coerce').fillna(0).astype('int8')
    for c in ['prior_ckd', 'prior_diabetes', 'prior_af_or_flutter', 'prior_ischemic_heart_disease', 'prior_copd_or_chronic_airway_disease']:
        matrix[c] = pd.to_numeric(matrix[c], errors='coerce').fillna(0).astype('int8')
    assert_outcome_blind(matrix, 'alternative predictor matrix')
    if len(matrix) != EXPECTED_PRIMARY_N or matrix.subject_id.nunique() != EXPECTED_PRIMARY_N or matrix.stay_id.nunique() != EXPECTED_PRIMARY_N:
        raise AssertionError('predictor matrix does not preserve primary one-row-per-subject/stay N=3457.')
    matrix_ids = set(map(tuple, matrix[['subject_id', 'stay_id']].astype('int64').to_numpy()))
    cohort_ids = set(map(tuple, idx[['subject_id', 'stay_id']].astype('int64').to_numpy()))
    if matrix_ids != cohort_ids:
        raise AssertionError('Predictor matrix membership differs from the primary cohort.')
    P_MATRIX = WORK_OUT / 'full_predictor_matrix.parquet'
    matrix.to_parquet(P_MATRIX, index=False)
    print('predictor matrix:', P_MATRIX)
    print('Rows:', len(matrix))
    print('Columns:', len(matrix.columns))
    rows = []

    def add(feature, domain, source, raw, timestamp, window, aggregation, unit_rule, plausibility, eicu_source, eicu_mapping, portable, role, leakage='PASS', notes=''):
        rows.append({'feature': feature, 'clinical_domain': domain, 'mimic_source_table': source, 'mimic_raw_field_or_itemid': raw, 'timestamp_field': timestamp, 'allowed_window': window, 'aggregation': aggregation, 'unit_rule': unit_rule, 'plausibility_rule': plausibility, 'available_by_landmark': 'YES', 'eicu_source': eicu_source, 'eicu_mapping': eicu_mapping, 'portable': portable, 'prespecified_role': role, 'leakage_status': leakage, 'notes': notes})
    add('age', 'demographics', 'hosp.patients + hosp.admissions', 'anchor_age, anchor_year, admittime', 'admittime', 'known pre-landmark', 'anchor_age + admission_year - anchor_year', 'years', 'MIMIC deidentification caps oldest ages via anchor_age', 'patient', 'age', 'YES', 'CORE')
    add('sex_female', 'demographics', 'hosp.patients', 'gender', 'none', 'known pre-landmark', 'F vs other', 'binary', 'none', 'patient', 'gender', 'YES', 'CORE')
    add('hospital_to_icu_hours', 'context', 'hosp.admissions + icu.icustays', 'admittime, intime', 'admittime/intime', 'known at ICU entry', '(intime-admittime) hours', 'hours', 'nonnegative expected; negatives retained for QC review', 'patient', 'hospitalAdmitOffset relative to unit admission', 'YES', 'CORE')
    add('admission_type', 'context', 'hosp.admissions', 'admission_type', 'none', 'known at admission', 'categorical', 'category', 'none', 'patient/admissionDx', 'conceptual admission urgency/source', 'REVIEW', 'REVIEW_PORTABILITY', notes='Category harmonisation across databases required.')
    add('first_careunit', 'context', 'icu.icustays', 'first_careunit', 'none', 'known at ICU entry', 'categorical', 'category', 'none', 'patient', 'unitType / unitStayType conceptual mapping', 'REVIEW', 'REVIEW_PORTABILITY')
    add('heart_rate_max_0_24h', 'vitals', 'icu.chartevents', '220045', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'maximum', 'beats/min', '0<HR<300', 'vitalPeriodic', 'heartrate', 'YES', 'CORE')
    add('map_min_0_24h', 'vitals', 'icu.chartevents', '220052|220181|225312', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'minimum', 'mmHg', '0<MAP<300', 'vitalPeriodic/vitalAperiodic', 'systemicmean/noninvasivemean', 'YES', 'CORE')
    add('resp_rate_max_0_24h', 'vitals', 'icu.chartevents', '220210|224690', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'maximum', 'breaths/min', '0<RR<70', 'vitalPeriodic', 'respiration', 'YES', 'CORE')
    add('temperature_min_0_24h', 'vitals', 'icu.chartevents', '223761|223762', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'minimum after F->C conversion', 'degC', '70<F<120 or 10<C<50', 'vitalPeriodic', 'temperature', 'YES', 'CORE')
    add('temperature_max_0_24h', 'vitals', 'icu.chartevents', '223761|223762', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'maximum after F->C conversion', 'degC', '70<F<120 or 10<C<50', 'vitalPeriodic', 'temperature', 'YES', 'CORE')
    add('spo2_min_0_24h', 'vitals', 'icu.chartevents', '220277', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'minimum', '%', '0<SpO2<=100', 'vitalPeriodic', 'sao2', 'YES', 'CORE')
    add('gcs_min_0_24h', 'neurologic', 'icu.chartevents', '223900|223901|220739', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'minimum MIT-LCP-style reconstructed GCS', 'score 3-15', 'MIT-LCP component/ETT logic; prior row within 6 h', 'apacheApsVar / nurseCharting', 'GCS concept mapping required', 'REVIEW', 'REVIEW_PORTABILITY', notes='MIMIC extraction follows core current MIT-LCP logic plus storetime guard; eICU mapping requires an explicit prespecified definition.')
    lab_specs = [('hemoglobin_min_0_24h', '51222', 'minimum', 'g/dL', '0<value<=50', 'Hgb/hemoglobin'), ('wbc_max_0_24h', '51301', 'maximum', 'K/uL', '0<value<=1000', 'WBC'), ('platelet_min_0_24h', '51265', 'minimum', 'K/uL', '0<value<=10000', 'platelets'), ('albumin_min_0_24h', '50862', 'minimum', 'g/dL', '0<value<=10', 'albumin'), ('bun_max_0_24h', '51006', 'maximum', 'mg/dL', '0<value<=300', 'BUN'), ('bicarbonate_min_0_24h', '50882', 'minimum', 'mEq/L', '0<value<=10000', 'bicarbonate/HCO3'), ('sodium_min_0_24h', '50983', 'minimum', 'mEq/L', '0<value<=200', 'sodium'), ('sodium_max_0_24h', '50983', 'maximum', 'mEq/L', '0<value<=200', 'sodium'), ('potassium_max_0_24h', '50971', 'maximum', 'mEq/L', '0<value<=30', 'potassium'), ('glucose_max_0_24h', '50931', 'maximum', 'mg/dL', '0<value<=10000', 'glucose'), ('lactate_max_0_24h', '50813', 'maximum', 'mmol/L', '0<value<=50', 'lactate'), ('bilirubin_total_max_0_24h', '50885', 'maximum', 'mg/dL', '0<value<=150', 'total bilirubin'), ('inr_max_0_24h', '51237', 'maximum', 'ratio', '0<value<=50', 'INR')]
    for feat, itemid, agg, uom, plaus, eicu_lab in lab_specs:
        add(feat, 'laboratory', 'hosp.labevents', itemid, 'charttime + storetime', '[intime, landmark], storetime<=landmark', agg, uom, plaus, 'lab', eicu_lab, 'YES', 'CORE')
    add('invasive_vent_any_by_landmark', 'respiratory_support', 'icu.procedureevents + icu.chartevents', '224385|225792|226732', 'start/charttime + storetime', 'by landmark', 'any direct procedure OR ETT/trach device evidence', 'binary', 'source-specific direct evidence only', 'respiratoryCare/treatment', 'invasive ventilation conceptual mapping', 'REVIEW', 'REVIEW_SOURCE_COMPLETENESS', notes='Inventory definition is deliberately direct-evidence based; compare with full official ventilation concept before model inclusion.')
    add('vasopressor_any_by_landmark', 'hemodynamic_support', 'icu.inputevents', '|'.join(map(str, vaso_ids)), 'starttime/endtime + storetime', 'overlap [intime,landmark], storetime<=landmark', 'any norepinephrine/epinephrine/phenylephrine/vasopressin/dopamine', 'binary', 'exclude rewritten rows', 'infusionDrug', 'drug-name mapping', 'YES', 'CORE')
    add('inotrope_any_by_landmark', 'hemodynamic_support', 'icu.inputevents', '|'.join(map(str, ino_ids)), 'starttime/endtime + storetime', 'overlap [intime,landmark], storetime<=landmark', 'any dobutamine/milrinone', 'binary', 'exclude rewritten rows', 'infusionDrug', 'dobutamine/milrinone name mapping', 'YES', 'CORE')
    add('kdigo_stage', 'renal_aki', 'adjudicated AKI target', 'kdigo_stage', '+24h landmark', 'by landmark', 'combined KDIGO stage', '1/2/3', 'AKI phenotype definition', 'constructed from eICU renal sources', 'conceptual harmonisation only', 'REVIEW', 'CORE', notes='Primary renal-severity alternative; exact eICU phenotype instrument differs.')
    add('creatinine_max_0_24h', 'renal_aki', 'hosp.labevents', '50912', 'charttime + storetime', '[intime, landmark], storetime<=landmark', 'maximum', 'mg/dL', '0<value<=150', 'lab', 'creatinine', 'YES', 'SECONDARY_REDUNDANCY_REVIEW', notes='Potentially redundant with the adjudicated KDIGO stage; not silently included together.')
    add('creatinine_stage', 'renal_aki', 'adjudicated AKI target', 'creatinine_stage', '+24h landmark', 'by landmark', 'creatinine component stage', 'stage', 'AKI phenotype', 'lab', 'conceptual', 'REVIEW', 'SECONDARY_REDUNDANCY_REVIEW')
    add('urine_output_stage', 'renal_aki', 'adjudicated AKI target', 'urine_output_stage', '+24h landmark', 'by landmark', 'urine-output component stage', 'stage', 'AKI phenotype', 'intakeOutput', 'conceptual', 'REVIEW', 'SECONDARY_REDUNDANCY_REVIEW')
    add('rrt_any_by_landmark', 'renal_aki', 'adjudicated AKI target', 'active_rrt_any_by_landmark', '+24h landmark', 'by landmark', 'adjudicated RRT evidence', 'binary', 'adjudicated phenotype policy', 'treatment', 'conceptual', 'REVIEW', 'SECONDARY_REDUNDANCY_REVIEW')
    for feat, desc, eicu in [('prior_ckd', 'prior_comorbidity', 'pastHistory/diagnosis'), ('prior_diabetes', 'prior_comorbidity', 'pastHistory/diagnosis'), ('prior_af_or_flutter', 'prior_comorbidity', 'pastHistory/diagnosis'), ('prior_ischemic_heart_disease', 'prior_comorbidity', 'pastHistory/diagnosis'), ('prior_copd_or_chronic_airway_disease', 'prior_comorbidity', 'pastHistory/diagnosis')]:
        add(feat, 'prior_comorbidity', 'prior completed admissions: hosp.admissions + diagnoses_icd', 'prespecified ICD prefix groups', 'prior-admission dischtime', 'strictly before index admittime', 'ever coded on a completed prior admission', 'binary', 'absence of prior MIMIC code is not proof of clinical absence', eicu, 'conceptual prior-history mapping', 'REVIEW', 'REVIEW_PORTABILITY', notes='Current index-admission discharge diagnosis codes are prohibited for predictor construction.')
    prov = pd.DataFrame(rows)
    id_cols = {'subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time'}
    feature_cols = [c for c in matrix.columns if c not in id_cols]
    summ = []
    for c in feature_cols:
        s = matrix[c]
        row = {'feature': c, 'n': len(s), 'nonmissing_n': int(s.notna().sum()), 'missing_n': int(s.isna().sum()), 'missing_pct': float(s.isna().mean() * 100), 'n_unique_nonmissing': int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s):
            q = pd.to_numeric(s, errors='coerce')
            row.update({'min': float(q.min()) if q.notna().any() else np.nan, 'p01': float(q.quantile(0.01)) if q.notna().any() else np.nan, 'p25': float(q.quantile(0.25)) if q.notna().any() else np.nan, 'median': float(q.quantile(0.5)) if q.notna().any() else np.nan, 'p75': float(q.quantile(0.75)) if q.notna().any() else np.nan, 'p99': float(q.quantile(0.99)) if q.notna().any() else np.nan, 'max': float(q.max()) if q.notna().any() else np.nan})
        else:
            row.update({'min': np.nan, 'p01': np.nan, 'p25': np.nan, 'median': np.nan, 'p75': np.nan, 'p99': np.nan, 'max': np.nan})
        summ.append(row)
    feature_summary = pd.DataFrame(summ)
    prov = prov.merge(feature_summary[['feature', 'nonmissing_n', 'missing_n', 'missing_pct', 'n_unique_nonmissing']], on='feature', how='left', validate='1:1')

    def readiness(r):
        if r.prespecified_role == 'SECONDARY_REDUNDANCY_REVIEW':
            return 'REVIEW_REDUNDANCY'
        if str(r.portable).upper() == 'REVIEW':
            return 'REVIEW_PORTABILITY_OR_SOURCE'
        if pd.isna(r.missing_pct):
            return 'REVIEW_NOT_EXTRACTED'
        if float(r.missing_pct) >= 50:
            return 'REVIEW_HIGH_MISSINGNESS_GE50'
        if float(r.missing_pct) >= 30:
            return 'REVIEW_MODERATE_MISSINGNESS_30_50'
        return 'READY_ON_INVENTORY'
    prov['inventory_readiness'] = prov.apply(readiness, axis=1)
    prov['inclusion_decision'] = 'OUTCOME_BLIND_ASSESSMENT_NEEDED'
    external_status_path = CLEAN_ROOT / 'quality_control' / 'external_validation_status.json'
    external_status = json.loads(external_status_path.read_text(encoding='utf-8'))
    if external_status.get('status') == 'FEASIBLE':
        if external_status.get('feasibility_assessment_completed') is not True:
            raise AssertionError('eICU FEASIBLE status lacks a completed feasibility assessment.')
        prov['missingness_eicu'] = 'NOT_ASSESSED_FEASIBILITY_ONLY'
    elif external_status.get('status') == 'DATA_UNAVAILABLE':
        prov['missingness_eicu'] = 'DATA_UNAVAILABLE'
    else:
        raise AssertionError(f"Unsupported eICU branch status: {external_status.get('status')}")
    prov['notes'] = prov['notes'].fillna('')
    binary_presence = {'invasive_vent_any_by_landmark', 'vasopressor_any_by_landmark', 'inotrope_any_by_landmark', 'prior_ckd', 'prior_diabetes', 'prior_af_or_flutter', 'prior_ischemic_heart_disease', 'prior_copd_or_chronic_airway_disease', 'sex_female', 'rrt_any_by_landmark'}
    prov.loc[prov.feature.isin(binary_presence), 'notes'] = prov.loc[prov.feature.isin(binary_presence), 'notes'].astype(str) + ' Binary absence is encoded as 0 under the stated source instrument.'
    feature_summary.to_csv(WORK_OUT / 'feature_missingness_distribution.csv', index=False)
    prov.to_csv(WORK_OUT / 'predictor_provenance.csv', index=False)
    cat_rows = []
    for c in ['admission_type', 'first_careunit']:
        vc = matrix[c].astype('string').fillna('<MISSING>').value_counts(dropna=False)
        for level, n in vc.items():
            cat_rows.append({'feature': c, 'level': str(level), 'n': int(n), 'pct': float(n / len(matrix) * 100)})
    pd.DataFrame(cat_rows).to_csv(WORK_OUT / 'categorical_frequencies.csv', index=False)
    display_table(prov[['feature', 'prespecified_role', 'portable', 'missing_pct', 'inventory_readiness', 'inclusion_decision']].sort_values(['prespecified_role', 'missing_pct'], na_position='last'))
    timing_qc = pd.DataFrame([{'rule': 'predictor matrix contains prohibited outcome-like columns', 'violations': sum((any((tok in c.lower() for tok in PROHIBITED_COLUMN_TOKENS)) for c in matrix.columns)), 'pass': not any((any((tok in c.lower() for tok in PROHIBITED_COLUMN_TOKENS)) for c in matrix.columns))}, {'rule': 'one row per primary selected stay', 'violations': int(len(matrix) - matrix.stay_id.nunique()), 'pass': len(matrix) == matrix.stay_id.nunique() == EXPECTED_PRIMARY_N}, {'rule': 'primary cohort membership preserved', 'violations': int(len(set(map(tuple, matrix[['subject_id','stay_id']].astype('int64').to_numpy()))) != EXPECTED_PRIMARY_N), 'pass': set(map(tuple, matrix[['subject_id','stay_id']].astype('int64').to_numpy())) == set(map(tuple, idx[['subject_id','stay_id']].astype('int64').to_numpy()))}, {'rule': 'landmark exactly +24h from ICU intime', 'violations': int((~np.isclose((matrix.landmark_time - matrix.intime).dt.total_seconds() / 3600.0, 24.0)).sum()), 'pass': bool(np.isclose((matrix.landmark_time - matrix.intime).dt.total_seconds() / 3600.0, 24.0).all())}, {'rule': 'current-admission diagnoses_icd used as predictor', 'violations': 0, 'pass': True}])
    display_table(timing_qc)
    if not bool(timing_qc['pass'].all()):
        raise AssertionError('Leakage/timing QC failed.')
    source_rows = [
        {'source_type': 'primary_cohort', 'path': logical_source_path(P_INDEX), 'size_bytes': P_INDEX.stat().st_size},
        {'source_type': 'aki_target', 'path': logical_source_path(P_TARGET), 'size_bytes': P_TARGET.stat().st_size},
    ]
    for name, raw_path in FILES.items():
        source_rows.append({
            'source_type': f'raw_mimic_{name}',
            'path': logical_source_path(raw_path),
            'size_bytes': raw_path.stat().st_size,
        })
    source_rows.append({
        'source_type': 'predictor_matrix',
        'path': logical_source_path(P_MATRIX),
        'size_bytes': P_MATRIX.stat().st_size,
    })
    source_prov = pd.DataFrame(source_rows)
    source_prov.to_csv(WORK_OUT / 'source_provenance.csv', index=False)

    decision_template = prov[[
        'feature', 'clinical_domain', 'prespecified_role', 'portable', 'missing_pct',
        'inventory_readiness', 'leakage_status', 'eicu_source', 'eicu_mapping',
        'mimic_source_table', 'mimic_raw_field_or_itemid', 'allowed_window',
        'aggregation', 'inclusion_decision', 'notes'
    ]].copy()
    decision_template['investigator_decision'] = 'ASSESSMENT_NEEDED'
    decision_template['investigator_rationale'] = ''
    decision_template.to_csv(WORK_OUT / 'predictor_decisions_template.csv', index=False)

    n_ready = int(prov.inventory_readiness.eq('READY_ON_INVENTORY').sum())
    n_review = int((~prov.inventory_readiness.eq('READY_ON_INVENTORY')).sum())
    manifest = {
        'status': 'PASS',
        'outcome_blind': True,
        'outcome_values_read': False,
        'model_fit_performed': False,
        'primary_n': EXPECTED_PRIMARY_N,
        'predictor_feature_count': int(len(prov)),
        'inventory_ready_count': n_ready,
        'assessment_needed_count': n_review,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    (WORK_OUT / 'predictor_inventory_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )

    top_missing = prov[[
        'feature', 'prespecified_role', 'portable', 'missing_pct', 'inventory_readiness'
    ]].sort_values('missing_pct', ascending=False, na_position='last')
    top_missing.to_csv(WORK_OUT / 'missingness_review.csv', index=False)


    print('PREDICTOR INVENTORY: PASS')
    print('Primary N:', EXPECTED_PRIMARY_N)
    print('Predictor features:', len(prov))


if __name__ == "__main__":
    main()
