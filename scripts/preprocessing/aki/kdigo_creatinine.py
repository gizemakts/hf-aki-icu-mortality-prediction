#!/usr/bin/env python3
"""01 Kdigo Creatinine.

Compute the KDIGO creatinine component.

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
    import sys, hashlib
    import pandas as pd
    import duckdb
    CONFIRMED_MIMIC_ROOT = Path('data/raw/mimic-iv-3.1')
    CONFIRMED_PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT)))
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', CONFIRMED_MIMIC_ROOT))
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', CONFIRMED_PROJECT_ROOT))
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    CREATININE_DIR = PHENOTYPE_DIR / 'creatinine'
    PHENOTYPE_QC_DIR = PROJECT_ROOT / 'quality_control' / 'phenotype'
    for d in [COHORT_DIR, PHENOTYPE_DIR, CREATININE_DIR, PHENOTYPE_QC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not MIMIC_ROOT.exists():
        raise FileNotFoundError(f'MIMIC_ROOT not found: {MIMIC_ROOT}')

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")

    def require_file(p: Path):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def require_cols(df, cols, label='dataframe'):
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise KeyError(f'{label} missing columns: {miss}')

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: null {key}')
        ndup = int(df[key].duplicated().sum())
        if ndup:
            raise AssertionError(f'{label}: duplicate {key} rows={ndup}')

    def sha256_file(p: Path, block=1024 * 1024):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while True:
                b = f.read(block)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    print('MIMIC_ROOT :', MIMIC_ROOT)
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('CREATININE_DIR:', CREATININE_DIR)
    print('Python     :', sys.version.split()[0])
    print('pandas     :', pd.__version__)
    print('duckdb     :', duckdb.__version__)
    con = duckdb.connect(database=':memory:')
    LANDMARK_PATH = require_file(COHORT_DIR / 'landmark_population.parquet')
    LAB_PATH = require_file(MIMIC_ROOT / 'hosp' / 'labevents.csv.gz')
    DLAB_PATH = require_file(MIMIC_ROOT / 'hosp' / 'd_labitems.csv.gz')
    con.execute(f"CREATE VIEW landmark_eligible AS SELECT * FROM read_parquet('{qpath(LANDMARK_PATH)}')")
    con.execute(f"CREATE VIEW labevents AS SELECT * FROM read_csv_auto('{qpath(LAB_PATH)}',header=true,sample_size=200000)")
    labels = pd.read_csv(DLAB_PATH, compression='gzip', usecols=['itemid', 'label'])
    row = labels.loc[labels.itemid.eq(50912)]
    if len(row) != 1:
        raise AssertionError(f'Expected exactly one d_labitems row for 50912; got {len(row)}')
    label = str(row.iloc[0].label)
    print('50912 label:', label)
    if 'creatin' not in label.lower():
        raise AssertionError(f'Unexpected 50912 label: {label}')
    CREATININE_ITEMID = 50912
    con.execute(f"\nCREATE OR REPLACE TEMP TABLE creat_same_time AS\nSELECT l.stay_id,l.subject_id,l.hadm_id,l.admittime,l.intime,l.landmark_time,\n       le.charttime,MAX(le.storetime) AS cr_storetime,AVG(le.valuenum) AS creatinine\nFROM landmark_eligible l JOIN labevents le ON le.subject_id=l.subject_id\nWHERE le.itemid={CREATININE_ITEMID} AND le.valuenum>0 AND le.valuenum<=150\n  AND le.charttime>=l.intime-INTERVAL '7 days' AND le.charttime<=l.landmark_time\n  AND le.storetime IS NOT NULL AND le.storetime<=l.landmark_time\nGROUP BY l.stay_id,l.subject_id,l.hadm_id,l.admittime,l.intime,l.landmark_time,le.charttime\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE creat_refs AS\nSELECT *,\n MIN(creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '48 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low48,\n MIN(creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low7d,\n ARG_MIN(charttime,creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '48 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low48_ref_time,\n ARG_MIN(charttime,creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low7d_ref_time,\n ARG_MIN(cr_storetime,creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '48 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low48_ref_storetime,\n ARG_MIN(cr_storetime,creatinine) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING) low7d_ref_storetime\nFROM creat_same_time\n")
    con.execute('\nCREATE OR REPLACE TEMP TABLE creat_events_stage AS\nSELECT *,CASE\n WHEN low48 IS NULL AND low7d IS NULL THEN NULL\n WHEN low7d IS NOT NULL AND creatinine>=3.0*low7d THEN 3\n WHEN creatinine>=4.0 AND ((low48 IS NOT NULL AND creatinine>=low48+0.3) OR (low7d IS NOT NULL AND creatinine>=1.5*low7d)) THEN 3\n WHEN low7d IS NOT NULL AND creatinine>=2.0*low7d THEN 2\n WHEN low48 IS NOT NULL AND creatinine>=low48+0.3 THEN 1\n WHEN low7d IS NOT NULL AND creatinine>=1.5*low7d THEN 1\n ELSE 0 END AS kdigo_creatinine_stage\nFROM creat_refs WHERE charttime>=intime AND charttime<=landmark_time\n')
    con.execute('\nCREATE OR REPLACE TEMP TABLE creat_events AS\nSELECT *,\n CASE WHEN low48 IS NULL AND low7d IS NULL THEN NULL WHEN low7d IS NOT NULL AND creatinine>=3.0*low7d THEN low7d_ref_time\n WHEN creatinine>=4.0 AND low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48_ref_time\n WHEN creatinine>=4.0 AND low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d_ref_time\n WHEN low7d IS NOT NULL AND creatinine>=2.0*low7d THEN low7d_ref_time WHEN low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48_ref_time\n WHEN low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d_ref_time WHEN low48 IS NOT NULL THEN low48_ref_time ELSE low7d_ref_time END baseline_reference_time,\n CASE WHEN low48 IS NULL AND low7d IS NULL THEN NULL WHEN low7d IS NOT NULL AND creatinine>=3.0*low7d THEN low7d_ref_storetime\n WHEN creatinine>=4.0 AND low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48_ref_storetime\n WHEN creatinine>=4.0 AND low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d_ref_storetime\n WHEN low7d IS NOT NULL AND creatinine>=2.0*low7d THEN low7d_ref_storetime WHEN low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48_ref_storetime\n WHEN low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d_ref_storetime WHEN low48 IS NOT NULL THEN low48_ref_storetime ELSE low7d_ref_storetime END baseline_reference_storetime,\n CASE WHEN low48 IS NULL AND low7d IS NULL THEN NULL WHEN low7d IS NOT NULL AND creatinine>=3.0*low7d THEN low7d\n WHEN creatinine>=4.0 AND low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48\n WHEN creatinine>=4.0 AND low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d\n WHEN low7d IS NOT NULL AND creatinine>=2.0*low7d THEN low7d WHEN low48 IS NOT NULL AND creatinine>=low48+0.3 THEN low48\n WHEN low7d IS NOT NULL AND creatinine>=1.5*low7d THEN low7d WHEN low48 IS NOT NULL THEN low48 ELSE low7d END baseline_reference_value\nFROM creat_events_stage\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE ranked AS\nSELECT *,CASE WHEN baseline_reference_time IS NULL THEN 'insufficient' WHEN baseline_reference_time<admittime THEN 'prior_pre_admission'\n WHEN baseline_reference_time<intime THEN 'current_hospital_pre_icu' ELSE 'current_icu_pre_event' END baseline_provenance,\n ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY kdigo_creatinine_stage DESC NULLS LAST,charttime ASC) stage_rank\nFROM creat_events\n")
    con.execute('\nCREATE OR REPLACE TEMP TABLE usage AS SELECT stay_id,MAX(charttime) max_cr_charttime_used,MAX(cr_storetime) max_cr_storetime_used,COUNT(*) creatinine_measurement_count_used FROM creat_same_time GROUP BY stay_id\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE creat_stay AS\nSELECT l.stay_id,MAX(c.kdigo_creatinine_stage) kdigo_creatinine_stage,\n MAX(CASE WHEN c.kdigo_creatinine_stage IS NOT NULL THEN 1 ELSE 0 END) creatinine_evaluable,\n COALESCE(w.baseline_provenance,'insufficient') baseline_provenance,w.baseline_reference_time,w.baseline_reference_storetime,w.baseline_reference_value,\n u.max_cr_charttime_used,u.max_cr_storetime_used,COALESCE(u.creatinine_measurement_count_used,0) creatinine_measurement_count_used,\n 50912 AS creatinine_itemid\nFROM landmark_eligible l LEFT JOIN creat_events c USING(stay_id) LEFT JOIN ranked w ON w.stay_id=l.stay_id AND w.stage_rank=1 LEFT JOIN usage u USING(stay_id)\nGROUP BY l.stay_id,w.baseline_provenance,w.baseline_reference_time,w.baseline_reference_storetime,w.baseline_reference_value,u.max_cr_charttime_used,u.max_cr_storetime_used,u.creatinine_measurement_count_used\n")
    qc = con.execute('SELECT COUNT(*) n,SUM(kdigo_creatinine_stage=0) kdigo_stage_0_n,SUM(kdigo_creatinine_stage=1) kdigo_stage_1_n,SUM(kdigo_creatinine_stage=2) kdigo_stage_2_n,SUM(kdigo_creatinine_stage=3) kdigo_stage_3_n,SUM(kdigo_creatinine_stage IS NULL) kdigo_stage_unknown_n FROM creat_stay').df()
    display_table(qc)
    tq = con.execute('SELECT SUM(c.max_cr_charttime_used>l.landmark_time) chart_after,SUM(c.max_cr_storetime_used>l.landmark_time) store_after,SUM(c.baseline_reference_time>=l.landmark_time) ref_not_before_landmark,SUM(c.baseline_reference_storetime>l.landmark_time) ref_store_after FROM creat_stay c JOIN landmark_eligible l USING(stay_id)').df()
    display_table(tq)
    assert (tq.iloc[0].fillna(0).astype(int) == 0).all()
    OUTPUT = CREATININE_DIR / 'kdigo_creatinine.parquet'
    con.execute(f"COPY creat_stay TO '{qpath(OUTPUT)}' (FORMAT PARQUET)")
    qc.to_csv(PHENOTYPE_QC_DIR / 'creatinine_summary.csv', index=False)
    print('Saved:', OUTPUT)
    print('KDIGO CREATININE COMPONENT: PASS')


if __name__ == "__main__":
    main()
