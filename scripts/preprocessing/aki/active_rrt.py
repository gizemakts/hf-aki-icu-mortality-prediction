#!/usr/bin/env python3
"""05 Active Rrt.

Extract active RRT evidence by the landmark.

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
    RRT_DIR = PHENOTYPE_DIR / 'rrt'
    PHENOTYPE_QC_DIR = PROJECT_ROOT / 'quality_control' / 'phenotype'
    for d in [COHORT_DIR, PHENOTYPE_DIR, RRT_DIR, PHENOTYPE_QC_DIR]:
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
    print('RRT_DIR     :', RRT_DIR)
    print('Python     :', sys.version.split()[0])
    print('pandas     :', pd.__version__)
    print('duckdb     :', duckdb.__version__)
    con = duckdb.connect(database=':memory:')
    P1 = require_file(COHORT_DIR / 'landmark_population.parquet')
    PROC = require_file(MIMIC_ROOT / 'icu' / 'procedureevents.csv.gz')
    CE = require_file(MIMIC_ROOT / 'icu' / 'chartevents.csv.gz')
    IE = require_file(MIMIC_ROOT / 'icu' / 'inputevents.csv.gz')
    con.execute(f"CREATE VIEW landmark AS SELECT * FROM read_parquet('{qpath(P1)}')")
    con.execute(f"CREATE VIEW procedureevents AS SELECT * FROM read_csv_auto('{qpath(PROC)}',header=true,sample_size=200000)")
    con.execute(f"CREATE VIEW chartevents AS SELECT * FROM read_csv_auto('{qpath(CE)}',header=true,sample_size=200000)")
    con.execute(f"CREATE VIEW inputevents AS SELECT * FROM read_csv_auto('{qpath(IE)}',header=true,sample_size=200000)")
    RRT_PROC_ACTIVE = (225441, 225802, 225803, 225805, 225809, 225955)
    RRT_INPUT_ACTIVE = (227536, 227525)
    RRT_CHART_ACTIVE = (226499, 224154, 225183, 227438, 224191, 225806, 225807, 228004, 228005, 228006, 224144, 224145, 224153, 226457)
    con.execute(f"\nCREATE OR REPLACE TEMP TABLE rrt_evidence AS\nSELECT l.stay_id,'procedureevents' source,pe.itemid,pe.starttime event_time,pe.storetime,TRY_CAST(pe.value AS DOUBLE) numeric_value\nFROM landmark l JOIN procedureevents pe USING(stay_id)\nWHERE pe.itemid IN {RRT_PROC_ACTIVE} AND pe.value IS NOT NULL AND pe.starttime<=l.landmark_time AND (pe.endtime IS NULL OR pe.endtime>=l.intime) AND pe.storetime IS NOT NULL AND pe.storetime<=l.landmark_time\nUNION ALL\nSELECT l.stay_id,'chartevents_active',ce.itemid,ce.charttime,ce.storetime,ce.valuenum\nFROM landmark l JOIN chartevents ce USING(stay_id)\nWHERE ce.itemid IN {RRT_CHART_ACTIVE} AND ce.charttime>=l.intime AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time AND ce.value IS NOT NULL\nUNION ALL\nSELECT l.stay_id,'pd_in_use',ce.itemid,ce.charttime,ce.storetime,NULL\nFROM landmark l JOIN chartevents ce USING(stay_id)\nWHERE ce.itemid=225965 AND LOWER(TRIM(ce.value))='in use' AND ce.charttime>=l.intime AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time\nUNION ALL\nSELECT l.stay_id,'inputevents_crrt',ie.itemid,ie.starttime,ie.storetime,ie.amount\nFROM landmark l JOIN inputevents ie USING(stay_id)\nWHERE ie.itemid IN {RRT_INPUT_ACTIVE} AND ie.amount>0 AND ie.starttime<=l.landmark_time AND (ie.endtime IS NULL OR ie.endtime>=l.intime) AND ie.storetime IS NOT NULL AND ie.storetime<=l.landmark_time\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE crrt_mode_evidence AS\nSELECT l.stay_id,ce.charttime event_time,ce.storetime\nFROM landmark l JOIN chartevents ce USING(stay_id)\nWHERE ce.itemid=227290 AND ce.value IS NOT NULL AND TRIM(ce.value)<>''\n AND ce.charttime>=l.intime AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time\n")
    con.execute('\nCREATE OR REPLACE TEMP TABLE rrt_stay AS\nSELECT l.stay_id,\n CASE WHEN COUNT(r.stay_id)>0 THEN 1 ELSE 0 END active_rrt_any_by_landmark,\n CASE WHEN COUNT(c.stay_id)>0 THEN 1 ELSE 0 END crrt_mode_by_landmark,\n MIN(r.event_time) first_rrt_time,MAX(r.event_time) max_rrt_time,MAX(r.storetime) max_rrt_storetime\nFROM landmark l LEFT JOIN rrt_evidence r USING(stay_id) LEFT JOIN crrt_mode_evidence c USING(stay_id)\nGROUP BY l.stay_id\n')
    source_qc = con.execute('SELECT source,COUNT(*) evidence_rows,COUNT(DISTINCT stay_id) stays,SUM(CASE WHEN numeric_value=0 THEN 1 ELSE 0 END) zero_numeric_rows FROM rrt_evidence GROUP BY source ORDER BY source').df()
    display_table(source_qc)
    rrt_compare = con.execute('SELECT COUNT(*) n,SUM(active_rrt_any_by_landmark) any_active,SUM(crrt_mode_by_landmark) crrt_mode,SUM(active_rrt_any_by_landmark=1 AND crrt_mode_by_landmark=0) any_not_crrt FROM rrt_stay').df()
    display_table(rrt_compare)
    tq = con.execute('SELECT SUM(r.max_rrt_time>l.landmark_time) rrt_after,\n                       SUM(r.max_rrt_storetime>l.landmark_time) store_after\n                FROM rrt_stay r JOIN landmark l USING(stay_id)').df()
    display_table(tq)
    assert (tq.iloc[0].fillna(0).astype(int) == 0).all()
    rrt_stay = con.execute('SELECT * FROM rrt_stay').df()
    rrt_stay.to_parquet(RRT_DIR / 'active_rrt.parquet', index=False)
    source_qc.to_csv(PHENOTYPE_QC_DIR / 'rrt_source_summary.csv', index=False)
    rrt_compare.to_csv(PHENOTYPE_QC_DIR / 'rrt_modality_summary.csv', index=False)
    print('Saved active-RRT artifact:', RRT_DIR / 'active_rrt.parquet')
    print('STATUS: ACTIVE RRT BY LANDMARK — PASS')


if __name__ == "__main__":
    main()
