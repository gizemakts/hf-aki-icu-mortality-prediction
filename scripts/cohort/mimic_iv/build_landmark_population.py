#!/usr/bin/env python3
"""01 Landmark Population.

Build the adult ICU population evaluable at the 24-hour landmark.

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
    import sys
    import pandas as pd
    import numpy as np
    import duckdb
    def _probe_mimic_root(root: Path) -> bool:
        try:
            patients = root / 'hosp' / 'patients.csv.gz'
            icustays = root / 'icu' / 'icustays.csv.gz'
            if not patients.is_file() or not icustays.is_file():
                return False
            with open(patients, 'rb') as f:
                f.read(1)
            with open(icustays, 'rb') as f:
                f.read(1)
            return True
        except OSError:
            return False

    def resolve_mimic_root():
        root_options = [Path(os.environ.get('MIMIC_ROOT', _RAW_DATA_ROOT))]
        checked = []
        for p in root_options:
            try:
                if _probe_mimic_root(p):
                    print(f'MIMIC-IV data found and readable: {p}')
                    return p.resolve()
                checked.append((str(p), 'required files not found/readable'))
            except OSError as e:
                checked.append((str(p), f'OSError: {e}'))
        print('Checked data locations:')
        for p, status in checked:
            print(' -', p, '->', status)
        raise FileNotFoundError('MIMIC-IV data are not readable. Expected default location: data/raw/mimic-iv-3.1')
    MIMIC_ROOT = resolve_mimic_root()
    if os.environ.get('HF_AKI_PROJECT_ROOT'):
        PROJECT_ROOT = Path(os.environ['HF_AKI_PROJECT_ROOT']).resolve()
    else:
        PROJECT_ROOT = _REPOSITORY_ROOT
    OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
    COHORT_DIR = OUTPUT_ROOT / '01_cohort'
    for d in [PROJECT_ROOT, OUTPUT_ROOT, COHORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    assert OUTPUT_ROOT != MIMIC_ROOT
    assert MIMIC_ROOT not in OUTPUT_ROOT.parents
    _write_probe = COHORT_DIR / '.write_probe.tmp'
    try:
        _write_probe.write_text('ok', encoding='utf-8')
        _write_probe.unlink()
    except OSError as e:
        raise OSError(f'Cohort output directory is not writable: {COHORT_DIR}') from e

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")
    con = duckdb.connect(database=':memory:')
    print('MIMIC_ROOT :', MIMIC_ROOT)
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('COHORT_DIR   :', COHORT_DIR)
    print('Python      :', sys.version.split()[0])
    print('pandas      :', pd.__version__)
    print('numpy       :', np.__version__)
    print('duckdb      :', duckdb.__version__)
    RAW_FILES = {'patients': MIMIC_ROOT / 'hosp' / 'patients.csv.gz', 'admissions': MIMIC_ROOT / 'hosp' / 'admissions.csv.gz', 'diagnoses_icd': MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz', 'labevents': MIMIC_ROOT / 'hosp' / 'labevents.csv.gz', 'icustays': MIMIC_ROOT / 'icu' / 'icustays.csv.gz', 'chartevents': MIMIC_ROOT / 'icu' / 'chartevents.csv.gz', 'outputevents': MIMIC_ROOT / 'icu' / 'outputevents.csv.gz', 'procedureevents': MIMIC_ROOT / 'icu' / 'procedureevents.csv.gz', 'inputevents': MIMIC_ROOT / 'icu' / 'inputevents.csv.gz'}

    def register_raw_views():
        missing = [str(p) for p in RAW_FILES.values() if not p.exists()]
        if missing:
            raise FileNotFoundError('Missing required MIMIC files:\n' + '\n'.join(missing))

        def _register_once():
            for name, path in RAW_FILES.items():
                con.execute(f"CREATE OR REPLACE VIEW {name} AS\n                    SELECT * FROM read_csv_auto(\n                        '{qpath(path)}', header=true, sample_size=200000\n                    );")
        _register_once()
    register_raw_views()
    print('Raw MIMIC views registered:', ', '.join(RAW_FILES))
    con.execute("\nCREATE OR REPLACE TEMP TABLE landmark_eligible AS\nSELECT i.subject_id,i.hadm_id,i.stay_id,\n       a.admittime,a.dischtime,a.deathtime,a.hospital_expire_flag,\n       i.intime,i.outtime,\n       i.intime+INTERVAL '24 hours' landmark_time,\n       p.gender,\n       p.anchor_age + EXTRACT(YEAR FROM a.admittime)-p.anchor_year age\nFROM icustays i\nJOIN admissions a USING(hadm_id)\nJOIN patients p ON p.subject_id=i.subject_id\nWHERE i.subject_id=a.subject_id\n  AND i.intime IS NOT NULL AND i.outtime IS NOT NULL\n  AND a.admittime IS NOT NULL AND a.dischtime IS NOT NULL\n  AND i.intime<i.outtime AND a.admittime<a.dischtime\n  AND p.anchor_age + EXTRACT(YEAR FROM a.admittime)-p.anchor_year >=18\n  AND i.outtime>i.intime+INTERVAL '24 hours'\n  AND a.dischtime>i.intime+INTERVAL '24 hours'\n  AND (a.deathtime IS NULL OR a.deathtime>i.intime+INTERVAL '24 hours')\n")
    qc = con.execute('\nSELECT COUNT(*) n,COUNT(DISTINCT stay_id) unique_stays,\n  SUM(CASE WHEN outtime<=landmark_time THEN 1 ELSE 0 END) not_in_icu,\n  SUM(CASE WHEN dischtime<=landmark_time THEN 1 ELSE 0 END) not_in_hospital,\n  SUM(CASE WHEN deathtime IS NOT NULL AND deathtime<=landmark_time THEN 1 ELSE 0 END) dead_at_landmark\nFROM landmark_eligible\n').df()
    display_table(qc)
    assert qc.loc[0, 'n'] == qc.loc[0, 'unique_stays']
    assert int(qc.loc[0, 'not_in_icu']) == 0
    assert int(qc.loc[0, 'not_in_hospital']) == 0
    assert int(qc.loc[0, 'dead_at_landmark']) == 0
    out = COHORT_DIR / 'landmark_population.parquet'
    con.execute(f"COPY landmark_eligible TO '{qpath(out)}' (FORMAT PARQUET)")
    print('Saved:', out)


if __name__ == "__main__":
    main()
