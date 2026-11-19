#!/usr/bin/env python3
"""00 Raw Data Check.

Verify software and licensed raw MIMIC inputs.

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
    import json, sys
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
    QUALITY_CONTROL_DIR = PROJECT_ROOT / 'quality_control' / 'raw_data'
    for d in [PROJECT_ROOT, OUTPUT_ROOT, QUALITY_CONTROL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    assert OUTPUT_ROOT != MIMIC_ROOT
    assert MIMIC_ROOT not in OUTPUT_ROOT.parents
    _write_probe = QUALITY_CONTROL_DIR / '.write_probe.tmp'
    try:
        _write_probe.write_text('ok', encoding='utf-8')
        _write_probe.unlink()
    except OSError as e:
        raise OSError(f'Quality-control directory is not writable: {QUALITY_CONTROL_DIR}') from e

    def qpath(p):
        return str(Path(p).resolve()).replace("'", "''")
    con = duckdb.connect(database=':memory:')
    print('MIMIC_ROOT :', MIMIC_ROOT)
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('QUALITY_CONTROL_DIR:', QUALITY_CONTROL_DIR)
    print('Python      :', sys.version.split()[0])
    print('pandas      :', pd.__version__)
    print('numpy       :', np.__version__)
    print('duckdb      :', duckdb.__version__)
    REQUIRED_COLUMNS = {'patients': {'subject_id', 'gender', 'anchor_age', 'anchor_year'}, 'admissions': {'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'}, 'diagnoses_icd': {'subject_id', 'hadm_id', 'icd_code', 'icd_version'}, 'labevents': {'subject_id', 'hadm_id', 'itemid', 'charttime', 'storetime', 'valuenum'}, 'icustays': {'subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime'}, 'chartevents': {'subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum'}, 'outputevents': {'subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'storetime', 'value'}, 'procedureevents': {'subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'endtime', 'storetime'}, 'inputevents': {'subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'endtime', 'storetime'}}
    rels = {'patients': 'hosp/patients.csv.gz', 'admissions': 'hosp/admissions.csv.gz', 'diagnoses_icd': 'hosp/diagnoses_icd.csv.gz', 'labevents': 'hosp/labevents.csv.gz', 'icustays': 'icu/icustays.csv.gz', 'chartevents': 'icu/chartevents.csv.gz', 'outputevents': 'icu/outputevents.csv.gz', 'procedureevents': 'icu/procedureevents.csv.gz', 'inputevents': 'icu/inputevents.csv.gz'}
    rows = []
    for name, rel in rels.items():
        p = MIMIC_ROOT / rel
        cols = set(pd.read_csv(p, nrows=0).columns) if p.exists() else set()
        miss = sorted(REQUIRED_COLUMNS[name] - cols)
        rows.append({'table': name, 'file_exists': p.exists(), 'missing_columns': ', '.join(miss)})
    schema_qc = pd.DataFrame(rows)
    display_table(schema_qc)
    assert schema_qc.file_exists.all()
    assert (schema_qc.missing_columns == '').all()
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
    qc = con.execute('\nSELECT\n  (SELECT COUNT(*)-COUNT(DISTINCT subject_id) FROM patients) AS dup_subject_id,\n  (SELECT COUNT(*)-COUNT(DISTINCT hadm_id) FROM admissions) AS dup_hadm_id,\n  (SELECT COUNT(*)-COUNT(DISTINCT stay_id) FROM icustays) AS dup_stay_id,\n  (SELECT COUNT(*) FROM admissions a LEFT JOIN patients p USING(subject_id)\n   WHERE p.subject_id IS NULL) AS orphan_admission,\n  (SELECT COUNT(*) FROM icustays i LEFT JOIN admissions a USING(hadm_id)\n   WHERE a.hadm_id IS NULL) AS orphan_icu,\n  (SELECT COUNT(*) FROM icustays i JOIN admissions a USING(hadm_id)\n   WHERE i.subject_id<>a.subject_id) AS subject_mismatch,\n  (SELECT COUNT(*) FROM admissions\n   WHERE admittime IS NULL OR dischtime IS NULL) AS admission_missing_time,\n  (SELECT COUNT(*) FROM admissions\n   WHERE admittime IS NOT NULL AND dischtime IS NOT NULL\n     AND admittime>=dischtime) AS admission_nonpositive_duration,\n  (SELECT COUNT(*) FROM icustays\n   WHERE intime IS NULL OR outtime IS NULL) AS icu_missing_time,\n  (SELECT COUNT(*) FROM icustays\n   WHERE intime IS NOT NULL AND outtime IS NOT NULL\n     AND intime>=outtime) AS icu_nonpositive_duration\n').df()
    display_table(qc)
    hard_blockers = ['dup_subject_id', 'dup_hadm_id', 'dup_stay_id', 'orphan_admission', 'orphan_icu', 'subject_mismatch']
    for col in hard_blockers:
        assert int(qc.loc[0, col]) == 0, f'BLOCKER: {col} != 0'
    time_anomalies = ['admission_missing_time', 'admission_nonpositive_duration', 'icu_missing_time', 'icu_nonpositive_duration']
    print('\nRaw timestamp anomaly audit:')
    for col in time_anomalies:
        print(f'{col}: {int(qc.loc[0, col])}')
    print('\nSTATUS: CORE RELATIONAL INTEGRITY — PASS')
    print('Timestamp anomalies are documented and will be explicitly excluded during landmark eligibility.')
    out = QUALITY_CONTROL_DIR / 'raw_environment_summary.json'
    out.write_text(json.dumps({'status': 'PASS', 'schema_qc': schema_qc.to_dict('records'), 'qc': qc.to_dict('records')}, indent=2, default=str), encoding='utf-8')
    print('RAW DATA ENVIRONMENT CHECK: PASS')
    print('Saved:', out)


if __name__ == "__main__":
    main()
