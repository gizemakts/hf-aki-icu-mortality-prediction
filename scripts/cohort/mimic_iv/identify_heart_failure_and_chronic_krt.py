#!/usr/bin/env python3
"""02 Heart Failure And Chronic Krt.

Define prior heart failure and chronic-KRT exclusion.

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
    COHORT_QC_DIR = PROJECT_ROOT / 'quality_control' / 'cohort'
    for d in [COHORT_DIR, COHORT_QC_DIR]:
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
    print('COHORT_DIR  :', COHORT_DIR)
    print('Python     :', sys.version.split()[0])
    print('pandas     :', pd.__version__)
    print('duckdb     :', duckdb.__version__)
    LANDMARK_PATH = require_file(COHORT_DIR / 'landmark_population.parquet')
    ADM_PATH = require_file(MIMIC_ROOT / 'hosp' / 'admissions.csv.gz')
    DIAG_PATH = require_file(MIMIC_ROOT / 'hosp' / 'diagnoses_icd.csv.gz')
    lm = pd.read_parquet(LANDMARK_PATH)
    require_cols(lm, ['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time'], '01 landmark')
    assert_unique(lm, 'stay_id', '01 landmark')
    adm = pd.read_csv(ADM_PATH, compression='gzip', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'], parse_dates=['admittime', 'dischtime'])
    diag = pd.read_csv(DIAG_PATH, compression='gzip', usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'], dtype={'icd_code': 'string', 'icd_version': 'Int64'})
    diag['icd_norm'] = diag['icd_code'].astype('string').str.upper().str.replace('.', '', regex=False).str.strip()
    diag['hf_direct_code'] = (diag.icd_version == 9) & diag.icd_norm.str.startswith('428', na=False) | (diag.icd_version == 10) & diag.icd_norm.str.startswith('I50', na=False)
    HF9_EXTRA = {'39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493'}
    HF10_EXTRA = {'I110', 'I130', 'I132'}
    diag['hf_expanded_explicit_code'] = diag.hf_direct_code | (diag.icd_version == 9) & diag.icd_norm.isin(HF9_EXTRA) | (diag.icd_version == 10) & diag.icd_norm.isin(HF10_EXTRA)
    diag['eskd_core_code'] = (diag.icd_version == 9) & diag.icd_norm.eq('5856') | (diag.icd_version == 10) & diag.icd_norm.eq('N186')
    diag['dialysis_dependence_code'] = (diag.icd_version == 9) & diag.icd_norm.eq('V4511') | (diag.icd_version == 10) & diag.icd_norm.eq('Z992')
    diag['dialysis_encounter_code'] = (diag.icd_version == 9) & diag.icd_norm.isin(['V560', 'V568']) | (diag.icd_version == 10) & diag.icd_norm.isin(['Z490', 'Z491', 'Z492'])
    dx_adm = diag.groupby(['subject_id', 'hadm_id'], as_index=False).agg(hf_direct_code=('hf_direct_code', 'max'), hf_expanded_explicit_code=('hf_expanded_explicit_code', 'max'), eskd_core_code=('eskd_core_code', 'max'), dialysis_dependence_code=('dialysis_dependence_code', 'max'), dialysis_encounter_code=('dialysis_encounter_code', 'max')).merge(adm, on=['subject_id', 'hadm_id'], how='left', validate='1:1')
    idx = lm[['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time']].merge(adm, on=['subject_id', 'hadm_id'], how='left', validate='m:1')
    if idx.admittime.isna().any():
        raise AssertionError('Missing index admittime.')
    cur = dx_adm[['subject_id', 'hadm_id', 'hf_direct_code', 'hf_expanded_explicit_code', 'eskd_core_code', 'dialysis_dependence_code', 'dialysis_encounter_code']].rename(columns={'hf_direct_code': 'current_hf_sensitivity', 'hf_expanded_explicit_code': 'current_hf_expanded_sensitivity', 'eskd_core_code': 'current_eskd_code', 'dialysis_dependence_code': 'current_dialysis_dependence_code', 'dialysis_encounter_code': 'current_dialysis_encounter_code'})
    out = idx.merge(cur, on=['subject_id', 'hadm_id'], how='left', validate='m:1')
    for c in ['current_hf_sensitivity', 'current_hf_expanded_sensitivity', 'current_eskd_code', 'current_dialysis_dependence_code', 'current_dialysis_encounter_code']:
        out[c] = out[c].fillna(False).astype('int8')
    prior_dx = dx_adm.loc[dx_adm.dischtime.notna()].copy()
    hist = out[['subject_id', 'hadm_id', 'stay_id', 'admittime']].merge(prior_dx, on='subject_id', how='left', suffixes=('__idx', '__hist'))
    hist = hist.loc[hist['hadm_id__hist'].notna() & (hist['hadm_id__hist'] != hist['hadm_id__idx']) & hist.dischtime.notna() & (hist.dischtime <= hist['admittime__idx'])].copy()
    hist['prior_hf_dischtime'] = hist['dischtime'].where(hist['hf_direct_code'].fillna(False))
    hist['prior_expanded_hf_dischtime'] = hist['dischtime'].where(hist['hf_expanded_explicit_code'].fillna(False))
    prior = hist.groupby('stay_id', as_index=False).agg(had_any_completed_prior_admission=('hadm_id__hist', 'count'), prior_documented_hf=('hf_direct_code', 'max'), prior_expanded_explicit_hf=('hf_expanded_explicit_code', 'max'), prior_eskd_history=('eskd_core_code', 'max'), prior_dialysis_dependence_history=('dialysis_dependence_code', 'max'), prior_dialysis_encounter_history=('dialysis_encounter_code', 'max'), n_completed_prior_hf_admissions=('hf_direct_code', 'sum'), latest_prior_hf_dischtime=('prior_hf_dischtime', 'max'), latest_prior_expanded_hf_dischtime=('prior_expanded_hf_dischtime', 'max'))
    out = out.merge(prior, on='stay_id', how='left', validate='1:1')
    for c in ['prior_documented_hf', 'prior_expanded_explicit_hf', 'prior_eskd_history', 'prior_dialysis_dependence_history', 'prior_dialysis_encounter_history']:
        out[c] = out[c].fillna(False).astype('int8')
    out['had_any_completed_prior_admission'] = (pd.to_numeric(out['had_any_completed_prior_admission'], errors='coerce').fillna(0) > 0).astype('int8')
    out['n_completed_prior_hf_admissions'] = pd.to_numeric(out['n_completed_prior_hf_admissions'], errors='coerce').fillna(0).astype('int32')
    out['days_since_latest_prior_hf'] = (out['admittime'] - pd.to_datetime(out['latest_prior_hf_dischtime'])).dt.total_seconds() / 86400.0
    out['prior_hf_within_1y'] = (out.prior_documented_hf.eq(1) & out.days_since_latest_prior_hf.between(0, 365, inclusive='both')).astype('int8')
    out['prior_hf_within_3y'] = (out.prior_documented_hf.eq(1) & out.days_since_latest_prior_hf.between(0, 365.25 * 3, inclusive='both')).astype('int8')
    out['hf_wide_sensitivity'] = (out.prior_documented_hf.eq(1) | out.current_hf_sensitivity.eq(1)).astype('int8')
    out['hf_expanded_sensitivity'] = (out.prior_expanded_explicit_hf.eq(1) | out.current_hf_expanded_sensitivity.eq(1)).astype('int8')
    out['chronic_krt_prior_evidence'] = (out.prior_eskd_history.eq(1) | out.prior_dialysis_dependence_history.eq(1)).astype('int8')
    out['current_chronic_krt_state_code'] = (out.current_eskd_code.eq(1) | out.current_dialysis_dependence_code.eq(1)).astype('int8')
    out['current_dialysis_encounter_only'] = (out.current_chronic_krt_state_code.eq(0) & out.current_dialysis_encounter_code.eq(1)).astype('int8')
    out['chronic_krt_definite'] = (out.chronic_krt_prior_evidence.eq(1) | out.current_chronic_krt_state_code.eq(1)).astype('int8')
    out['chronic_krt_possible_current_code'] = (out.chronic_krt_prior_evidence.eq(0) & (out.current_chronic_krt_state_code.eq(1) | out.current_dialysis_encounter_code.eq(1))).astype('int8')
    out['prior_dialysis_encounter_only'] = (out.chronic_krt_prior_evidence.eq(0) & out.prior_dialysis_encounter_history.eq(1)).astype('int8')
    out['current_hf_only'] = (out.prior_documented_hf.eq(0) & out.current_hf_sensitivity.eq(1)).astype('int8')
    out['prior_explicit_hf'] = out.prior_documented_hf.astype('int8')
    out['prior_known_eskd'] = out.chronic_krt_prior_evidence.astype('int8')
    out['current_eskd_sensitivity'] = (out.current_eskd_code.eq(1) | out.current_dialysis_dependence_code.eq(1)).astype('int8')
    assert_unique(out, 'stay_id', 'heart-failure/KRT phenotype')
    assert len(out) == len(lm)
    assert (out.prior_explicit_hf == out.prior_documented_hf).all()
    assert (out.prior_known_eskd == out.chronic_krt_prior_evidence).all()
    assert ((out.chronic_krt_definite == 0) & (out.current_chronic_krt_state_code == 1)).sum() == 0
    qc = pd.DataFrame({'metric': ['n_landmark', 'no_completed_prior_admission', 'prior_documented_hf', 'prior_hf_within_1y', 'prior_hf_within_3y', 'current_hf_only', 'chronic_krt_prior_evidence', 'current_chronic_krt_state_code', 'current_dialysis_encounter_only', 'chronic_krt_definite'], 'n': [len(out), int((out.had_any_completed_prior_admission == 0).sum()), int(out.prior_documented_hf.sum()), int(out.prior_hf_within_1y.sum()), int(out.prior_hf_within_3y.sum()), int(out.current_hf_only.sum()), int(out.chronic_krt_prior_evidence.sum()), int(out.current_chronic_krt_state_code.sum()), int(out.current_dialysis_encounter_only.sum()), int(out.chronic_krt_definite.sum())]})
    display_table(qc)
    OUTPUT = COHORT_DIR / 'heart_failure_krt_phenotypes.parquet'
    out.to_parquet(OUTPUT, index=False)
    qc.to_csv(COHORT_QC_DIR / 'heart_failure_krt_summary.csv', index=False)
    print('Saved:', OUTPUT)
    print('HEART FAILURE / CHRONIC KRT PHENOTYPE: PASS')


if __name__ == "__main__":
    main()
