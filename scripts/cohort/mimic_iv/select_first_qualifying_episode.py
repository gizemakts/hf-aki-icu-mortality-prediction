#!/usr/bin/env python3
"""03 First Qualifying Episode.

Select the first qualifying episode per subject.

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
    import json, hashlib
    import pandas as pd
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT)))
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    COMBINED_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype' / 'combined'
    COHORT_QC_DIR = PROJECT_ROOT / 'quality_control' / 'cohort'
    for d in [COHORT_DIR, COMBINED_DIR, COHORT_QC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    def require_file(p):
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p

    def sha256_file(p, block=1024 * 1024):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while True:
                b = f.read(block)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    def assert_unique(df, key, label):
        if df[key].isna().any():
            raise AssertionError(f'{label}: missing {key}')
        if df[key].duplicated().any():
            raise AssertionError(f'{label}: duplicate {key}')

    def assert_outcome_blind(df, label):
        bad = [c for c in df.columns if any((t in c.lower() for t in ('expire', 'mortality', 'death', 'outcome')))]
        if bad:
            raise AssertionError(f'{label}: outcome-like fields present: {bad}')
    PRIMARY_UO_VARIANT = 'event_gap6_full'
    STATUS = 'combined_status_ibw_anyrrt'
    STAGE = 'combined_stage_ibw_anyrrt'
    P = require_file(COMBINED_DIR / 'combined_kdigo_variants.parquet')
    comb = pd.read_parquet(P)
    assert_outcome_blind(comb, 'combined KDIGO variants')
    if comb.duplicated(['stay_id', 'variant_id']).any():
        raise AssertionError('duplicate stay_id x variant_id')
    cand = comb.loc[comb['variant_id'].eq(PRIMARY_UO_VARIANT)].copy()
    if cand.empty:
        raise AssertionError(f'No rows for primary alternative {PRIMARY_UO_VARIANT}')
    required_cols = ['stay_id', 'subject_id', 'intime', 'variant_id', 'prior_documented_hf', 'chronic_krt_definite', 'current_chronic_krt_state_code', STATUS, STAGE]
    missing = [c for c in required_cols if c not in cand.columns]
    if missing:
        raise KeyError(f'Required columns missing from combined KDIGO artifact: {missing}')
    qual = cand.loc[cand['prior_documented_hf'].eq(1) & cand['chronic_krt_definite'].fillna(0).eq(0) & cand[STATUS].eq('positive')].copy()
    qual = qual.sort_values(['subject_id', 'intime', 'stay_id'], kind='mergesort')
    idx = qual.drop_duplicates('subject_id', keep='first').copy()
    assert_unique(idx, 'subject_id', 'reference cohort')
    assert_unique(idx, 'stay_id', 'reference cohort')
    assert_outcome_blind(idx, 'reference cohort')
    if not idx[STAGE].between(1, 3).all():
        bad = idx.loc[~idx[STAGE].between(1, 3), ['stay_id', STAGE]].head(20)
        raise AssertionError('Invalid combined KDIGO stage:\n' + bad.to_string(index=False))
    if not idx['prior_documented_hf'].eq(1).all():
        raise AssertionError('Non-prior-HF stay entered primary cohort')
    if int(idx['chronic_krt_definite'].fillna(0).sum()) != 0:
        raise AssertionError('Definite chronic KRT leaked into reference cohort')
    if int(idx['current_chronic_krt_state_code'].fillna(0).sum()) != 0:
        raise AssertionError('Current chronic-KRT state leaked into reference cohort')
    if not idx['variant_id'].eq(PRIMARY_UO_VARIANT).all():
        raise AssertionError('Non-primary urine-output alternative entered reference cohort')
    OUT = COHORT_DIR / 'reference_cohort_preoutcome.parquet'
    idx.to_parquet(OUT, index=False)
    H = sha256_file(OUT)
    flow = pd.DataFrame([{'step': '01_landmark_eligible_stays', 'n': int(cand['stay_id'].nunique())}, {'step': '02_prior_documented_HF_stays', 'n': int(cand.loc[cand['prior_documented_hf'].eq(1), 'stay_id'].nunique())}, {'step': '03_after_chronic_krt_exclusion_stays', 'n': int(cand.loc[cand['prior_documented_hf'].eq(1) & cand['chronic_krt_definite'].fillna(0).eq(0), 'stay_id'].nunique())}, {'step': '04_combined_KDIGO_positive_qualifying_stays', 'n': int(len(qual))}, {'step': '05_first_qualifying_episode_per_subject', 'n': int(len(idx))}])
    display_table(flow)
    flow.to_csv(COHORT_QC_DIR / 'reference_cohort_flow.csv', index=False)
    manifest = {'status': 'PASS', 'primary_n': int(len(idx)), 'sha256': H, 'primary_definition': {'hf': 'prior documented HF', 'chronic_krt_exclusion': 'prior chronic state OR current 585.6/N18.6/V45.11/Z99.2 chronic-state code; encounter-only V56/Z49 insufficient', 'creatinine': 'KDIGO creatinine component', 'uo': 'event_gap6_full, IBW, direct urinary sources including ureteral stents; GU irrigation excluded from primary', 'rrt': 'any active RRT by landmark', 'episode': 'first qualifying episode'}, 'source_sha256': {P.name: sha256_file(P)}}
    M = COHORT_DIR / 'reference_cohort_manifest.json'
    M.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Saved:', OUT)
    print('SHA256:', H)
    print('Primary N:', len(idx))
    print('FIRST QUALIFYING EPISODE SELECTION: PASS')


if __name__ == "__main__":
    main()
