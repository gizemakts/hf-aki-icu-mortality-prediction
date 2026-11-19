#!/usr/bin/env python3
"""06 Combined Kdigo.

Combine creatinine urine-output and RRT evidence.

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
    import numpy as np
    import pandas as pd
    import duckdb
    CONFIRMED_MIMIC_ROOT = Path('data/raw/mimic-iv-3.1')
    CONFIRMED_PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', str(_REPOSITORY_ROOT)))
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', CONFIRMED_MIMIC_ROOT))
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', CONFIRMED_PROJECT_ROOT))
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    CREATININE_DIR = PHENOTYPE_DIR / 'creatinine'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    RRT_DIR = PHENOTYPE_DIR / 'rrt'
    COMBINED_DIR = PHENOTYPE_DIR / 'combined'
    PHENOTYPE_QC_DIR = PROJECT_ROOT / 'quality_control' / 'phenotype'
    for d in [COHORT_DIR, PHENOTYPE_DIR, CREATININE_DIR, URINE_OUTPUT_DIR, RRT_DIR, COMBINED_DIR, PHENOTYPE_QC_DIR]:
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
    OUTCOME_TOKENS = ('expire', 'mortality', 'death', 'outcome')

    def assert_outcome_blind(df, label):
        bad = [c for c in df.columns if any((tok in c.lower() for tok in OUTCOME_TOKENS))]
        if bad:
            raise AssertionError(f'{label}: outcome-like fields prohibited: {bad}')
    print('MIMIC_ROOT     :', MIMIC_ROOT)
    print('PROJECT_ROOT   :', PROJECT_ROOT)
    print('URINE_OUTPUT_DIR :', URINE_OUTPUT_DIR)
    print('duckdb         :', duckdb.__version__)
    P1 = require_file(COHORT_DIR / 'landmark_population.parquet')
    P2 = require_file(COHORT_DIR / 'heart_failure_krt_phenotypes.parquet')
    P3 = require_file(CREATININE_DIR / 'kdigo_creatinine.parquet')
    PB = require_file(URINE_OUTPUT_DIR / 'urine_output_event_time_base.parquet')
    PU = require_file(URINE_OUTPUT_DIR / 'urine_output_event_time_variants.parquet')
    PR = require_file(RRT_DIR / 'active_rrt.parquet')
    landmark = pd.read_parquet(P1, columns=['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time'])
    phen = pd.read_parquet(P2)
    cr = pd.read_parquet(P3)
    base = pd.read_parquet(PB)
    uo = pd.read_parquet(PU)
    rrt = pd.read_parquet(PR)
    for df, name in [(landmark, 'landmark'), (phen, 'phen'), (cr, 'cr'), (base, 'base'), (rrt, 'rrt')]:
        assert_unique(df, 'stay_id', name)
        assert_outcome_blind(df, name)
    assert_outcome_blind(uo, 'uo')
    if uo.duplicated(['stay_id', 'variant_id']).any():
        raise AssertionError('duplicate stay_id x variant_id')
    phen_cols = ['stay_id', 'prior_documented_hf', 'prior_hf_within_1y', 'prior_hf_within_3y', 'chronic_krt_prior_evidence', 'chronic_krt_definite', 'current_chronic_krt_state_code', 'current_dialysis_encounter_only', 'chronic_krt_possible_current_code', 'current_eskd_code', 'current_dialysis_dependence_code', 'current_dialysis_encounter_code']
    cr_cols = ['stay_id', 'kdigo_creatinine_stage', 'creatinine_evaluable']
    base_cols = ['stay_id', 'admission_weight_kg', 'height_cm', 'height_source', 'ibw_kg', 'official_like_uo_stage_dynamic_abw', 'official_like_evaluable', 'max_uo_charttime', 'max_uo_storetime']
    rrt_cols = ['stay_id', 'active_rrt_any_by_landmark', 'crrt_mode_by_landmark']
    combined = uo.merge(landmark, on='stay_id', how='left', validate='m:1').merge(phen[phen_cols], on='stay_id', how='left', validate='m:1').merge(cr[cr_cols], on='stay_id', how='left', validate='m:1').merge(base[base_cols], on='stay_id', how='left', validate='m:1').merge(rrt[rrt_cols], on='stay_id', how='left', validate='m:1')

    def add_combined(df, denom, rrt_kind):
        u_stage = pd.to_numeric(df[f'positive_stage_{denom}'], errors='coerce')
        cr_stage = pd.to_numeric(df['kdigo_creatinine_stage'], errors='coerce')
        rrt_col = 'active_rrt_any_by_landmark' if rrt_kind == 'anyrrt' else 'crrt_mode_by_landmark'
        rrt_pos = df[rrt_col].fillna(0).eq(1)
        positive = cr_stage.fillna(-1).ge(1) | u_stage.fillna(-1).ge(1) | rrt_pos
        stage = pd.concat([cr_stage.fillna(-1), u_stage.fillna(-1)], axis=1).max(axis=1)
        stage = stage.where(stage >= 1, np.nan).mask(rrt_pos, 3)
        absence_eval = df[f'absence_evaluable_{denom}'].fillna(0).eq(1)
        cr_eval = df['creatinine_evaluable'].fillna(0).eq(1)
        status = np.where(positive, 'positive', np.where(cr_eval & absence_eval, 'negative', 'indeterminate'))
        df[f'combined_stage_{denom}_{rrt_kind}'] = stage
        df[f'combined_status_{denom}_{rrt_kind}'] = status
    for denom in ('abw', 'ibw'):
        for rk in ('anyrrt', 'crrt'):
            add_combined(combined, denom, rk)
    assert_outcome_blind(combined, 'combined')
    assert not combined.duplicated(['stay_id', 'variant_id']).any()
    rows = []
    rrt_rows = []
    for cid, cc in combined.groupby('variant_id'):
        for denom in ('abw', 'ibw'):
            for rk in ('anyrrt', 'crrt'):
                status_col = f'combined_status_{denom}_{rk}'
                x = cc.loc[cc.prior_documented_hf.eq(1) & cc.chronic_krt_definite.fillna(0).eq(0) & cc[status_col].eq('positive')].sort_values(['subject_id', 'intime', 'stay_id'])
                selected_subjects = x.drop_duplicates('subject_id', keep='first').copy()
                rows.append({'variant_id': cid, 'denominator': denom.upper(), 'rrt_definition': rk, 'qualifying_stays': int(len(x)), 'first_qualifying_subjects': int(len(selected_subjects))})
                if rk == 'anyrrt':
                    crp = selected_subjects.kdigo_creatinine_stage.fillna(0).ge(1)
                    uop = selected_subjects[f'uo_positive_{denom}'].fillna(0).eq(1)
                    rp = selected_subjects.active_rrt_any_by_landmark.fillna(0).eq(1)
                    ro = rp & ~crp & ~uop
                    rrt_rows.append({'variant_id': cid, 'denominator': denom.upper(), 'rrt_only_n': int(ro.sum()), 'rrt_only_current_possible_chronic_krt_n': int((ro & selected_subjects.chronic_krt_possible_current_code.fillna(0).eq(1)).sum()), 'rrt_only_current_chronic_state_n': int((ro & selected_subjects.current_chronic_krt_state_code.fillna(0).eq(1)).sum())})
    counts = pd.DataFrame(rows)
    rrt_qc = pd.DataFrame(rrt_rows)
    display_table(counts)
    display_table(rrt_qc)
    POUT = COMBINED_DIR / 'combined_kdigo_variants.parquet'
    combined.to_parquet(POUT, index=False)
    counts.to_csv(PHENOTYPE_QC_DIR / 'combined_kdigo_variant_counts.csv', index=False)
    rrt_qc.to_csv(PHENOTYPE_QC_DIR / 'rrt_only_summary.csv', index=False)
    print('COMBINED KDIGO VARIANTS: PASS')


if __name__ == "__main__":
    main()
