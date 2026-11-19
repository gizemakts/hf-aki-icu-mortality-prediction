#!/usr/bin/env python3
"""Isolated-ultrafiltration RRT-suppression sensitivity analysis."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT, TEMP_ROOT, display_table
from helpers._fingerprint import analysis_fingerprint, fingerprinted_resume_dir


def main() -> None:
    REPOSITORY_ROOT = _REPOSITORY_ROOT
    RAW_ROOT = RAW_DATA_ROOT
    OUT_DIR = REPOSITORY_ROOT / 'outputs' / '06_sensitivity' / 'isolated_uf'
    QC_DIR = REPOSITORY_ROOT / 'quality_control' / 'sensitivity' / 'isolated_uf'
    PROGRESS_ROOT = TEMP_ROOT / 'isolated_uf_sensitivity'
    MODEL_DIR = REPOSITORY_ROOT / 'outputs' / '04_modeling'
    VALIDATION_DIR = REPOSITORY_ROOT / 'outputs' / '05_validation' / 'internal_validation'
    PHENOTYPE_SENS_DIR = REPOSITORY_ROOT / 'outputs' / '06_sensitivity' / 'phenotype'
    POPULATION_DIR = REPOSITORY_ROOT / 'outputs' / '06_sensitivity' / 'populations'
    PREDICTOR_DIR = REPOSITORY_ROOT / 'outputs' / '03_predictors'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    UF_OUTPUT_ITEMID = 226457
    ARM = 'isolated_uf_rrt_suppression'
    OUTCOME = 'hospital_expire_flag'
    EXPECTED_N = 3457
    EXPECTED_EVENTS = 707
    N_IMPUTATIONS = 30
    MICE_CYCLES = 10
    PMM_DONORS = 5
    N_BOOTSTRAP = 500
    LOWESS_FRAC = 0.3
    FULLFIT_SEED = 202610010401
    BOOT_INDEX_BASE_SEED = 202610010402
    BOOT_MI_BASE_SEED = 202610010403
    PRIMARY_PREDICTORS = [
        'age_per10y', 'sex_female', 'heart_rate_max_per10', 'map_min_per10',
        'resp_rate_max_per5', 'temperature_max_abs_deviation_from_37c',
        'spo2_deficit_per5', 'gcs_deficit_from15', 'hemoglobin_min_gdl',
        'wbc_max_log2', 'platelet_min_log2', 'bun_max_log2',
        'bicarbonate_min_per5', 'sodium_max_abs_deviation_from140_per5',
        'potassium_max_mmol_l', 'glucose_max_log2', 'vasoactive_any_by_landmark',
        'kdigo_stage_2_indicator', 'kdigo_stage_3_indicator',
    ]

    PRIMARY_MATRIX = PREDICTOR_DIR / 'primary_predictor_matrix.parquet'
    PRIMARY_POPULATION = POPULATION_DIR / '01_primary.parquet'
    MODEL_PREDICTIONS = MODEL_DIR / 'apparent_predictions.csv'
    MODEL_COEFFICIENTS = MODEL_DIR / 'primary_pooled_coefficients.csv'
    VALIDATION_SUMMARY = VALIDATION_DIR / 'internal_validation_summary.csv'
    UF_DEFINITION = PREDICTOR_DIR / 'isolated_uf_sensitivity_definition.json'
    PHENOTYPE_SENS_MANIFEST = PHENOTYPE_SENS_DIR / 'phenotype_sensitivity_manifest.json'
    for path in [PRIMARY_MATRIX, PRIMARY_POPULATION, MODEL_PREDICTIONS, MODEL_COEFFICIENTS, VALIDATION_SUMMARY, UF_DEFINITION, PHENOTYPE_SENS_MANIFEST]:
        if not path.is_file():
            raise FileNotFoundError(path)
    uf_contract = json.loads(UF_DEFINITION.read_text(encoding='utf-8'))
    phenotype_sens_manifest = json.loads(PHENOTYPE_SENS_MANIFEST.read_text(encoding='utf-8'))
    if phenotype_sens_manifest.get('status') != 'PASS' or phenotype_sens_manifest.get('alternative_arms_completely_refit') != 11:
        raise AssertionError('Phenotype sensitivity refits are incomplete.')
    if uf_contract.get('direct_scuf_target_stays') != 0:
        raise AssertionError('Direct SCUF target count must remain zero.')
    if uf_contract.get('isolated_226457_uf_output_only_target_stays') != '<11':
        raise AssertionError('Isolated ultrafiltration cell must be disclosed only as <11.')
    if uf_contract.get('selected_kdigo_stage_changes_when_suppressed') != '<11':
        raise AssertionError('Selected KDIGO-stage change cell must be disclosed only as <11.')
    if uf_contract.get('membership_or_timezero_change_when_suppressed') is not False:
        raise AssertionError('Isolated-UF suppression must not change membership or time zero.')

    def find_mimic_file(rel):
        path = RAW_ROOT / Path(rel)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def atomic_csv(df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    primary_matrix = pd.read_parquet(PRIMARY_MATRIX).reset_index(drop=True)
    expected_cols = ['subject_id', 'stay_id'] + PRIMARY_PREDICTORS
    if primary_matrix.columns.tolist() != expected_cols:
        raise AssertionError('Primary predictor matrix schema mismatch.')
    if len(primary_matrix) != EXPECTED_N or primary_matrix['stay_id'].duplicated().any():
        raise AssertionError('Primary predictor matrix identity check failed.')
    ph = pd.read_parquet(PRIMARY_POPULATION).copy()
    if len(ph) != EXPECTED_N:
        raise AssertionError('Primary phenotype population size mismatch.')
    for c in ['subject_id', 'hadm_id', 'stay_id']:
        ph[c] = pd.to_numeric(ph[c], errors='raise').astype('int64')
    ph['intime'] = pd.to_datetime(ph['intime'])
    ph['landmark_time'] = pd.to_datetime(ph['landmark_time'])
    if ph['stay_id'].duplicated().any() or ph['subject_id'].duplicated().any():
        raise AssertionError('Primary phenotype identity uniqueness failure.')
    if set(ph['stay_id']) != set(primary_matrix['stay_id'].astype('int64')):
        raise AssertionError('Primary phenotype membership differs from the primary predictor matrix.')
    yref = pd.read_csv(MODEL_PREDICTIONS)
    ymap = yref.set_index('stay_id')[OUTCOME]
    model_df = primary_matrix.copy()
    model_df[OUTCOME] = model_df['stay_id'].map(ymap)
    if model_df[OUTCOME].isna().any():
        raise AssertionError('Missing outcome after primary-model outcome attachment.')
    model_df[OUTCOME] = pd.to_numeric(model_df[OUTCOME], errors='raise').astype(int)
    if int(model_df[OUTCOME].sum()) != EXPECTED_EVENTS:
        raise AssertionError('Primary event count must equal 707.')
    s2 = pd.to_numeric(model_df['kdigo_stage_2_indicator'], errors='raise').astype(int)
    s3 = pd.to_numeric(model_df['kdigo_stage_3_indicator'], errors='raise').astype(int)
    if ((s2 == 1) & (s3 == 1)).any():
        raise AssertionError('Invalid KDIGO indicator combination in the primary model matrix.')
    harmonized_model = pd.Series(np.where(s3.eq(1), 3, np.where(s2.eq(1), 2, 1)), index=model_df.index)
    pstage = pd.to_numeric(ph.set_index('stay_id')['combined_stage_variant'].reindex(model_df['stay_id']).reset_index(drop=True), errors='raise').astype(int)
    if not pstage.eq(harmonized_model).all():
        raise AssertionError('Primary model KDIGO indicators disagree with the phenotype stage.')
    cr = pd.to_numeric(ph['primary_creatinine_stage'], errors='coerce')
    uo = pd.to_numeric(ph['uo_stage_mean_ibw_elapsed'], errors='coerce')
    no_rrt_stage = pd.concat([cr, uo], axis=1).max(axis=1, skipna=True)
    active_rrt = pd.to_numeric(ph['active_rrt_any_by_landmark'], errors='coerce').fillna(0).astype(int)
    combined = pd.to_numeric(ph['combined_stage_variant'], errors='raise').astype(int)
    nonrrt = active_rrt.eq(0)
    if no_rrt_stage.loc[nonrrt].isna().any():
        raise AssertionError('Non-RRT primary rows contain unresolved no-RRT stage.')
    if not combined.loc[nonrrt].eq(no_rrt_stage.loc[nonrrt].astype(int)).all():
        bad = ph.loc[nonrrt & ~combined.eq(no_rrt_stage.fillna(-99).astype(int)), ['stay_id']].head(20)
        raise AssertionError(f'Component-stage reconstruction failed in non-RRT rows:\n{bad}')
    if not combined.loc[active_rrt.eq(1)].eq(3).all():
        raise AssertionError('Active-RRT rows are not all KDIGO Stage 3.')
    CHARTEVENTS = find_mimic_file(Path('icu') / 'chartevents.csv.gz')
    INPUTEVENTS = find_mimic_file(Path('icu') / 'inputevents.csv.gz')
    PROCEDUREEVENTS = find_mimic_file(Path('icu') / 'procedureevents.csv.gz')
    D_ITEMS = find_mimic_file(Path('icu') / 'd_items.csv.gz')
    ACTIVE_CE_OTHER = [226499, 224154, 225183, 227438, 224191, 225806, 225807, 228004, 228005, 228006, 224144, 224145, 224153]
    ACTIVE_IE_OTHER = [227536, 227525]
    ACTIVE_PE_OTHER = [225441, 225802, 225803, 225805, 225809, 225955]
    ditem = pd.read_csv(D_ITEMS, usecols=['itemid', 'label', 'linksto'], low_memory=False)
    ditem['itemid'] = pd.to_numeric(ditem['itemid'], errors='coerce')
    d226457 = ditem.loc[ditem['itemid'].eq(UF_OUTPUT_ITEMID)].copy()
    if len(d226457) != 1:
        raise AssertionError(f'd_items does not contain exactly one row for itemid {UF_OUTPUT_ITEMID}.')
    raw_label = str(d226457.iloc[0]['label'])
    raw_linksto = str(d226457.iloc[0]['linksto']).strip().lower()
    if raw_linksto != 'chartevents':
        raise AssertionError(f"itemid {UF_OUTPUT_ITEMID} linksto={raw_linksto!r}, expected 'chartevents'.")
    if 'ultrafiltrate' not in raw_label.lower():
        raise AssertionError(f'itemid {UF_OUTPUT_ITEMID} unexpected label={raw_label!r}.')
    print(f'MIMIC d_items source check: {UF_OUTPUT_ITEMID} = {raw_label!r}, linksto={raw_linksto!r} — PASS')
    target_times = ph[['stay_id', 'intime', 'landmark_time']].copy()
    con = duckdb.connect(database=':memory:')
    con.register('target_times_df', target_times)
    con.execute('\n    CREATE OR REPLACE TEMP TABLE target_times AS\n    SELECT\n        CAST(stay_id AS BIGINT) stay_id,\n        CAST(intime AS TIMESTAMP) intime,\n        CAST(landmark_time AS TIMESTAMP) landmark_time\n    FROM target_times_df\n')
    ce_ids = [UF_OUTPUT_ITEMID] + ACTIVE_CE_OTHER + [225965]
    q_ce = f"""\nSELECT\n    CAST(c.stay_id AS BIGINT) AS stay_id,\n\n    MAX(\n        CASE\n            WHEN CAST(c.itemid AS BIGINT) = {UF_OUTPUT_ITEMID}\n                 AND c.value IS NOT NULL\n            THEN 1 ELSE 0\n        END\n    ) AS has_226457,\n\n    MAX(\n        CASE\n            WHEN CAST(c.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_CE_OTHER))})\n                 AND c.value IS NOT NULL\n            THEN 1\n            WHEN CAST(c.itemid AS BIGINT) = 225965\n                 AND c.value = 'In use'\n            THEN 1\n            ELSE 0\n        END\n    ) AS has_other_active_ce,\n\n    SUM(\n        CASE\n            WHEN CAST(c.itemid AS BIGINT) = {UF_OUTPUT_ITEMID}\n                 AND c.value IS NOT NULL\n            THEN 1 ELSE 0\n        END\n    ) AS uf226457_rows,\n\n    MIN(\n        CASE\n            WHEN CAST(c.itemid AS BIGINT) = {UF_OUTPUT_ITEMID}\n            THEN TRY_CAST(c.valuenum AS DOUBLE)\n        END\n    ) AS uf226457_min_valuenum,\n\n    MAX(\n        CASE\n            WHEN CAST(c.itemid AS BIGINT) = {UF_OUTPUT_ITEMID}\n            THEN TRY_CAST(c.valuenum AS DOUBLE)\n        END\n    ) AS uf226457_max_valuenum,\n\n    COUNT(\n        DISTINCT CASE\n            WHEN CAST(c.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_CE_OTHER))})\n                 AND c.value IS NOT NULL\n            THEN CAST(c.itemid AS BIGINT)\n            WHEN CAST(c.itemid AS BIGINT) = 225965\n                 AND c.value = 'In use'\n            THEN CAST(c.itemid AS BIGINT)\n        END\n    ) AS other_active_ce_itemid_count\n\nFROM read_csv_auto(\n    '{str(CHARTEVENTS).replace("'", "''")}',\n    header=true,\n    sample_size=200000,\n    ignore_errors=true\n) c\nJOIN target_times t\n  ON CAST(c.stay_id AS BIGINT)=t.stay_id\nWHERE CAST(c.itemid AS BIGINT) IN ({','.join(map(str, ce_ids))})\n  AND TRY_CAST(c.charttime AS TIMESTAMP) >= t.intime\n  AND TRY_CAST(c.charttime AS TIMESTAMP) <= t.landmark_time\nGROUP BY CAST(c.stay_id AS BIGINT)\n"""
    ce = con.execute(q_ce).df()
    q_ie = f"""\nSELECT\n    CAST(i.stay_id AS BIGINT) AS stay_id,\n    MAX(\n        CASE\n            WHEN CAST(i.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_IE_OTHER))})\n                 AND TRY_CAST(i.amount AS DOUBLE) > 0\n            THEN 1 ELSE 0\n        END\n    ) AS has_other_active_ie,\n    COUNT(\n        DISTINCT CASE\n            WHEN CAST(i.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_IE_OTHER))})\n                 AND TRY_CAST(i.amount AS DOUBLE) > 0\n            THEN CAST(i.itemid AS BIGINT)\n        END\n    ) AS other_active_ie_itemid_count\nFROM read_csv_auto(\n    '{str(INPUTEVENTS).replace("'", "''")}',\n    header=true,\n    sample_size=200000,\n    ignore_errors=true\n) i\nJOIN target_times t\n  ON CAST(i.stay_id AS BIGINT)=t.stay_id\nWHERE CAST(i.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_IE_OTHER))})\n  AND TRY_CAST(i.amount AS DOUBLE) > 0\n  AND TRY_CAST(i.starttime AS TIMESTAMP) <= t.landmark_time\n  AND COALESCE(\n        TRY_CAST(i.endtime AS TIMESTAMP),\n        TRY_CAST(i.starttime AS TIMESTAMP)\n      ) >= t.intime\nGROUP BY CAST(i.stay_id AS BIGINT)\n"""
    ie = con.execute(q_ie).df()
    q_pe = f"""\nSELECT\n    CAST(p.stay_id AS BIGINT) AS stay_id,\n    MAX(\n        CASE\n            WHEN CAST(p.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_PE_OTHER))})\n                 AND p.value IS NOT NULL\n            THEN 1 ELSE 0\n        END\n    ) AS has_other_active_pe,\n    MAX(\n        CASE\n            WHEN CAST(p.itemid AS BIGINT) = 225955\n                 AND p.value IS NOT NULL\n            THEN 1 ELSE 0\n        END\n    ) AS has_direct_scuf,\n    COUNT(\n        DISTINCT CASE\n            WHEN CAST(p.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_PE_OTHER))})\n                 AND p.value IS NOT NULL\n            THEN CAST(p.itemid AS BIGINT)\n        END\n    ) AS other_active_pe_itemid_count\nFROM read_csv_auto(\n    '{str(PROCEDUREEVENTS).replace("'", "''")}',\n    header=true,\n    sample_size=200000,\n    ignore_errors=true\n) p\nJOIN target_times t\n  ON CAST(p.stay_id AS BIGINT)=t.stay_id\nWHERE CAST(p.itemid AS BIGINT) IN ({','.join(map(str, ACTIVE_PE_OTHER))})\n  AND p.value IS NOT NULL\n  AND TRY_CAST(p.starttime AS TIMESTAMP) <= t.landmark_time\n  AND COALESCE(\n        TRY_CAST(p.endtime AS TIMESTAMP),\n        TRY_CAST(p.starttime AS TIMESTAMP)\n      ) >= t.intime\nGROUP BY CAST(p.stay_id AS BIGINT)\n"""
    pe = con.execute(q_pe).df()
    con.close()
    for d in [ce, ie, pe]:
        if len(d):
            d['stay_id'] = pd.to_numeric(d['stay_id'], errors='raise').astype('int64')
    rrt_src = ph[['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time', 'primary_creatinine_stage', 'uo_stage_mean_ibw_elapsed', 'active_rrt_any_by_landmark', 'combined_stage_variant']].copy()
    rrt_src = rrt_src.merge(ce, on='stay_id', how='left', validate='1:1').merge(ie, on='stay_id', how='left', validate='1:1').merge(pe, on='stay_id', how='left', validate='1:1')
    flag_cols = ['has_226457', 'has_other_active_ce', 'has_other_active_ie', 'has_other_active_pe', 'has_direct_scuf']
    count_cols = ['uf226457_rows', 'other_active_ce_itemid_count', 'other_active_ie_itemid_count', 'other_active_pe_itemid_count']
    for c in flag_cols + count_cols:
        if c not in rrt_src:
            rrt_src[c] = 0
        rrt_src[c] = pd.to_numeric(rrt_src[c], errors='coerce').fillna(0)
    rrt_src['has_any_other_active_rrt_source'] = rrt_src['has_other_active_ce'].eq(1) | rrt_src['has_other_active_ie'].eq(1) | rrt_src['has_other_active_pe'].eq(1)
    rrt_src['_no_rrt_stage'] = pd.concat([pd.to_numeric(rrt_src['primary_creatinine_stage'], errors='coerce'), pd.to_numeric(rrt_src['uo_stage_mean_ibw_elapsed'], errors='coerce')], axis=1).max(axis=1, skipna=True)
    rrt_src['isolated_226457_output_only'] = rrt_src['has_226457'].eq(1) & ~rrt_src['has_any_other_active_rrt_source']
    rrt_src.to_csv(OUT_DIR / 'rrt_source_isolation_audit.csv', index=False)
    isolated = rrt_src.loc[rrt_src['isolated_226457_output_only']].copy()
    # Do not print the overlapping source counts: subtraction would reveal
    # the suppressed isolated-UF cell in a publicly shared execution log.
    print('Isolated UF-output-only cell: suppressed (<11)')
    if not 0 < len(isolated) < 11:
        diag_path = QC_DIR / 'isolated_uf_count_mismatch.csv'
        rrt_src.loc[rrt_src['has_226457'].eq(1)].sort_values(['has_any_other_active_rrt_source', 'stay_id']).to_csv(diag_path, index=False)
        raise RuntimeError(f"Source-isolated UF cell does not satisfy the restricted small-cell contract. See {diag_path}")
    if isolated['has_direct_scuf'].eq(1).any():
        raise AssertionError('an isolated 226457 UF-output-only target also has direct SCUF evidence.')
    alternative = isolated.loc[pd.to_numeric(isolated['active_rrt_any_by_landmark'], errors='coerce').fillna(0).eq(1) & pd.to_numeric(isolated['combined_stage_variant'], errors='coerce').eq(3) & isolated['_no_rrt_stage'].isin([1, 2])].copy()
    print('Isolated 226457 stays with KDIGO stage changes: suppressed (<11)')
    if len(alternative) != len(isolated):
        diag_path = QC_DIR / 'isolated_uf_stage_change_mismatch.csv'
        isolated.sort_values('stay_id').to_csv(diag_path, index=False)
        raise RuntimeError(f"Isolated UF stage-change subset does not match the source-isolated group. See {diag_path}")
    if not alternative['_no_rrt_stage'].isin([1, 2]).all():
        raise AssertionError('suppressing isolated 226457 would change AKI membership.')
    affected = alternative[['subject_id', 'hadm_id', 'stay_id', 'intime', 'landmark_time', 'primary_creatinine_stage', 'uo_stage_mean_ibw_elapsed', 'active_rrt_any_by_landmark', 'combined_stage_variant', '_no_rrt_stage', 'uf226457_rows', 'uf226457_min_valuenum', 'uf226457_max_valuenum', 'has_other_active_ce', 'has_other_active_ie', 'has_other_active_pe', 'has_direct_scuf']].copy()
    affected = affected.rename(columns={'combined_stage_variant': 'primary_kdigo_stage', '_no_rrt_stage': 'suppressed_kdigo_stage'})
    affected['suppressed_kdigo_stage'] = affected['suppressed_kdigo_stage'].astype(int)
    affected['stage_changed'] = affected['primary_kdigo_stage'].astype(int).ne(affected['suppressed_kdigo_stage'])
    affected['source_table'] = 'chartevents'
    affected['source_itemid'] = UF_OUTPUT_ITEMID
    affected['source_label'] = raw_label
    affected['isolation_definition'] = '226457 present with no other active RRT source by landmark'
    if len(affected) != len(isolated):
        raise AssertionError('Affected isolated-UF stay count differs from source-isolated group.')
    if int(affected['stage_changed'].sum()) != len(affected):
        raise AssertionError('Not every affected source-isolated stay changed KDIGO stage.')
    if not affected['primary_kdigo_stage'].astype(int).eq(3).all():
        raise AssertionError('All isolated-UF target stays must be primary Stage 3.')
    if not affected['suppressed_kdigo_stage'].isin([1, 2]).all():
        raise AssertionError('Suppressed stages must remain KDIGO Stage 1 or 2.')
    if affected[['has_other_active_ce', 'has_other_active_ie', 'has_other_active_pe']].to_numpy().any():
        raise AssertionError('Isolation guard failed: other active RRT source detected.')
    affected.to_csv(OUT_DIR / 'affected_stays.csv', index=False)
    # Restricted patient-level affected rows remain in the authorized output only.
    print('ISOLATED-UF SOURCE VERIFICATION: PASS')
    print('Definition: 226457 UF-output-only; no other active RRT source by landmark')
    print('Affected stay identities and small-cell stage counts: restricted')
    sens = model_df.copy()
    affected_map = affected.set_index('stay_id')['suppressed_kdigo_stage'].to_dict()
    change_mask = sens['stay_id'].isin(affected_map)
    before = sens.loc[change_mask, ['stay_id', 'kdigo_stage_2_indicator', 'kdigo_stage_3_indicator']].copy()
    for idx in sens.index[change_mask]:
        sid = int(sens.at[idx, 'stay_id'])
        st = int(affected_map[sid])
        sens.at[idx, 'kdigo_stage_2_indicator'] = int(st == 2)
        sens.at[idx, 'kdigo_stage_3_indicator'] = 0
    after = sens.loc[change_mask, ['stay_id', 'kdigo_stage_2_indicator', 'kdigo_stage_3_indicator']].copy()
    if int(change_mask.sum()) != len(affected):
        raise AssertionError('Sensitivity matrix changed a different number of rows than source-verified subset.')
    if len(sens) != EXPECTED_N or int(sens[OUTCOME].sum()) != EXPECTED_EVENTS:
        raise AssertionError('Sensitivity N/events changed unexpectedly.')
    if not sens['stay_id'].eq(model_df['stay_id']).all():
        raise AssertionError('Sensitivity membership/order changed.')
    if not sens['subject_id'].eq(model_df['subject_id']).all():
        raise AssertionError('Sensitivity subject identity changed.')
    nonstage = [c for c in PRIMARY_PREDICTORS if c not in ['kdigo_stage_2_indicator', 'kdigo_stage_3_indicator']]
    for c in nonstage:
        a = model_df[c].to_numpy(float)
        b = sens[c].to_numpy(float)
        if not np.isclose(a, b, rtol=0, atol=0, equal_nan=True).all():
            raise AssertionError(f'Non-stage predictor changed unexpectedly: {c}')
    delta = before.merge(after, on='stay_id', suffixes=('_primary', '_suppressed'), validate='1:1')
    delta.to_csv(OUT_DIR / 'kdigo_indicator_changes.csv', index=False)
    # Do not emit patient-level stage changes into public-facing logs.
    print('SENSITIVITY MATRIX: PASS — only source-verified KDIGO-stage rows changed (<11).')

    def _draw_linear_beta(X, y, rng):
        """
        Bayesian-style normal draw around OLS beta with an inverse-chi-square
        residual-variance draw. Uses pinv for numerical robustness.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta_hat = xtx_inv @ X.T @ y
        resid = y - X @ beta_hat
        rank = np.linalg.matrix_rank(X)
        df_resid = max(int(len(y) - rank), 1)
        rss = float(resid @ resid)
        sigma2_hat = max(rss / df_resid, 1e-12)
        chi = max(float(rng.chisquare(df_resid)), 1e-12)
        sigma2_draw = df_resid * sigma2_hat / chi
        cov = sigma2_draw * xtx_inv
        cov = (cov + cov.T) / 2
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.clip(eigval, 0, None)
        cov_psd = eigvec * eigval @ eigvec.T
        beta_draw = rng.multivariate_normal(mean=beta_hat, cov=cov_psd, check_valid='ignore')
        return beta_draw

    def _pmm_update(cur, original, target, model_cols, missing_mask, rng, donors=5):
        obs_mask = ~missing_mask
        other = [c for c in model_cols if c != target]
        X = cur[other].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        X_obs = X[obs_mask.to_numpy()]
        X_mis = X[missing_mask.to_numpy()]
        y_obs = original.loc[obs_mask, target].to_numpy(dtype=float)
        if len(y_obs) < 20:
            raise RuntimeError(f'Too few observed donors for PMM variable {target}.')
        if len(X_mis) == 0:
            return cur
        beta_draw = _draw_linear_beta(X_obs, y_obs, rng)
        pred_obs = X_obs @ beta_draw
        pred_mis = X_mis @ beta_draw
        k = min(int(donors), len(y_obs))
        new_vals = np.empty(len(pred_mis), dtype=float)
        for j, pm in enumerate(pred_mis):
            dist = np.abs(pred_obs - pm)
            donor_idx = np.argpartition(dist, k - 1)[:k]
            chosen = int(rng.choice(donor_idx))
            new_vals[j] = y_obs[chosen]
        cur.loc[missing_mask, target] = new_vals
        return cur

    def mice_pmm_imputations(df, predictors, outcome, m, cycles, donors, seed, verbose=False, return_trace=False):
        """
        Exact primary model stochastic MICE/PMM sequence.

        primary model returned (imputations, trace).  For Part-3 convenience the default
        return is imputations only; return_trace=True exposes the exact trace too.
        This wrapper does not alter any RNG calls.
        """
        cols = predictors + [outcome]
        original = df[cols].copy()
        if original[outcome].isna().any():
            raise AssertionError('Outcome must be fully observed before MICE.')
        missing_masks = {c: original[c].isna() for c in predictors}
        incomplete = [c for c in predictors if missing_masks[c].any()]
        for c in incomplete:
            vals = set(original[c].dropna().unique().tolist())
            is_binary = vals.issubset({0, 1, 0.0, 1.0}) and len(vals) <= 2
            if is_binary:
                raise RuntimeError(f'Binary missing predictor requires logistic imputation: {c}')
        imputations = []
        trace_rows = []
        master = np.random.SeedSequence(seed)
        child_seeds = master.spawn(m)
        for imp_idx in range(m):
            rng = np.random.default_rng(child_seeds[imp_idx])
            cur = original.copy()
            for c in incomplete:
                mask = missing_masks[c]
                donors_obs = original.loc[~mask, c].to_numpy(dtype=float)
                cur.loc[mask, c] = rng.choice(donors_obs, size=int(mask.sum()), replace=True)
            for cycle in range(1, cycles + 1):
                update_order = list(rng.permutation(incomplete))
                for c in update_order:
                    cur = _pmm_update(cur=cur, original=original, target=c, model_cols=cols, missing_mask=missing_masks[c], rng=rng, donors=donors)
                    vals_imp = cur.loc[missing_masks[c], c].to_numpy(dtype=float)
                    trace_rows.append({'imputation': imp_idx + 1, 'cycle': cycle, 'variable': c, 'imputed_n': len(vals_imp), 'imputed_mean': float(np.mean(vals_imp)), 'imputed_sd': float(np.std(vals_imp, ddof=1)) if len(vals_imp) > 1 else 0.0, 'imputed_min': float(np.min(vals_imp)), 'imputed_max': float(np.max(vals_imp))})
            if cur[predictors].isna().any().any():
                raise AssertionError(f'Imputation {imp_idx + 1}: missing values remain.')
            if not np.isfinite(cur[predictors].to_numpy(dtype=float)).all():
                raise AssertionError(f'Imputation {imp_idx + 1}: non-finite values after imputation.')
            imputations.append(cur)
            if verbose and ((imp_idx + 1) % 5 == 0 or imp_idx == 0):
                print(f'  imputation {imp_idx + 1}/{m}')
        trace = pd.DataFrame(trace_rows)
        if return_trace:
            return (imputations, trace)
        return imputations

    def fit_logistic_mi(imputations, predictors, outcome, model_name=None):
        """
        Exact primary model GLM fit sequence.
        Returns fits + MI prediction matrix for Part-3.
        """
        fits = []
        pred_matrix = np.zeros((len(imputations[0]), len(imputations)), dtype=float)
        label = model_name if model_name is not None else 'MODEL'
        for i, dat in enumerate(imputations, start=1):
            X = sm.add_constant(dat[predictors].astype(float), has_constant='add')
            y = dat[outcome].astype(float)
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
            if not getattr(fit, 'converged', True):
                raise RuntimeError(f'{label}: imputation {i} did not converge.')
            params = fit.params
            cov = fit.cov_params()
            if not np.isfinite(params.to_numpy()).all():
                raise RuntimeError(f'{label}: non-finite coefficients in imputation {i}.')
            if not np.isfinite(cov.to_numpy()).all():
                raise RuntimeError(f'{label}: non-finite covariance matrix in imputation {i}.')
            pred = fit.predict(X)
            if np.any(~np.isfinite(pred)) or np.any((pred < 0) | (pred > 1)):
                raise RuntimeError(f'{label}: invalid probabilities in imputation {i}.')
            pred_matrix[:, i - 1] = pred
            fits.append(fit)
        return (fits, pred_matrix)

    def rubin_pool(fits, model_name):
        """
        Exact primary model Rubin pooling calculations.
        """
        terms = list(fits[0].params.index)
        Q = np.vstack([f.params.reindex(terms).to_numpy(dtype=float) for f in fits])
        U = np.stack([f.cov_params().loc[terms, terms].to_numpy(dtype=float) for f in fits])
        m = Q.shape[0]
        qbar = Q.mean(axis=0)
        ubar = U.mean(axis=0)
        B = np.cov(Q, rowvar=False, ddof=1)
        T = ubar + (1 + 1 / m) * B
        se = np.sqrt(np.diag(T))
        udiag = np.diag(ubar)
        bdiag = np.diag(B)
        extra = (1 + 1 / m) * bdiag
        df = np.full(len(terms), np.inf)
        nonzero = extra > 1e-15
        df[nonzero] = (m - 1) * (1 + udiag[nonzero] / extra[nonzero]) ** 2
        crit = np.where(np.isfinite(df), student_t.ppf(0.975, df), 1.959963984540054)
        lo = qbar - crit * se
        hi = qbar + crit * se
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_increase = extra / udiag
            fmi = (relative_increase + 2 / (df + 3)) / (relative_increase + 1)
        return pd.DataFrame({'model': model_name, 'term': terms, 'pooled_coef': qbar, 'pooled_se': se, 'df_rubin': df, 'ci95_low_coef': lo, 'ci95_high_coef': hi, 'odds_ratio': np.exp(qbar), 'ci95_low_or': np.exp(lo), 'ci95_high_or': np.exp(hi), 'fraction_missing_information_approx': fmi})

    def safe_logit(prob, eps=1e-09):
        prob = np.clip(np.asarray(prob, float), eps, 1 - eps)
        return np.log(prob / (1 - prob))

    def calibration_metrics(y, prob):
        y = np.asarray(y, float)
        prob = np.asarray(prob, float)
        lp = safe_logit(prob)
        fi = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(maxiter=200, tol=1e-10)
        X = sm.add_constant(lp, has_constant='add')
        fs = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
        oe = float(y.sum() / prob.sum())
        return {'calibration_intercept': float(fi.params[0]), 'calibration_slope': float(fs.params[1]), 'observed_expected_ratio': oe, 'log_observed_expected_ratio': float(np.log(oe))}

    def all_metrics(y, prob):
        out = {'auroc': float(roc_auc_score(y, prob)), 'brier': float(brier_score_loss(y, prob))}
        out.update(calibration_metrics(y, prob))
        return out

    def nearest_pmm_values(pred_obs, pred_mis, y_obs, rng, k=5):
        pred_obs = np.asarray(pred_obs, float)
        pred_mis = np.asarray(pred_mis, float)
        y_obs = np.asarray(y_obs, float)
        if pred_mis.size == 0:
            return np.empty(0, float)
        order = np.argsort(pred_obs)
        s = pred_obs[order]
        pos = np.searchsorted(s, pred_mis)
        window = min(2 * k + 1, len(s))
        start = np.clip(pos - k, 0, max(len(s) - window, 0))
        cand_pos = start[:, None] + np.arange(window)[None, :]
        cand_idx = order[cand_pos]
        dist = np.abs(pred_obs[cand_idx] - pred_mis[:, None])
        kk = min(k, cand_idx.shape[1])
        local = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        near = np.take_along_axis(cand_idx, local, axis=1)
        pick = rng.integers(0, kk, size=len(pred_mis))
        return y_obs[near[np.arange(len(pred_mis)), pick]]

    def batched_beta_draw(Xobs, yobs, rng):
        xtx = np.einsum('mni,mnj->mij', Xobs, Xobs, optimize=True)
        xty = np.einsum('mni,n->mi', Xobs, yobs, optimize=True)
        inv = np.linalg.pinv(xtx)
        bhat = np.einsum('mij,mj->mi', inv, xty, optimize=True)
        fitted = np.einsum('mni,mi->mn', Xobs, bhat, optimize=True)
        resid = yobs[None, :] - fitted
        rank = np.linalg.matrix_rank(xtx)
        dfres = np.maximum(len(yobs) - rank, 1)
        rss = np.sum(resid * resid, axis=1)
        s2hat = np.maximum(rss / dfres, 1e-12)
        chi = np.maximum(rng.chisquare(dfres), 1e-12)
        s2draw = dfres * s2hat / chi
        cov = inv * s2draw[:, None, None]
        cov = (cov + np.swapaxes(cov, 1, 2)) / 2
        ev, evec = np.linalg.eigh(cov)
        ev = np.clip(ev, 0, None)
        z = rng.standard_normal((Xobs.shape[0], cov.shape[1]))
        return bhat + np.einsum('mij,mj->mi', evec, np.sqrt(ev) * z, optimize=True)

    def nested_mice_train_test(train_df, test_df, predictors, seed):
        cols = predictors + [OUTCOME]
        tr0 = train_df[cols].to_numpy(float)
        te0 = test_df[cols].to_numpy(float)
        pnum = len(predictors)
        ntr = len(train_df)
        nte = len(test_df)
        trmiss = np.isnan(tr0[:, :pnum])
        temiss = np.isnan(te0[:, :pnum])
        incomplete = np.where(trmiss.any(0) | temiss.any(0))[0]
        rng = np.random.default_rng(seed)
        tr = np.repeat(tr0[None, :, :], N_IMPUTATIONS, axis=0)
        te = np.repeat(te0[None, :, :], N_IMPUTATIONS, axis=0)
        for t in incomplete:
            donor = tr0[~trmiss[:, t], t]
            vals = set(np.unique(donor).tolist())
            if vals.issubset({0.0, 1.0}) and len(vals) <= 2:
                raise RuntimeError(f'Binary predictor {predictors[t]} has missing values inside bootstrap.')
            a = np.where(trmiss[:, t])[0]
            b = np.where(temiss[:, t])[0]
            if len(a):
                tr[:, a, t] = rng.choice(donor, size=(N_IMPUTATIONS, len(a)), replace=True)
            if len(b):
                te[:, b, t] = rng.choice(donor, size=(N_IMPUTATIONS, len(b)), replace=True)
        for _ in range(MICE_CYCLES):
            for t in rng.permutation(incomplete):
                obs = np.where(~trmiss[:, t])[0]
                a = np.where(trmiss[:, t])[0]
                b = np.where(temiss[:, t])[0]
                other = [j for j in range(pnum + 1) if j != t]
                Xtr = np.concatenate([np.ones((N_IMPUTATIONS, ntr, 1)), tr[:, :, other]], axis=2)
                Xobs = Xtr[:, obs, :]
                yobs = tr0[obs, t]
                beta = batched_beta_draw(Xobs, yobs, rng)
                predobs = np.einsum('mni,mi->mn', Xobs, beta, optimize=True)
                if len(a):
                    pa = np.einsum('mni,mi->mn', Xtr[:, a, :], beta, optimize=True)
                    for mm in range(N_IMPUTATIONS):
                        tr[mm, a, t] = nearest_pmm_values(predobs[mm], pa[mm], yobs, rng, PMM_DONORS)
                if len(b):
                    Xte = np.concatenate([np.ones((N_IMPUTATIONS, nte, 1)), te[:, :, other]], axis=2)
                    pb = np.einsum('mni,mi->mn', Xte[:, b, :], beta, optimize=True)
                    for mm in range(N_IMPUTATIONS):
                        te[mm, b, t] = nearest_pmm_values(predobs[mm], pb[mm], yobs, rng, PMM_DONORS)
        if np.isnan(tr[:, :, :pnum]).any() or np.isnan(te[:, :, :pnum]).any():
            raise AssertionError('Missing values remain after nested MICE.')
        return (tr, te)

    def fit_predict_mi_arrays(train_mi, test_mi, n_predictors):
        m = train_mi.shape[0]
        ytr = train_mi[0, :, n_predictors].astype(float)
        yte = test_mi[0, :, n_predictors].astype(float)
        ptr = np.zeros(len(ytr))
        pte = np.zeros(len(yte))
        for mm in range(m):
            Xtr = sm.add_constant(train_mi[mm, :, :n_predictors], has_constant='add')
            Xte = sm.add_constant(test_mi[mm, :, :n_predictors], has_constant='add')
            fit = sm.GLM(ytr, Xtr, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
            if not getattr(fit, 'converged', True):
                raise RuntimeError(f'bootstrap MI fit {mm + 1} failed')
            ptr += np.asarray(fit.predict(Xtr), float)
            pte += np.asarray(fit.predict(Xte), float)
        return (ytr, yte, ptr / m, pte / m)
    ARM_DIR = OUT_DIR
    ARM_DIR.mkdir(parents=True, exist_ok=True)

    def boot_indices(rep, n):
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_INDEX_BASE_SEED, int(rep)]))
        return rng.integers(0, n, n)

    def one_bootstrap(rep, df, fixed_prob):
        n = len(df)
        idx = boot_indices(rep, n)
        train = df.iloc[idx].reset_index(drop=True)
        test = df.reset_index(drop=True)
        if train[OUTCOME].nunique() != 2:
            raise RuntimeError('bootstrap has one outcome class')
        seed = int(np.random.SeedSequence([BOOT_MI_BASE_SEED, int(rep)]).generate_state(1, dtype=np.uint64)[0])
        tr, te = nested_mice_train_test(train, test, PRIMARY_PREDICTORS, seed)
        ytr, yte, ptr, pte = fit_predict_mi_arrays(tr, te, len(PRIMARY_PREDICTORS))
        a = all_metrics(ytr, ptr)
        t = all_metrics(yte, pte)
        f = all_metrics(df[OUTCOME].to_numpy(float)[idx], fixed_prob[idx])
        row = {'replicate': rep, 'analysis_arm': ARM, 'bootstrap_unique_rows': len(np.unique(idx)), 'bootstrap_events': int(ytr.sum())}
        for k, v in a.items():
            row['bootstrap_apparent_' + k] = v
        for k, v in t.items():
            row['test_original_' + k] = v
        for k in ['auroc', 'brier', 'calibration_intercept', 'calibration_slope']:
            row['optimism_' + k] = a[k] - t[k]
        row['optimism_log_observed_expected_ratio'] = a['log_observed_expected_ratio'] - t['log_observed_expected_ratio']
        for k, v in f.items():
            row['fixed_fullfit_resample_' + k] = v
        return row

    def run_bootstrap(df, fixed_prob, progress_path):
        progress_path = Path(progress_path)
        if progress_path.exists():
            out = pd.read_csv(progress_path)
            if 'replicate' not in out.columns:
                raise AssertionError('Existing bootstrap progress_path schema is invalid.')
            done = set(out['replicate'].astype(int))
            print(f'resume bootstrap {len(done)}/500')
        else:
            out = pd.DataFrame()
            done = set()
        start = time.time()
        initial = len(done)
        for rep in range(1, N_BOOTSTRAP + 1):
            if rep in done:
                continue
            t0 = time.time()
            row = one_bootstrap(rep, df, fixed_prob)
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True).drop_duplicates('replicate', keep='last').sort_values('replicate')
            atomic_csv(out, progress_path)
            completed = len(out)
            new = max(completed - initial, 1)
            eta = (N_BOOTSTRAP - completed) * ((time.time() - start) / new) / 3600
            print(f'{ARM}: {completed:3d}/500 | rep {(time.time() - t0) / 60:.2f} min | ETA ~{eta:.1f} h')
        if len(out) != N_BOOTSTRAP or set(out['replicate'].astype(int)) != set(range(1, N_BOOTSTRAP + 1)):
            raise AssertionError('Bootstrap incomplete.')
        return out.reset_index(drop=True)

    def summarize_sensitivity(apparent_metrics, boot):
        rows = []
        original = {'auroc': apparent_metrics['auroc'], 'brier': apparent_metrics['brier'], 'calibration_intercept': apparent_metrics['calibration_intercept'], 'calibration_slope': apparent_metrics['calibration_slope'], 'observed_expected_ratio': apparent_metrics['observed_expected_ratio']}
        for metric in ['auroc', 'brier', 'calibration_intercept', 'calibration_slope']:
            opt = boot['optimism_' + metric].to_numpy(float)
            fixed = boot['fixed_fullfit_resample_' + metric].to_numpy(float)
            mo = float(opt.mean())
            lo, hi = np.percentile(fixed, [2.5, 97.5])
            rows.append({'analysis_arm': ARM, 'metric': metric, 'apparent_estimate': original[metric], 'mean_bootstrap_optimism': mo, 'optimism_corrected_estimate': original[metric] - mo, 'optimism_adjusted_ci95_low': float(lo - mo), 'optimism_adjusted_ci95_high': float(hi - mo), 'bootstrap_resamples': N_BOOTSTRAP})
        appoe = original['observed_expected_ratio']
        opt = boot['optimism_log_observed_expected_ratio'].to_numpy(float)
        fixed = boot['fixed_fullfit_resample_log_observed_expected_ratio'].to_numpy(float)
        mo = float(opt.mean())
        lo, hi = np.percentile(fixed, [2.5, 97.5])
        rows.append({'analysis_arm': ARM, 'metric': 'observed_expected_ratio', 'apparent_estimate': appoe, 'mean_bootstrap_optimism': mo, 'optimism_corrected_estimate': float(np.exp(np.log(appoe) - mo)), 'optimism_adjusted_ci95_low': float(np.exp(lo - mo)), 'optimism_adjusted_ci95_high': float(np.exp(hi - mo)), 'bootstrap_resamples': N_BOOTSTRAP})
        return pd.DataFrame(rows)

    def make_calibration(df, prob, summary, png, csv):
        smth = lowess(df[OUTCOME].to_numpy(float), prob, frac=LOWESS_FRAC, it=0, return_sorted=True)
        curve = pd.DataFrame({'predicted_probability': smth[:, 0], 'lowess_observed_probability': smth[:, 1]}).groupby('predicted_probability', as_index=False).mean()
        curve.to_csv(csv, index=False)
        ss = summary.set_index('metric')
        ci = ss.loc['calibration_intercept', 'optimism_corrected_estimate']
        cs = ss.loc['calibration_slope', 'optimism_corrected_estimate']
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], '--', linewidth=1.2, label='Ideal')
        plt.plot(curve.iloc[:, 0], curve.iloc[:, 1], linewidth=2, label='LOWESS apparent')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Predicted in-hospital mortality probability')
        plt.ylabel('Observed in-hospital mortality')
        plt.title(ARM)
        plt.legend(loc='upper left')
        plt.text(0.04, 0.96, f'N={len(df)}, events={int(df[OUTCOME].sum())}\nOptimism-corrected intercept={ci:.3f}\nOptimism-corrected slope={cs:.3f}', transform=plt.gca().transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.15))
        plt.tight_layout()
        plt.savefig(png, dpi=160, bbox_inches='tight')
        plt.close()
    sens = sens.sort_values('stay_id').reset_index(drop=True)
    resume_contract = {
        'resume_schema': 'isolated-uf-refit-v1',
        'model_identifier': ARM,
        'model_family': 'multivariable binary logistic regression, maximum likelihood',
        'predictors': PRIMARY_PREDICTORS,
        'imputations': N_IMPUTATIONS,
        'mice_cycles': MICE_CYCLES,
        'pmm_donors': PMM_DONORS,
        'bootstrap_resamples': N_BOOTSTRAP,
        'seeds': [FULLFIT_SEED, BOOT_INDEX_BASE_SEED, BOOT_MI_BASE_SEED],
        'source_itemid': UF_OUTPUT_ITEMID,
    }
    resume_fingerprint, resume_payload = analysis_fingerprint(
        resume_contract, {'isolated_uf_model_input': (sens, ['stay_id'])}
    )
    PROGRESS_DIR = fingerprinted_resume_dir(PROGRESS_ROOT, resume_fingerprint, resume_payload)
    coef_path = ARM_DIR / 'pooled_coefficients.csv'
    pred_path = ARM_DIR / 'apparent_predictions.csv'
    app_path = ARM_DIR / 'apparent_metrics.csv'
    fullfit_marker = ARM_DIR / 'full_fit_status.json'
    progress_path = PROGRESS_DIR / 'bootstrap_progress.csv'
    summary_path = ARM_DIR / 'internal_validation_summary.csv'
    cal_png = ARM_DIR / 'calibration_lowess.png'
    cal_csv = ARM_DIR / 'calibration_lowess_data.csv'
    complete = ARM_DIR / 'arm_complete.json'
    if fullfit_marker.exists() and coef_path.exists() and pred_path.exists() and app_path.exists():
        fullfit_status = json.loads(fullfit_marker.read_text())
        if fullfit_status.get('resume_fingerprint') != resume_fingerprint:
            raise RuntimeError('Saved isolated-UF full fit does not match current inputs/configuration.')
        pooled = pd.read_csv(coef_path)
        pred_df = pd.read_csv(pred_path)
        app = pd.read_csv(app_path).iloc[0].to_dict()
        if len(pred_df) != EXPECTED_N or not pred_df['stay_id'].eq(sens['stay_id']).all():
            raise AssertionError('Saved full-fit prediction IDs mismatch.')
        fixed_prob = pred_df['apparent_probability_mi_mean'].to_numpy(float)
        print('Reusing completed MI30 full fit.')
    else:
        print('MI30 full sensitivity refit...')
        imps = mice_pmm_imputations(sens, PRIMARY_PREDICTORS, OUTCOME, N_IMPUTATIONS, MICE_CYCLES, PMM_DONORS, FULLFIT_SEED, verbose=True)
        fits, predmat = fit_logistic_mi(imps, PRIMARY_PREDICTORS, OUTCOME, model_name=ARM)
        pooled = rubin_pool(fits, ARM)
        fixed_prob = predmat.mean(axis=1)
        app = all_metrics(sens[OUTCOME].to_numpy(float), fixed_prob)
        pooled.to_csv(coef_path, index=False)
        pd.DataFrame({'subject_id': sens['subject_id'].astype(int), 'stay_id': sens['stay_id'].astype(int), OUTCOME: sens[OUTCOME].astype(int), 'apparent_probability_mi_mean': fixed_prob}).to_csv(pred_path, index=False)
        pd.DataFrame([{'analysis_arm': ARM, 'n': len(sens), 'events': int(sens[OUTCOME].sum()), **app}]).to_csv(app_path, index=False)
        fullfit_marker.write_text(json.dumps({'status': 'PASS', 'analysis_arm': ARM, 'n': len(sens), 'events': int(sens[OUTCOME].sum()), 'mi': N_IMPUTATIONS, 'mice_cycles': MICE_CYCLES, 'pmm_donors': PMM_DONORS, 'fullfit_seed': FULLFIT_SEED, 'resume_fingerprint': resume_fingerprint}, indent=2))
        del imps, fits, predmat
    print('Bootstrap 500 with nested MI30...')
    boot = run_bootstrap(sens, fixed_prob, progress_path)
    boot.to_csv(ARM_DIR / 'bootstrap_replicates.csv', index=False)
    summary = summarize_sensitivity(app, boot)
    summary.to_csv(summary_path, index=False)
    make_calibration(sens, fixed_prob, summary, cal_png, cal_csv)
    complete.write_text(json.dumps({'status': 'PASS', 'analysis_arm': ARM, 'n': len(sens), 'events': int(sens[OUTCOME].sum()), 'affected_stays': len(affected), 'full_refit': True, 'bootstrap_resamples': N_BOOTSTRAP, 'nested_mi': N_IMPUTATIONS, 'resume_fingerprint': resume_fingerprint}, indent=2))
    print('ISOLATED-UF SENSITIVITY MODEL: COMPLETE')
    if not complete.exists():
        raise AssertionError('Sensitivity arm is not complete.')
    arm_complete = json.loads(complete.read_text())
    if arm_complete.get('status') != 'PASS':
        raise AssertionError('Sensitivity completion status is not PASS.')
    if arm_complete.get('resume_fingerprint') != resume_fingerprint:
        raise RuntimeError('Saved isolated-UF completion marker does not match current inputs/configuration.')

    primary_validation = pd.read_csv(VALIDATION_SUMMARY)
    primary_validation.to_csv(OUT_DIR / 'primary_reference_internal_validation.csv', index=False)
    primary_coefficients = pd.read_csv(MODEL_COEFFICIENTS)
    primary_coefficients.to_csv(OUT_DIR / 'primary_reference_pooled_coefficients.csv', index=False)
    if 'model' in primary_validation.columns:
        primary_validation_reference = primary_validation.loc[primary_validation['model'].astype(str).str.lower().eq('primary')].copy()
    else:
        primary_validation_reference = primary_validation.copy()
    reference = primary_validation_reference[['metric', 'optimism_corrected_estimate', 'optimism_adjusted_ci95_low', 'optimism_adjusted_ci95_high']].copy()
    reference = reference.rename(columns={
        'optimism_corrected_estimate': 'primary_optimism_corrected_estimate',
        'optimism_adjusted_ci95_low': 'primary_ci95_low',
        'optimism_adjusted_ci95_high': 'primary_ci95_high',
    })
    comparison = summary.merge(reference, on='metric', how='left', validate='1:1')
    if comparison[['primary_optimism_corrected_estimate', 'primary_ci95_low', 'primary_ci95_high']].isna().any().any():
        raise AssertionError('Primary-versus-isolated-UF comparison is incomplete.')
    comparison['sensitivity_minus_primary'] = comparison['optimism_corrected_estimate'] - comparison['primary_optimism_corrected_estimate']
    comparison.to_csv(OUT_DIR / 'sensitivity_vs_primary.csv', index=False)

    manifest = {
        'status': 'PASS',
        'analysis_arm': ARM,
        'source_table': 'chartevents',
        'source_itemid': UF_OUTPUT_ITEMID,
        'source_label': raw_label,
        'source_verified_affected_stays': '<11',
        'selected_kdigo_stage_changes': '<11',
        'membership_changed': False,
        'timezero_changed': False,
        'primary_n': EXPECTED_N,
        'primary_events': EXPECTED_EVENTS,
        'predictor_specification_changed': False,
        'only_kdigo_stage_indicators_changed_for_source_verified_rows': True,
        'alternative_full_refit': True,
        'number_of_imputations': N_IMPUTATIONS,
        'mice_cycles': MICE_CYCLES,
        'pmm_donors': PMM_DONORS,
        'bootstrap_resamples': N_BOOTSTRAP,
        'imputation_nested_within_bootstrap': True,
        'resume_fingerprint': resume_fingerprint,
    }
    (OUT_DIR / 'isolated_uf_sensitivity_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('ISOLATED-UF RRT-SUPPRESSION SENSITIVITY: PASS')


if __name__ == '__main__':
    main()
