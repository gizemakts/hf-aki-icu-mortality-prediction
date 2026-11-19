#!/usr/bin/env python3
"""Specify the canonical full-KDIGO MIMIC-IV prediction models.

The canonical full-KDIGO models undergo internal validation in MIMIC-IV.
The eICU-CRD evaluation uses a separately fitted creatinine/RRT model.
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
    from datetime import datetime, timezone
    import json, math
    import numpy as np
    import pandas as pd
    CLEAN_ROOT = Path(os.environ['HF_AKI_PROJECT_ROOT']).resolve()
    if not CLEAN_ROOT.is_dir():
        raise FileNotFoundError(CLEAN_ROOT)
    WORK = CLEAN_ROOT / 'outputs' / '03_predictors'
    WORK.mkdir(parents=True, exist_ok=True)
    EXPECTED_N = 3457
    PROHIBITED_TOKENS = ('hospital_expire', 'expire_flag', 'mortality', 'death', 'deathtime', 'survival', 'outcome', 'dod')

    def assert_outcome_blind(df, label):
        bad = [c for c in df.columns if any((t in c.lower() for t in PROHIBITED_TOKENS))]
        if bad:
            raise AssertionError(f'{label}: prohibited outcome-like columns: {bad}')

    P_MAN = WORK / 'predictor_inventory_manifest.json'
    P_MATRIX = WORK / 'full_predictor_matrix.parquet'
    P_PROV = WORK / 'predictor_provenance.csv'
    P_MISS = WORK / 'feature_missingness_distribution.csv'
    for path in (P_MAN, P_MATRIX, P_PROV, P_MISS):
        if not path.is_file():
            raise FileNotFoundError(path)

    inventory_manifest = json.loads(P_MAN.read_text(encoding='utf-8'))
    matrix = pd.read_parquet(P_MATRIX)
    prov = pd.read_csv(P_PROV)
    miss = pd.read_csv(P_MISS)
    assert_outcome_blind(matrix, 'predictor inventory matrix')
    guards = pd.DataFrame([
        {'check': 'inventory_status', 'expected': 'PASS', 'observed': inventory_manifest.get('status'), 'pass': inventory_manifest.get('status') == 'PASS'},
        {'check': 'inventory_outcome_blind', 'expected': True, 'observed': inventory_manifest.get('outcome_blind'), 'pass': inventory_manifest.get('outcome_blind') is True},
        {'check': 'inventory_model_fit', 'expected': False, 'observed': inventory_manifest.get('model_fit_performed'), 'pass': inventory_manifest.get('model_fit_performed') is False},
        {'check': 'primary_n', 'expected': EXPECTED_N, 'observed': len(matrix), 'pass': len(matrix) == EXPECTED_N},
        {'check': 'unique_subjects', 'expected': EXPECTED_N, 'observed': matrix.subject_id.nunique(), 'pass': matrix.subject_id.nunique() == EXPECTED_N},
        {'check': 'unique_stays', 'expected': EXPECTED_N, 'observed': matrix.stay_id.nunique(), 'pass': matrix.stay_id.nunique() == EXPECTED_N},
    ])
    guards.to_csv(WORK / 'model_design_hard_guards.csv', index=False)
    display_table(guards)
    if not bool(guards['pass'].all()):
        raise AssertionError('Model-design predictor inventory guard failed.')
    required = ['subject_id', 'stay_id', 'age', 'sex_female', 'heart_rate_max_0_24h', 'map_min_0_24h', 'resp_rate_max_0_24h', 'temperature_min_0_24h', 'temperature_max_0_24h', 'spo2_min_0_24h', 'gcs_min_0_24h', 'hemoglobin_min_0_24h', 'wbc_max_0_24h', 'platelet_min_0_24h', 'bun_max_0_24h', 'bicarbonate_min_0_24h', 'sodium_min_0_24h', 'sodium_max_0_24h', 'potassium_max_0_24h', 'glucose_max_0_24h', 'lactate_max_0_24h', 'inr_max_0_24h', 'vasopressor_any_by_landmark', 'inotrope_any_by_landmark', 'kdigo_stage']
    missing_cols = [c for c in required if c not in matrix.columns]
    if missing_cols:
        raise KeyError(f'Missing required inventory columns: {missing_cols}')
    d = matrix[required].copy()
    assert_outcome_blind(d, 'model-design source matrix')
    d['age_per10y'] = pd.to_numeric(d.age, errors='coerce').clip(upper=91) / 10.0
    d['heart_rate_max_per10'] = pd.to_numeric(d.heart_rate_max_0_24h, errors='coerce') / 10.0
    d['map_min_per10'] = pd.to_numeric(d.map_min_0_24h, errors='coerce') / 10.0
    d['resp_rate_max_per5'] = pd.to_numeric(d.resp_rate_max_0_24h, errors='coerce') / 5.0
    tmin = pd.to_numeric(d.temperature_min_0_24h, errors='coerce')
    tmax = pd.to_numeric(d.temperature_max_0_24h, errors='coerce')
    d['temperature_max_abs_deviation_from_37c'] = pd.concat([(tmin - 37.0).abs(), (tmax - 37.0).abs()], axis=1).max(axis=1, skipna=False)
    spo2 = pd.to_numeric(d.spo2_min_0_24h, errors='coerce')
    d['spo2_deficit_per5'] = (100.0 - spo2) / 5.0
    gcs = pd.to_numeric(d.gcs_min_0_24h, errors='coerce')
    d['gcs_deficit_from15'] = 15.0 - gcs
    d['hemoglobin_min_gdl'] = pd.to_numeric(d.hemoglobin_min_0_24h, errors='coerce')
    for src, new in [('wbc_max_0_24h', 'wbc_max_log2'), ('platelet_min_0_24h', 'platelet_min_log2'), ('bun_max_0_24h', 'bun_max_log2'), ('glucose_max_0_24h', 'glucose_max_log2'), ('lactate_max_0_24h', 'lactate_max_log2'), ('inr_max_0_24h', 'inr_max_log2')]:
        x = pd.to_numeric(d[src], errors='coerce')
        if (x.dropna() <= 0).any():
            raise AssertionError(f'{src}: nonpositive values incompatible with the prespecified log2 transform.')
        d[new] = np.log2(x)
    d['bicarbonate_min_per5'] = pd.to_numeric(d.bicarbonate_min_0_24h, errors='coerce') / 5.0
    na = pd.to_numeric(d.sodium_min_0_24h, errors='coerce')
    nb = pd.to_numeric(d.sodium_max_0_24h, errors='coerce')
    d['sodium_max_abs_deviation_from140_per5'] = pd.concat([(na - 140.0).abs(), (nb - 140.0).abs()], axis=1).max(axis=1, skipna=False) / 5.0
    d['potassium_max_mmol_l'] = pd.to_numeric(d.potassium_max_0_24h, errors='coerce')
    d['vasoactive_any_by_landmark'] = (pd.to_numeric(d.vasopressor_any_by_landmark, errors='coerce').fillna(0).eq(1) | pd.to_numeric(d.inotrope_any_by_landmark, errors='coerce').fillna(0).eq(1)).astype('int8')
    stage = pd.to_numeric(d.kdigo_stage, errors='raise').astype(int)
    if not set(stage.unique()).issubset({1, 2, 3}):
        raise AssertionError(f'Unexpected KDIGO stages: {sorted(stage.unique())}')
    d['kdigo_stage_2_indicator'] = (stage == 2).astype('int8')
    d['kdigo_stage_3_indicator'] = (stage == 3).astype('int8')
    d['sex_female'] = pd.to_numeric(d.sex_female, errors='raise').astype('int8')
    CORE = ['age_per10y', 'sex_female', 'heart_rate_max_per10', 'map_min_per10', 'resp_rate_max_per5', 'temperature_max_abs_deviation_from_37c', 'spo2_deficit_per5', 'gcs_deficit_from15', 'hemoglobin_min_gdl', 'wbc_max_log2', 'platelet_min_log2', 'bun_max_log2', 'bicarbonate_min_per5', 'sodium_max_abs_deviation_from140_per5', 'potassium_max_mmol_l', 'glucose_max_log2', 'vasoactive_any_by_landmark', 'kdigo_stage_2_indicator', 'kdigo_stage_3_indicator']
    AUGMENT = ['lactate_max_log2', 'inr_max_log2']
    ENHANCED = CORE + AUGMENT
    if len(CORE) != 19 or len(ENHANCED) != 21:
        raise AssertionError((len(CORE), len(ENHANCED)))
    core = d[['subject_id', 'stay_id'] + CORE].copy()
    enh = d[['subject_id', 'stay_id'] + ENHANCED].copy()
    assert_outcome_blind(core, 'core matrix')
    assert_outcome_blind(enh, 'enhanced matrix')
    inventory_ids = set(map(tuple, matrix[['subject_id', 'stay_id']].astype('int64').to_numpy()))
    if set(map(tuple, core[['subject_id', 'stay_id']].astype('int64').to_numpy())) != inventory_ids or set(map(tuple, enh[['subject_id', 'stay_id']].astype('int64').to_numpy())) != inventory_ids:
        raise AssertionError('Transformed matrix membership changed.')
    P_CORE = WORK / 'primary_predictor_matrix.parquet'
    P_ENH = WORK / 'secondary_predictor_matrix.parquet'
    core.to_parquet(P_CORE, index=False)
    enh.to_parquet(P_ENH, index=False)
    print('Core predictors:', len(CORE))
    print('Enhanced predictors:', len(ENHANCED))
    inv_missing = miss.set_index('feature')['missing_pct'].to_dict()
    decisions = [('age', 'PRIMARY_CORE', 'INCLUDE', 'age capped at 91 then scaled per 10 y; known by landmark; portable'), ('sex_female', 'PRIMARY_CORE', 'INCLUDE', 'known by landmark; portable'), ('heart_rate_max_0_24h', 'PRIMARY_CORE', 'INCLUDE', 'rare missingness (<11 observations); portable first-24h physiology'), ('map_min_0_24h', 'PRIMARY_CORE', 'INCLUDE', 'rare missingness (<11 observations); portable first-24h physiology'), ('resp_rate_max_0_24h', 'PRIMARY_CORE', 'INCLUDE', 'rare missingness (<11 observations); portable first-24h physiology'), ('temperature_min_0_24h + temperature_max_0_24h', 'PRIMARY_CORE', 'INCLUDE_DERIVED', 'combined as maximum absolute deviation from 37C; 4.08% missing'), ('spo2_min_0_24h', 'PRIMARY_CORE', 'INCLUDE', 'rare missingness (<11 observations); transformed to deficit from 100%'), ('gcs_min_0_24h', 'PRIMARY_CORE', 'INCLUDE', 'rare missingness (<11 observations); eICU conceptual GCS mapping feasible'), ('hemoglobin_min_0_24h', 'PRIMARY_CORE', 'INCLUDE', '1.30% missing; portable'), ('wbc_max_0_24h', 'PRIMARY_CORE', 'INCLUDE_LOG2', '1.45% missing; skew handled outcome-blindly'), ('platelet_min_0_24h', 'PRIMARY_CORE', 'INCLUDE_LOG2', '1.33% missing; skew handled outcome-blindly'), ('bun_max_0_24h', 'PRIMARY_CORE', 'INCLUDE_LOG2', '0.55% missing; renal/metabolic severity'), ('bicarbonate_min_0_24h', 'PRIMARY_CORE', 'INCLUDE', '0.55% missing; acid-base severity'), ('sodium_min_0_24h + sodium_max_0_24h', 'PRIMARY_CORE', 'INCLUDE_DERIVED', 'maximum absolute deviation from 140; avoids two correlated linear terms'), ('potassium_max_0_24h', 'PRIMARY_CORE', 'INCLUDE', '0.55% missing; hyperkalemic severity in AKI'), ('glucose_max_0_24h', 'PRIMARY_CORE', 'INCLUDE_LOG2', '0.87% missing; stress/metabolic severity'), ('vasopressor_any_by_landmark + inotrope_any_by_landmark', 'PRIMARY_CORE', 'INCLUDE_DERIVED', 'combined vasoactive support indicator; avoids redundant treatment terms'), ('kdigo_stage', 'PRIMARY_CORE', 'INCLUDE_CATEGORICAL_2DF', 'AKI renal-severity phenotype; stage 1 reference'), ('lactate_max_0_24h', 'SECONDARY_ENHANCED_ONLY', 'INCLUDE_LOG2', '25.72% missing; clinically valuable but less complete'), ('inr_max_0_24h', 'SECONDARY_ENHANCED_ONLY', 'INCLUDE_LOG2', '12.12% missing; clinically useful but less complete'), ('albumin_min_0_24h', 'EXCLUDED_PRIMARY', 'EXCLUDE', '70.90% missing'), ('bilirubin_total_max_0_24h', 'EXCLUDED_PRIMARY', 'EXCLUDE', '47.79% missing'), ('hospital_to_icu_hours', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'workflow-dependent; observed -22.8 to 1799.5 h; portability concern'), ('admission_type', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'database/workflow category portability concern'), ('first_careunit', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'site-specific care-unit taxonomy'), ('invasive_vent_any_by_landmark', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'inventory extraction is direct-evidence only and not source-complete vs full MIT-LCP ventilation concept'), ('prior_ckd', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'prior-completed-admission ICD instrument not equivalent to eICU pastHistory'), ('prior_diabetes', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'prior-completed-admission ICD instrument not equivalent to eICU pastHistory'), ('prior_af_or_flutter', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'prior-completed-admission ICD instrument not equivalent to eICU pastHistory'), ('prior_ischemic_heart_disease', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'prior-completed-admission ICD instrument not equivalent to eICU pastHistory'), ('prior_copd_or_chronic_airway_disease', 'EXCLUDED_PRIMARY', 'EXCLUDE', 'prior-completed-admission ICD instrument not equivalent to eICU pastHistory'), ('creatinine_max_0_24h', 'EXCLUDED_PRIMARY', 'EXCLUDE_REDUNDANT', 'avoid duplication with combined KDIGO stage'), ('creatinine_stage', 'EXCLUDED_PRIMARY', 'EXCLUDE_REDUNDANT', 'avoid component/combined-stage redundancy'), ('urine_output_stage', 'EXCLUDED_PRIMARY', 'EXCLUDE_REDUNDANT', 'avoid component/combined-stage redundancy and 38.4% component missingness'), ('rrt_any_by_landmark', 'EXCLUDED_PRIMARY', 'EXCLUDE_REDUNDANT', 'avoid component/combined-stage redundancy; retained for phenotype sensitivity')]
    dec = pd.DataFrame(decisions, columns=['source_feature_or_group', 'model_role', 'decision', 'outcome_blind_rationale'])
    dec.to_csv(WORK / 'predictor_decisions.csv', index=False)
    display_table(dec)
    PLAN_PREVALENCE = 0.2
    PLAN_CS_R2 = 0.06
    TARGET_SHRINKAGE = 0.9
    AVAILABLE_N = EXPECTED_N

    def pmsampsize_binary(rsquared, parameters, prevalence, shrinkage=0.9):
        r2a = float(rsquared)
        p = int(parameters)
        prev = float(prevalence)
        s = float(shrinkage)
        n1 = math.ceil(p / ((s - 1.0) * math.log(1.0 - r2a / s)))
        E1 = n1 * prev
        lnLnull = E1 * math.log(E1 / n1) + (n1 - E1) * math.log(1 - E1 / n1)
        max_r2a = 1 - math.exp(2 * lnLnull / n1)
        nag_r2 = r2a / max_r2a
        s_small = r2a / (r2a + 0.05 * max_r2a)
        n2 = math.ceil(p / ((s_small - 1.0) * math.log(1.0 - r2a / s_small)))
        n3 = math.ceil((1.96 / 0.05) ** 2 * (prev * (1 - prev)))
        required_n = max(n1, n2, n3)
        return {'parameters': p, 'planning_prevalence': prev, 'anticipated_cox_snell_r2': r2a, 'target_shrinkage': s, 'criterion1_n_shrinkage': n1, 'criterion2_n_r2_difference': n2, 'criterion3_n_overall_risk_precision': n3, 'required_n': required_n, 'available_n': AVAILABLE_N, 'margin_n': AVAILABLE_N - required_n, 'passes_available_n': AVAILABLE_N >= required_n, 'implied_max_cox_snell_r2': max_r2a, 'implied_nagelkerke_r2': nag_r2, 'required_epp_at_planning_prevalence': required_n * prev / p}
    budget = pd.DataFrame([{'model': 'PRIMARY_TRANSPORTABLE_CORE', **pmsampsize_binary(PLAN_CS_R2, len(CORE), PLAN_PREVALENCE, TARGET_SHRINKAGE)}, {'model': 'SECONDARY_BIOMARKER_AUGMENTED', **pmsampsize_binary(PLAN_CS_R2, len(ENHANCED), PLAN_PREVALENCE, TARGET_SHRINKAGE)}, {'model': 'ABSOLUTE_MAX_PARAMETER_GUARD_23', **pmsampsize_binary(PLAN_CS_R2, 23, PLAN_PREVALENCE, TARGET_SHRINKAGE)}])
    budget.to_csv(WORK / 'parameter_budget.csv', index=False)
    display_table(budget)
    if not bool(budget.loc[budget.model != 'ABSOLUTE_MAX_PARAMETER_GUARD_23', 'passes_available_n'].all()):
        raise AssertionError('Parameter budget failed for a prespecified model.')
    MAX_PARAMETERS = 23
    policy = {'primary_model_name': 'Transportable Core Model', 'secondary_model_name': 'Biomarker-Augmented Model', 'primary_parameters': len(CORE), 'secondary_parameters': len(ENHANCED), 'absolute_parameter_cap': MAX_PARAMETERS, 'model_family': 'multivariable binary logistic regression (maximum likelihood), no variable selection', 'penalization_primary': 'none; sample-size budget is prespecified and adequate', 'penalization_sensitivity': 'ridge logistic regression with the identical prespecified predictors may be reported as a prespecified modelling sensitivity', 'stepwise_selection': False, 'univariable_screening': False, 'interaction_search': False, 'data_driven_cutpoints': False, 'missing_data_primary': 'multiple imputation by chained equations', 'number_of_imputations': 30, 'continuous_imputation': 'predictive mean matching', 'binary_imputation': 'logistic where needed', 'outcome_in_development_imputation_model': True, 'outcome_used_for_predictor_selection': False, 'imputation_nested_within_internal_validation': True, 'internal_validation': 'bootstrap', 'bootstrap_resamples_primary': 500, 'bootstrap_resamples_secondary': 500, 'random_train_test_split': False, 'continuous_nonlinearity_policy': 'No spline expansion in primary/secondary models. Outcome-blind clinical transforms are prespecified above to preserve <=21 alternative parameters.', 'standardization_required_for_primary_mle': False, 'mandatory_metrics': ['AUROC/C-statistic with 95% CI', 'Brier score with 95% CI', 'calibration intercept', 'calibration slope', 'observed/expected ratio', 'smoothed calibration plot'], 'hosmer_lemeshow_primary_metric': False, 'external_validation_policy': 'Evaluate the separately developed harmonized creatinine/RRT model in eICU-CRD using the fitted MIMIC-IV coefficients and development-derived imputation procedure.'}
    (WORK / 'modeling_policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    sens = pd.DataFrame([('primary', 3457, True), ('uo_consecutive_ibw_elapsed', 2972, True), ('hf_wide', 7006, True), ('prior_only_krt', 3583, True), ('rolling_only_creatinine', 3279, True), ('hierarchical_creatinine', 3405, True), ('hf_recency_3y', 3174, True), ('hf_recency_1y', 2710, True), ('uo_mean_abw_elapsed', 4019, True), ('fixed_first_eligible', 2741, True), ('uo_source_p95_cap', 3365, True), ('uo_no_interval_cap', 3698, True)], columns=['analysis_arm', 'subjects', 'complete_refit'])
    sens['predictor_specification'] = 'identical primary predictor definitions'
    sens['model_pipeline'] = 'same prespecified modeling pipeline; refit within each phenotype arm'
    sens.to_csv(WORK / 'phenotype_sensitivity_contract.csv', index=False)
    rrt_note = {'direct_scuf_target_stays': 0, 'isolated_226457_uf_output_only_target_stays': '<11', 'membership_or_timezero_change_when_suppressed': False, 'selected_kdigo_stage_changes_when_suppressed': '<11', 'required_action': 'If KDIGO stage or RRT-derived stage-3 enters an analysis, repeat relevant analysis with isolated-UF RRT suppression.'}
    (WORK / 'isolated_uf_sensitivity_definition.json').write_text(json.dumps(rrt_note, indent=2), encoding='utf-8')
    manifest = {
        'status': 'PASS',
        'outcome_values_read': False,
        'model_fit_performed': False,
        'primary_n': EXPECTED_N,
        'primary_model': 'Transportable Core Model',
        'primary_parameter_count': len(CORE),
        'secondary_model': 'Biomarker-Augmented Model',
        'secondary_parameter_count': len(ENHANCED),
        'absolute_parameter_cap': MAX_PARAMETERS,
        'parameter_budget_pass_primary': bool(budget.loc[budget.model == 'PRIMARY_TRANSPORTABLE_CORE', 'passes_available_n'].iloc[0]),
        'parameter_budget_pass_secondary': bool(budget.loc[budget.model == 'SECONDARY_BIOMARKER_AUGMENTED', 'passes_available_n'].iloc[0]),
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    (WORK / 'model_design_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('MODEL DESIGN: PASS')


if __name__ == "__main__":
    main()
