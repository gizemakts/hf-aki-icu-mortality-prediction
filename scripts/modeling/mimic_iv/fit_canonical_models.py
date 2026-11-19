#!/usr/bin/env python3
"""Fit the prespecified primary and secondary mortality models."""
from __future__ import annotations

import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT
from helpers._runtime import display_table


def main() -> None:
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))

    try:
        import statsmodels.api as sm
    except Exception as exc:
        raise RuntimeError("statsmodels is required.") from exc
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
    except Exception as exc:
        raise RuntimeError("scikit-learn is required.") from exc
    try:
        from scipy.stats import t as student_t
    except Exception as exc:
        raise RuntimeError("scipy is required.") from exc

    EXPECTED_N = 3457
    EXPECTED_EVENTS = 707
    OUTCOME = "hospital_expire_flag"
    PRIMARY_PREDICTORS = [
        "age_per10y", "sex_female", "heart_rate_max_per10", "map_min_per10",
        "resp_rate_max_per5", "temperature_max_abs_deviation_from_37c",
        "spo2_deficit_per5", "gcs_deficit_from15", "hemoglobin_min_gdl",
        "wbc_max_log2", "platelet_min_log2", "bun_max_log2",
        "bicarbonate_min_per5", "sodium_max_abs_deviation_from140_per5",
        "potassium_max_mmol_l", "glucose_max_log2", "vasoactive_any_by_landmark",
        "kdigo_stage_2_indicator", "kdigo_stage_3_indicator",
    ]
    SECONDARY_PREDICTORS = PRIMARY_PREDICTORS + ["lactate_max_log2", "inr_max_log2"]
    N_IMPUTATIONS = 30
    MICE_CYCLES = 10
    PMM_DONORS = 5
    PRIMARY_SEED = 2026091701
    SECONDARY_SEED = 2026091702

    inputs = _REPOSITORY_ROOT / "outputs" / "03_predictors"
    outcome_path = _REPOSITORY_ROOT / "outputs" / "01_cohort" / "primary_cohort_with_outcome.parquet"
    primary_path = inputs / "primary_predictor_matrix.parquet"
    secondary_path = inputs / "secondary_predictor_matrix.parquet"
    policy_path = inputs / "modeling_policy.json"
    for path in (outcome_path, primary_path, secondary_path, policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    guards = {
        "primary_parameters": 19,
        "secondary_parameters": 21,
        "number_of_imputations": N_IMPUTATIONS,
        "bootstrap_resamples_primary": 500,
        "bootstrap_resamples_secondary": 500,
    }
    for key, expected in guards.items():
        if policy.get(key) != expected:
            raise AssertionError(f"Model policy mismatch for {key}: {policy.get(key)!r} != {expected!r}")
    for key in ("stepwise_selection", "univariable_screening", "interaction_search", "data_driven_cutpoints"):
        if policy.get(key) is not False:
            raise AssertionError(f"Model policy requires {key}=False")

    OUT = _REPOSITORY_ROOT / "outputs" / "04_modeling"
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"{OUT} must be empty before model fitting.")
    OUT.mkdir(parents=True, exist_ok=True)

    primary = pd.read_parquet(primary_path)
    secondary = pd.read_parquet(secondary_path)
    outcomes = pd.read_parquet(outcome_path)
    id_cols = ["subject_id", "stay_id"]
    if primary.columns.tolist() != id_cols + PRIMARY_PREDICTORS:
        raise AssertionError("Primary predictor matrix schema/order mismatch.")
    if secondary.columns.tolist() != id_cols + SECONDARY_PREDICTORS:
        raise AssertionError("Secondary predictor matrix schema/order mismatch.")
    for label, frame in (("primary", primary), ("secondary", secondary)):
        if len(frame) != EXPECTED_N or frame.subject_id.duplicated().any() or frame.stay_id.duplicated().any():
            raise AssertionError(f"{label} predictor matrix cohort guard failed.")
    if OUTCOME not in outcomes.columns or outcomes.stay_id.duplicated().any():
        raise AssertionError("Outcome artifact schema/uniqueness guard failed.")
    if set(primary.stay_id.astype(int)) != set(secondary.stay_id.astype(int)):
        raise AssertionError("Primary and secondary predictor matrices differ in stay membership.")

    out_attach = outcomes[id_cols + [OUTCOME]].copy()
    out_attach[OUTCOME] = pd.to_numeric(out_attach[OUTCOME], errors="coerce")
    out_attach = out_attach[out_attach.stay_id.isin(primary.stay_id)].copy()
    if len(out_attach) != EXPECTED_N or out_attach[OUTCOME].isna().any():
        raise AssertionError("Outcome attachment cohort guard failed.")
    if not set(out_attach[OUTCOME].unique()).issubset({0, 1, 0.0, 1.0, False, True}):
        raise AssertionError("Outcome must be binary 0/1.")

    p = primary.merge(out_attach, on=id_cols, how="left", validate="1:1")
    s = secondary.merge(out_attach, on=id_cols, how="left", validate="1:1")
    for label, frame, predictors in (("primary", p, PRIMARY_PREDICTORS), ("secondary", s, SECONDARY_PREDICTORS)):
        for column in predictors:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
                raise AssertionError(f"{label}: {column} contains infinite values.")
    events = int(p[OUTCOME].sum())
    if events != EXPECTED_EVENTS:
        raise AssertionError(f"Event guard failed: {events} != {EXPECTED_EVENTS}")
    event_rate = float(p[OUTCOME].mean())
    def missingness_table(df, predictors, model_name):
        rows = []
        for c in predictors:
            nonmiss = df[c].dropna()
            uniq = sorted(pd.unique(nonmiss).tolist())
            binary = set(uniq).issubset({0, 1, 0.0, 1.0}) and len(uniq) <= 2
            rows.append({'model': model_name, 'variable': c, 'missing_n': int(df[c].isna().sum()), 'missing_pct': float(df[c].isna().mean() * 100), 'n_unique_nonmissing': int(nonmiss.nunique()), 'binary_nonmissing': bool(binary), 'min': float(nonmiss.min()) if len(nonmiss) else np.nan, 'max': float(nonmiss.max()) if len(nonmiss) else np.nan})
        return pd.DataFrame(rows)
    miss_primary = missingness_table(p, PRIMARY_PREDICTORS, 'PRIMARY')
    miss_secondary = missingness_table(s, SECONDARY_PREDICTORS, 'SECONDARY')
    miss = pd.concat([miss_primary, miss_secondary], ignore_index=True)
    display_table(miss)
    for label, df_, preds in [('PRIMARY', p, PRIMARY_PREDICTORS), ('SECONDARY', s, SECONDARY_PREDICTORS)]:
        for c in preds:
            vals = set(df_[c].dropna().unique().tolist())
            is_binary = vals.issubset({0, 1, 0.0, 1.0}) and len(vals) <= 2
            if is_binary and df_[c].isna().any():
                raise RuntimeError(f'{label} binary predictor {c} has missing values. The model-fit implementation requires the prespecified logistic binary imputer; do not substitute a continuous PMM imputer.')
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

    def mice_pmm_imputations(df, predictors, outcome, m, cycles, donors, seed):
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
            if (imp_idx + 1) % 5 == 0 or imp_idx == 0:
                print(f'Completed imputation {imp_idx + 1}/{m}')
        return (imputations, pd.DataFrame(trace_rows))
    print('PRIMARY MICE / PMM')
    primary_imputations, primary_trace = mice_pmm_imputations(p, predictors=PRIMARY_PREDICTORS, outcome=OUTCOME, m=N_IMPUTATIONS, cycles=MICE_CYCLES, donors=PMM_DONORS, seed=PRIMARY_SEED)
    primary_trace.to_csv(OUT / 'primary_mice_trace.csv', index=False)
    print('\nSECONDARY MICE / PMM')
    secondary_imputations, secondary_trace = mice_pmm_imputations(s, predictors=SECONDARY_PREDICTORS, outcome=OUTCOME, m=N_IMPUTATIONS, cycles=MICE_CYCLES, donors=PMM_DONORS, seed=SECONDARY_SEED)
    secondary_trace.to_csv(OUT / 'secondary_mice_trace.csv', index=False)
    print('\nMULTIPLE IMPUTATION: COMPLETE')
    def fit_logistic_mi(imputations, predictors, outcome, model_name):
        fits = []
        coef_rows = []
        pred_matrix = np.zeros((len(imputations[0]), len(imputations)), dtype=float)
        for i, dat in enumerate(imputations, start=1):
            X = sm.add_constant(dat[predictors].astype(float), has_constant='add')
            y = dat[outcome].astype(float)
            fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
            if not getattr(fit, 'converged', True):
                raise RuntimeError(f'{model_name}: imputation {i} did not converge.')
            params = fit.params
            cov = fit.cov_params()
            if not np.isfinite(params.to_numpy()).all():
                raise RuntimeError(f'{model_name}: non-finite coefficients in imputation {i}.')
            if not np.isfinite(cov.to_numpy()).all():
                raise RuntimeError(f'{model_name}: non-finite covariance matrix in imputation {i}.')
            pred = fit.predict(X)
            if np.any(~np.isfinite(pred)) or np.any((pred < 0) | (pred > 1)):
                raise RuntimeError(f'{model_name}: invalid probabilities in imputation {i}.')
            pred_matrix[:, i - 1] = pred
            fits.append(fit)
            for term in params.index:
                coef_rows.append({'model': model_name, 'imputation': i, 'term': term, 'coef': float(params[term]), 'se_complete_data': float(fit.bse[term])})
            if i % 5 == 0 or i == 1:
                print(f'{model_name}: fitted {i}/{len(imputations)}')
        return (fits, pd.DataFrame(coef_rows), pred_matrix)
    primary_fits, primary_coef_each, primary_pred_matrix = fit_logistic_mi(primary_imputations, PRIMARY_PREDICTORS, OUTCOME, 'PRIMARY')
    secondary_fits, secondary_coef_each, secondary_pred_matrix = fit_logistic_mi(secondary_imputations, SECONDARY_PREDICTORS, OUTCOME, 'SECONDARY')
    primary_coef_each.to_csv(OUT / 'primary_coefficients_by_imputation.csv', index=False)
    secondary_coef_each.to_csv(OUT / 'secondary_coefficients_by_imputation.csv', index=False)
    def rubin_pool(fits, model_name):
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
        out = pd.DataFrame({'model': model_name, 'term': terms, 'pooled_coef': qbar, 'pooled_se': se, 'df_rubin': df, 'ci95_low_coef': lo, 'ci95_high_coef': hi, 'odds_ratio': np.exp(qbar), 'ci95_low_or': np.exp(lo), 'ci95_high_or': np.exp(hi), 'fraction_missing_information_approx': fmi})
        return (out, qbar, T)
    primary_pooled, primary_beta, primary_total_cov = rubin_pool(primary_fits, 'PRIMARY')
    secondary_pooled, secondary_beta, secondary_total_cov = rubin_pool(secondary_fits, 'SECONDARY')
    display_table(primary_pooled)
    display_table(secondary_pooled)
    primary_pooled.to_csv(OUT / 'primary_pooled_coefficients.csv', index=False)
    secondary_pooled.to_csv(OUT / 'secondary_pooled_coefficients.csv', index=False)
    def calibration_metrics(y, p):
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        eps = 1e-09
        p_clip = np.clip(p, eps, 1 - eps)
        lp = np.log(p_clip / (1 - p_clip))
        X_int = np.ones((len(y), 1))
        citl_fit = sm.GLM(y, X_int, family=sm.families.Binomial(), offset=lp).fit(maxiter=200, tol=1e-10)
        X_cal = sm.add_constant(lp, has_constant='add')
        slope_fit = sm.GLM(y, X_cal, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
        return {'calibration_intercept': float(citl_fit.params[0]), 'calibration_slope': float(slope_fit.params[1]), 'calibration_model_intercept': float(slope_fit.params[0]), 'observed_expected_ratio': float(y.sum() / p.sum())}

    def apparent_metrics(df, prediction_matrix, model_name):
        y = df[OUTCOME].to_numpy(dtype=float)
        p_pool = prediction_matrix.mean(axis=1)
        cal = calibration_metrics(y, p_pool)
        metrics = {'model': model_name, 'n': len(y), 'events': int(y.sum()), 'apparent_auroc': float(roc_auc_score(y, p_pool)), 'apparent_brier': float(brier_score_loss(y, p_pool)), **cal, 'note': 'Apparent development-sample estimate; internal validation is performed in the next workflow step'}
        return (metrics, p_pool)
    primary_metrics, primary_p_pool = apparent_metrics(p, primary_pred_matrix, 'PRIMARY')
    secondary_metrics, secondary_p_pool = apparent_metrics(s, secondary_pred_matrix, 'SECONDARY')
    apparent = pd.DataFrame([primary_metrics, secondary_metrics])
    display_table(apparent)
    apparent.to_csv(OUT / 'apparent_performance.csv', index=False)
    pd.DataFrame({'subject_id': p['subject_id'].astype(int), 'stay_id': p['stay_id'].astype(int), OUTCOME: p[OUTCOME].astype(int), 'primary_apparent_probability_mi_mean': primary_p_pool, 'secondary_apparent_probability_mi_mean': secondary_p_pool}).to_csv(OUT / 'apparent_predictions.csv', index=False)
    print('\nIMPORTANT: These performance values are apparent QC values only. Do not report them as internally validated performance.')
    def coefficient_stability_qc(fits, model_name):
        terms = list(fits[0].params.index)
        Q = np.vstack([f.params.reindex(terms).to_numpy(dtype=float) for f in fits])
        return pd.DataFrame({'model': model_name, 'term': terms, 'coef_min_across_imputations': Q.min(axis=0), 'coef_max_across_imputations': Q.max(axis=0), 'coef_sd_across_imputations': Q.std(axis=0, ddof=1), 'all_fits_converged': True})
    stability = pd.concat([coefficient_stability_qc(primary_fits, 'PRIMARY'), coefficient_stability_qc(secondary_fits, 'SECONDARY')], ignore_index=True)
    display_table(stability)

    run_manifest = {
        "status": "PASS",
        "outcome": OUTCOME,
        "n": int(len(p)),
        "events": events,
        "event_rate": event_rate,
        "primary_model": "Transportable Core Model",
        "secondary_model": "Biomarker-Augmented Model",
        "primary_parameters_excluding_intercept": len(PRIMARY_PREDICTORS),
        "secondary_parameters_excluding_intercept": len(SECONDARY_PREDICTORS),
        "model_family": "multivariable binary logistic regression, maximum likelihood",
        "penalization": "none",
        "variable_selection": "none",
        "number_of_imputations": N_IMPUTATIONS,
        "imputation_method_continuous": "chained equations predictive mean matching",
        "pmm_donors": PMM_DONORS,
        "mice_cycles": MICE_CYCLES,
        "outcome_in_imputation_model": True,
        "primary_seed": PRIMARY_SEED,
        "secondary_seed": SECONDARY_SEED,
        "internal_validation_completed": False,
        "bootstrap_resamples_required": 500,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "statsmodels": version("statsmodels"),
            "duckdb": version("duckdb"),
            "pyarrow": version("pyarrow"),
        },
    }
    (OUT / "model_fit_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("MODEL FIT: PASS")


if __name__ == "__main__":
    main()
