#!/usr/bin/env python3
"""Perform nested-bootstrap internal validation for both prespecified models."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import TEMP_ROOT as _TEMP_ROOT
from helpers._runtime import display_table
from helpers._fingerprint import analysis_fingerprint, fingerprinted_resume_dir


def main() -> None:
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
    N_BOOTSTRAP = 500
    BOOTSTRAP_INDEX_SEED = 2026092401
    PRIMARY_BOOT_MI_SEED = 2026092402
    SECONDARY_BOOT_MI_SEED = 2026092403
    LOWESS_FRAC = 0.3

    predictor_dir = _REPOSITORY_ROOT / "outputs" / "03_predictors"
    model_dir = _REPOSITORY_ROOT / "outputs" / "04_modeling"
    OUT = _REPOSITORY_ROOT / "outputs" / "05_validation" / "internal_validation"
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"{OUT} must be empty before internal validation.")
    OUT.mkdir(parents=True, exist_ok=True)

    required = {
        "primary": predictor_dir / "primary_predictor_matrix.parquet",
        "secondary": predictor_dir / "secondary_predictor_matrix.parquet",
        "manifest": model_dir / "model_fit_manifest.json",
        "apparent": model_dir / "apparent_performance.csv",
        "predictions": model_dir / "apparent_predictions.csv",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("n") != EXPECTED_N or manifest.get("events") != EXPECTED_EVENTS:
        raise AssertionError("Model-fit manifest does not match the expected development cohort.")
    if manifest.get("number_of_imputations") != N_IMPUTATIONS or manifest.get("pmm_donors") != PMM_DONORS or manifest.get("mice_cycles") != MICE_CYCLES:
        raise AssertionError("Model-fit imputation settings do not match internal-validation settings.")

    primary = pd.read_parquet(required["primary"])
    secondary = pd.read_parquet(required["secondary"])
    model_fit_predictions = pd.read_csv(required["predictions"])
    model_fit_apparent = pd.read_csv(required["apparent"])
    if primary.columns.tolist() != ["subject_id", "stay_id"] + PRIMARY_PREDICTORS:
        raise AssertionError("Primary predictor matrix schema/order mismatch.")
    if secondary.columns.tolist() != ["subject_id", "stay_id"] + SECONDARY_PREDICTORS:
        raise AssertionError("Secondary predictor matrix schema/order mismatch.")
    needed_prediction_columns = {
        "subject_id", "stay_id", OUTCOME,
        "primary_apparent_probability_mi_mean", "secondary_apparent_probability_mi_mean",
    }
    if not needed_prediction_columns.issubset(model_fit_predictions.columns):
        raise AssertionError("Model-fit prediction artifact is missing required columns.")

    p = primary.merge(
        model_fit_predictions[["stay_id", "subject_id", OUTCOME, "primary_apparent_probability_mi_mean"]],
        on="stay_id", how="left", validate="1:1", suffixes=("", "_model_fit"),
    )
    s = secondary.merge(
        model_fit_predictions[["stay_id", "subject_id", OUTCOME, "secondary_apparent_probability_mi_mean"]],
        on="stay_id", how="left", validate="1:1", suffixes=("", "_model_fit"),
    )
    for label, frame, predictors in (("PRIMARY", p, PRIMARY_PREDICTORS), ("SECONDARY", s, SECONDARY_PREDICTORS)):
        if len(frame) != EXPECTED_N or frame.stay_id.duplicated().any() or frame.subject_id.duplicated().any():
            raise AssertionError(f"{label}: cohort identity guard failed.")
        if not frame.subject_id.eq(frame.subject_id_model_fit).all():
            raise AssertionError(f"{label}: subject identifier mismatch after model-fit merge.")
        frame.drop(columns=["subject_id_model_fit"], inplace=True)
        frame[OUTCOME] = pd.to_numeric(frame[OUTCOME], errors="coerce")
        if frame[OUTCOME].isna().any() or int(frame[OUTCOME].sum()) != EXPECTED_EVENTS:
            raise AssertionError(f"{label}: outcome guard failed.")
        for column in predictors:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any():
                raise AssertionError(f"{label}: infinite values in {column}.")
    p = p.sort_values("stay_id").reset_index(drop=True)
    s = s.sort_values("stay_id").reset_index(drop=True)
    if not p.stay_id.eq(s.stay_id).all() or not p[OUTCOME].eq(s[OUTCOME]).all():
        raise AssertionError("Primary and secondary validation datasets are not aligned.")

    progress_contract = {
        "resume_schema": "internal-validation-v1",
        "model_identifiers": ["PRIMARY", "SECONDARY"],
        "model_family": "multivariable binary logistic regression, maximum likelihood",
        "n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "bootstrap_resamples": N_BOOTSTRAP,
        "imputations": N_IMPUTATIONS,
        "mice_cycles": MICE_CYCLES,
        "pmm_donors": PMM_DONORS,
        "primary_predictors": PRIMARY_PREDICTORS,
        "secondary_predictors": SECONDARY_PREDICTORS,
        "seeds": [BOOTSTRAP_INDEX_SEED, PRIMARY_BOOT_MI_SEED, SECONDARY_BOOT_MI_SEED],
    }
    resume_fingerprint, resume_payload = analysis_fingerprint(
        progress_contract,
        {
            "primary_validation_input": (p, ["stay_id"]),
            "secondary_validation_input": (s, ["stay_id"]),
        },
    )
    runtime = fingerprinted_resume_dir(
        _TEMP_ROOT / "internal_validation", resume_fingerprint, resume_payload
    )
    def safe_logit(prob, eps=1e-09):
        prob = np.clip(np.asarray(prob, float), eps, 1 - eps)
        return np.log(prob / (1 - prob))

    def calibration_metrics(y, prob):
        y = np.asarray(y, float)
        prob = np.asarray(prob, float)
        lp = safe_logit(prob)
        fit_int = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(maxiter=200, tol=1e-10)
        X = sm.add_constant(lp, has_constant='add')
        fit_slope = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
        oe = float(y.sum() / prob.sum())
        return {'calibration_intercept': float(fit_int.params[0]), 'calibration_slope': float(fit_slope.params[1]), 'observed_expected_ratio': oe, 'log_observed_expected_ratio': float(np.log(oe))}

    def all_metrics(y, prob):
        out = {'auroc': float(roc_auc_score(y, prob)), 'brier': float(brier_score_loss(y, prob))}
        out.update(calibration_metrics(y, prob))
        return out

    def deterministic_boot_indices(rep):
        rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_INDEX_SEED, int(rep)]))
        return rng.integers(0, EXPECTED_N, EXPECTED_N)

    def atomic_write_csv(df, path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + '.tmp')
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
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
        chosen = near[np.arange(len(pred_mis)), pick]
        return y_obs[chosen]

    def batched_beta_draw(X_obs, y_obs, rng):
        xtx = np.einsum('mni,mnj->mij', X_obs, X_obs, optimize=True)
        xty = np.einsum('mni,n->mi', X_obs, y_obs, optimize=True)
        inv = np.linalg.pinv(xtx)
        bhat = np.einsum('mij,mj->mi', inv, xty, optimize=True)
        fitted = np.einsum('mni,mi->mn', X_obs, bhat, optimize=True)
        resid = y_obs[None, :] - fitted
        rank = np.linalg.matrix_rank(xtx)
        dfres = np.maximum(len(y_obs) - rank, 1)
        rss = np.sum(resid * resid, axis=1)
        s2hat = np.maximum(rss / dfres, 1e-12)
        chi = np.maximum(rng.chisquare(dfres), 1e-12)
        s2draw = dfres * s2hat / chi
        cov = inv * s2draw[:, None, None]
        cov = (cov + np.swapaxes(cov, 1, 2)) / 2
        ev, evec = np.linalg.eigh(cov)
        ev = np.clip(ev, 0, None)
        z = rng.standard_normal((X_obs.shape[0], cov.shape[1]))
        delta = np.einsum('mij,mj->mi', evec, np.sqrt(ev) * z, optimize=True)
        return bhat + delta

    def nested_mice_pmm_train_test(train_df, test_df, predictors, seed):
        cols = predictors + [OUTCOME]
        tr0 = train_df[cols].to_numpy(float)
        te0 = test_df[cols].to_numpy(float)
        pnum = len(predictors)
        ntr, nte = (len(train_df), len(test_df))
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
                raise RuntimeError(f'Binary predictor {predictors[t]} has missing values; logistic imputation required.')
            if len(donor) < 20:
                raise RuntimeError(f'Too few PMM donors for {predictors[t]}.')
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
            raise AssertionError('Missing predictors remain after nested MICE.')
        if not np.isfinite(tr[:, :, :pnum]).all() or not np.isfinite(te[:, :, :pnum]).all():
            raise AssertionError('Non-finite predictors after nested MICE.')
        return (tr, te)
    def fit_predict_mi_arrays(train_mi, test_mi, n_predictors):
        m = train_mi.shape[0]
        ytr = train_mi[0, :, n_predictors].astype(float)
        yte = test_mi[0, :, n_predictors].astype(float)
        ptr = np.zeros(len(ytr), float)
        pte = np.zeros(len(yte), float)
        for mm in range(m):
            Xtr = sm.add_constant(train_mi[mm, :, :n_predictors], has_constant='add')
            Xte = sm.add_constant(test_mi[mm, :, :n_predictors], has_constant='add')
            fit = sm.GLM(ytr, Xtr, family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
            if not getattr(fit, 'converged', True):
                raise RuntimeError(f'MI logistic fit {mm + 1} did not converge.')
            ptr += np.asarray(fit.predict(Xtr), float)
            pte += np.asarray(fit.predict(Xte), float)
        return (ytr, yte, ptr / m, pte / m)

    def one_bootstrap_replicate(rep, full_df, predictors, fixed_prob_col, model_name, model_seed):
        idx = deterministic_boot_indices(rep)
        train = full_df.iloc[idx].reset_index(drop=True)
        test = full_df.reset_index(drop=True)
        if train[OUTCOME].nunique() != 2:
            raise RuntimeError('Bootstrap sample contains one outcome class.')
        seed = int(np.random.SeedSequence([int(model_seed), int(rep)]).generate_state(1, dtype=np.uint64)[0])
        tr_mi, te_mi = nested_mice_pmm_train_test(train, test, predictors, seed)
        ytr, yte, ptr, pte = fit_predict_mi_arrays(tr_mi, te_mi, len(predictors))
        a = all_metrics(ytr, ptr)
        t = all_metrics(yte, pte)
        fixed_p = full_df[fixed_prob_col].to_numpy(float)
        fixed_y = full_df[OUTCOME].to_numpy(float)
        f = all_metrics(fixed_y[idx], fixed_p[idx])
        row = {'replicate': int(rep), 'model': model_name, 'bootstrap_unique_rows': int(len(np.unique(idx))), 'bootstrap_events': int(ytr.sum())}
        for k, v in a.items():
            row[f'bootstrap_apparent_{k}'] = v
        for k, v in t.items():
            row[f'test_original_{k}'] = v
        for k in ['auroc', 'brier', 'calibration_intercept', 'calibration_slope']:
            row[f'optimism_{k}'] = a[k] - t[k]
        row['optimism_log_observed_expected_ratio'] = a['log_observed_expected_ratio'] - t['log_observed_expected_ratio']
        for k, v in f.items():
            row[f'fixed_model_fit_resample_{k}'] = v
        return row
    def run_bootstrap_model(full_df, predictors, fixed_prob_col, model_name, model_seed, progress_path):
        progress_path = Path(progress_path)
        if progress_path.exists():
            out = pd.read_csv(progress_path)
            done = set(out['replicate'].astype(int))
            print(f'{model_name}: resume {len(done)}/{N_BOOTSTRAP}')
        else:
            out = pd.DataFrame()
            done = set()
        session_start = time.time()
        initial_done = len(done)
        for rep in range(1, N_BOOTSTRAP + 1):
            if rep in done:
                continue
            t0 = time.time()
            try:
                row = one_bootstrap_replicate(rep, full_df, predictors, fixed_prob_col, model_name, model_seed)
            except Exception as e:
                fail = {'model': model_name, 'replicate': rep, 'error_type': type(e).__name__, 'error': str(e), 'timestamp_utc': pd.Timestamp.utcnow().isoformat()}
                fp = runtime / f'{model_name.lower()}_failed_replicate_{rep:04d}.json'
                fp.write_text(json.dumps(fail, indent=2))
                raise RuntimeError(f'Bootstrap failure at {model_name} replicate {rep}. See {fp}') from e
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
            out = out.drop_duplicates('replicate', keep='last').sort_values('replicate')
            atomic_write_csv(out, progress_path)
            completed = len(out)
            newn = max(completed - initial_done, 1)
            avg = (time.time() - session_start) / newn
            eta_h = (N_BOOTSTRAP - completed) * avg / 3600
            print(f'{model_name}: {completed:3d}/{N_BOOTSTRAP} | rep {rep}: {(time.time() - t0) / 60:.2f} min | ETA ~{eta_h:.1f} h')
        if len(out) != N_BOOTSTRAP:
            raise AssertionError(f'{model_name}: not all 500 replicates completed.')
        if set(out['replicate'].astype(int)) != set(range(1, N_BOOTSTRAP + 1)):
            raise AssertionError(f'{model_name}: replicate IDs incomplete.')
        return out.reset_index(drop=True)
    PRIMARY_PROGRESS = runtime / 'primary_bootstrap_progress.csv'
    SECONDARY_PROGRESS = runtime / 'secondary_bootstrap_progress.csv'
    print('Progress system ready.')
    primary_boot = run_bootstrap_model(p, PRIMARY_PREDICTORS, 'primary_apparent_probability_mi_mean', 'PRIMARY', PRIMARY_BOOT_MI_SEED, PRIMARY_PROGRESS)
    print('PRIMARY 500/500 COMPLETE')
    secondary_boot = run_bootstrap_model(s, SECONDARY_PREDICTORS, 'secondary_apparent_probability_mi_mean', 'SECONDARY', SECONDARY_BOOT_MI_SEED, SECONDARY_PROGRESS)
    print('SECONDARY 500/500 COMPLETE')
    def get_apparent(model):
        r = model_fit_apparent.loc[model_fit_apparent['model'].eq(model)]
        if len(r) != 1:
            raise AssertionError(f'Model-fit apparent row not unique: {model}')
        return r.iloc[0]

    def summarize_validation(model, boot):
        app = get_apparent(model)
        original = {'auroc': float(app['apparent_auroc']), 'brier': float(app['apparent_brier']), 'calibration_intercept': float(app['calibration_intercept']), 'calibration_slope': float(app['calibration_slope']), 'observed_expected_ratio': float(app['observed_expected_ratio'])}
        rows = []
        for metric in ['auroc', 'brier', 'calibration_intercept', 'calibration_slope']:
            opt = boot[f'optimism_{metric}'].to_numpy(float)
            fixed = boot[f'fixed_model_fit_resample_{metric}'].to_numpy(float)
            mean_opt = float(opt.mean())
            lo, hi = np.percentile(fixed, [2.5, 97.5])
            rows.append({'model': model, 'metric': metric, 'apparent_estimate': original[metric], 'mean_bootstrap_optimism': mean_opt, 'optimism_corrected_estimate': original[metric] - mean_opt, 'apparent_bootstrap_ci95_low': float(lo), 'apparent_bootstrap_ci95_high': float(hi), 'optimism_adjusted_ci95_low': float(lo - mean_opt), 'optimism_adjusted_ci95_high': float(hi - mean_opt), 'bootstrap_resamples': N_BOOTSTRAP})
        app_oe = original['observed_expected_ratio']
        app_log = float(np.log(app_oe))
        opt_log = boot['optimism_log_observed_expected_ratio'].to_numpy(float)
        fixed_log = boot['fixed_model_fit_resample_log_observed_expected_ratio'].to_numpy(float)
        mean_opt_log = float(opt_log.mean())
        lo, hi = np.percentile(fixed_log, [2.5, 97.5])
        rows.append({'model': model, 'metric': 'observed_expected_ratio', 'apparent_estimate': app_oe, 'mean_bootstrap_optimism': mean_opt_log, 'optimism_corrected_estimate': float(np.exp(app_log - mean_opt_log)), 'apparent_bootstrap_ci95_low': float(np.exp(lo)), 'apparent_bootstrap_ci95_high': float(np.exp(hi)), 'optimism_adjusted_ci95_low': float(np.exp(lo - mean_opt_log)), 'optimism_adjusted_ci95_high': float(np.exp(hi - mean_opt_log)), 'bootstrap_resamples': N_BOOTSTRAP})
        return pd.DataFrame(rows)
    primary_summary = summarize_validation('PRIMARY', primary_boot)
    secondary_summary = summarize_validation('SECONDARY', secondary_boot)
    summary = pd.concat([primary_summary, secondary_summary], ignore_index=True)
    display_table(summary)
    def calibration_plot(df, prob_col, title, model_summary, png_path, csv_path):
        y = df[OUTCOME].to_numpy(float)
        prob = df[prob_col].to_numpy(float)
        smoothed = lowess(y, prob, frac=LOWESS_FRAC, it=0, return_sorted=True)
        curve = pd.DataFrame({'predicted_probability': smoothed[:, 0], 'lowess_observed_probability': smoothed[:, 1]})
        curve = curve.groupby('predicted_probability', as_index=False).mean()
        curve.to_csv(csv_path, index=False)
        ms = model_summary.set_index('metric')
        cint = ms.loc['calibration_intercept', 'optimism_corrected_estimate']
        cslope = ms.loc['calibration_slope', 'optimism_corrected_estimate']
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], '--', linewidth=1.2, label='Ideal')
        plt.plot(curve['predicted_probability'], curve['lowess_observed_probability'], linewidth=2, label=f'LOWESS apparent (frac={LOWESS_FRAC:.2f})')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Predicted in-hospital mortality probability')
        plt.ylabel('Observed in-hospital mortality')
        plt.title(title)
        plt.legend(loc='upper left')
        txt = f'N={len(df)}, events={int(y.sum())}\nOptimism-corrected intercept={cint:.3f}\nOptimism-corrected slope={cslope:.3f}\nLOWESS curve: apparent; scalar calibration: bootstrap-corrected'
        plt.text(0.04, 0.96, txt, transform=plt.gca().transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round', alpha=0.15))
        plt.tight_layout()
        plt.savefig(png_path, dpi=180, bbox_inches='tight')
        None
        plt.close()
    calibration_plot(p, 'primary_apparent_probability_mi_mean', 'PRIMARY — Transportable Core', primary_summary, runtime / 'primary_calibration_lowess.png', runtime / 'primary_calibration_lowess_data.csv')
    calibration_plot(s, 'secondary_apparent_probability_mi_mean', 'SECONDARY — Biomarker-Augmented', secondary_summary, runtime / 'secondary_calibration_lowess.png', runtime / 'secondary_calibration_lowess_data.csv')

    primary_boot.to_csv(OUT / "primary_bootstrap_replicates.csv", index=False)
    secondary_boot.to_csv(OUT / "secondary_bootstrap_replicates.csv", index=False)
    summary.to_csv(OUT / "internal_validation_summary.csv", index=False)
    for name in (
        "primary_calibration_lowess.png", "secondary_calibration_lowess.png",
        "primary_calibration_lowess_data.csv", "secondary_calibration_lowess_data.csv",
    ):
        source = runtime / name
        target = OUT / name
        target.write_bytes(source.read_bytes())

    method = {
        "method": "bootstrap optimism correction",
        "bootstrap_resamples_primary": N_BOOTSTRAP,
        "bootstrap_resamples_secondary": N_BOOTSTRAP,
        "number_of_imputations_inside_each_bootstrap": N_IMPUTATIONS,
        "mice_cycles": MICE_CYCLES,
        "pmm_donors": PMM_DONORS,
        "imputation_nested_within_bootstrap": True,
        "outcome_in_development_imputation_model": True,
        "point_estimate_correction": "apparent estimate minus mean bootstrap optimism",
        "oe_correction": "performed on log(O/E) scale",
        "lowess_frac": LOWESS_FRAC,
    }
    (OUT / "internal_validation_method.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    output_manifest = {
        "status": "PASS",
        "n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "primary_model": "Transportable Core Model",
        "secondary_model": "Biomarker-Augmented Model",
        "primary_parameters": 19,
        "secondary_parameters": 21,
        "bootstrap_resamples_primary": N_BOOTSTRAP,
        "bootstrap_resamples_secondary": N_BOOTSTRAP,
        "nested_imputations_per_bootstrap": N_IMPUTATIONS,
        "internal_validation_completed": True,
        "resume_fingerprint": resume_fingerprint,
    }
    (OUT / "internal_validation_manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print("INTERNAL VALIDATION: PASS")


if __name__ == "__main__":
    main()
