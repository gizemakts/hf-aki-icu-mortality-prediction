#!/usr/bin/env python3
"""External validation of the unchanged harmonized mortality model.

Evaluate discrimination, calibration, and Brier score in the outcome-evaluable
external cohort, with 2,000 hospital-cluster bootstrap resamples. Results
are aggregate-only; patient-level predictions are not exported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import rankdata

N_REPS = 2000
SEED = 20261009
CONFIDENCE = 95
VALID_MIN = 1980
GRID_N = 120
PRED_N = 19
# Exact AMBER construct differences from the original fixed Gate-1 harmonization matrix.
FIXED_AMBER_MAPPING_DEVIATIONS = {
  'age_per10y': 'eICU patient.age >89 top-coded; fixed deterministic age proxy differs from MIMIC',
  'heart_rate_max_per10': 'nurseCharting charted extrema differ from monitor-derived high-frequency extrema',
  'map_min_per10': 'charted invasive/noninvasive MAP sources and coverage vary by site',
  'resp_rate_max_per5': 'charted respiratory rate, not systematically higher monitor extrema',
  'temperature_max_abs_deviation_from_37c': 'charted temperature, not sparse periodic probe source',
  'spo2_deficit_per5': 'charted SpO2, not systematically lower periodic minima',
  'gcs_deficit_from15': 'numeric eICU GCS total lacks exact MIMIC ETT-specific reconstruction',
  'vasoactive_any_by_landmark': 'heterogeneous eICU infusionDrug site capture; governed absence-vs-zero policy',
  'kdigo_stage_2_indicator': 'harmonized Cr/RRT stage, not original urine-output-inclusive KDIGO',
  'kdigo_stage_3_indicator': 'harmonized Cr/RRT stage, not original urine-output-inclusive KDIGO',
}
ROOT_PREFIX = "outputs/08_external_transport/"
PRE = ROOT_PREFIX+"eicu_final_preoutcome/"
ATT = ROOT_PREFIX+"eicu_outcome_attachment/"
COEF = ROOT_PREFIX+"mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"
FULL_COHORT = ATT+"eicu_final_cohort_with_outcome_availability.parquet"
FULL_MATRIX = ATT+"eicu_prediction_ready_with_outcome_availability.parquet"
EVAL_MATRIX = ATT+"eicu_external_validation_evaluable.parquet"
UNAVAILABLE = ATT+"eicu_outcome_unavailable_rows.parquet"
PRE_COHORT = PRE+"eicu_final_cohort_preoutcome.parquet"
PRE_MATRIX = PRE+"eicu_prediction_ready_matrix_preoutcome.parquet"
RAW_MATRIX = PRE+"eicu_predictor_matrix_preimputation.parquet"
FLAGS = PRE+"eicu_prespecified_sensitivity_flags_preoutcome.parquet"
RESULT_DIR = "outputs/08_external_transport/eicu_locked_external_validation"
KEY = ["unique_pid", "hstay_id", "stay_id", "hospital_id"]


def fail(msg: str):
    raise RuntimeError(msg)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if not path.is_file(): fail(f'MISSING {path}')
    actual = sha_file(path)
    if actual != expected.lower(): fail(f'SHA mismatch {path}: {actual} != {expected.lower()}')


def dump_json(path: Path, obj: dict):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_set(series, integer: bool) -> str:
    vals = [int(v) if integer else str(v).strip() for v in series if pd.notna(v)]
    payload = '\n'.join(str(v) for v in sorted(set(vals))) + '\n'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def check_unique(df: pd.DataFrame, name: str):
    if df["stay_id"].isna().any() or df.duplicated("stay_id").any(): fail(f'{name}: duplicate/missing stay_id')
    if df["hstay_id"].isna().any() or df.duplicated("hstay_id").any(): fail(f'{name}: duplicate/missing hstay_id')
    if df["unique_pid"].isna().any() or df.duplicated("unique_pid").any(): fail(f'{name}: duplicate/missing unique_pid')
    if df["hospital_id"].isna().any(): fail(f'{name}: missing hospital_id')


def assert_aligned_identity(a: pd.DataFrame, b: pd.DataFrame, name: str):
    """Compare canonical identity *values*, not pandas container dtype metadata.

    Numeric identifiers must agree exactly as integers. eICU unique_pid is a
    string identity already stripped in the fixed attachment builder. Do not
    output row-level identifiers or any patient-level data on failure.
    """
    if len(a) != len(b):
        fail(f'{name}: identity row count mismatch ({len(a)} vs {len(b)})')
    for key in KEY:
        if key not in a.columns or key not in b.columns:
            fail(f'{name}: missing identity column {key}')
        if a[key].isna().any() or b[key].isna().any():
            fail(f'{name}: missing identity values in {key}')
        if key == 'unique_pid':
            # Mirrors original eICU outcome attachment builder's canonicalization.
            left = a[key].astype('string').str.strip().to_numpy(dtype=str)
            right = b[key].astype('string').str.strip().to_numpy(dtype=str)
        else:
            for frame in (a, b):
                if not pd.api.types.is_integer_dtype(frame[key]):
                    fail(f'{name}: {key} is not an integer-typed fixed ID')
            left = a[key].to_numpy(dtype=np.int64)
            right = b[key].to_numpy(dtype=np.int64)
        mismatch = int(np.count_nonzero(left != right))
        if mismatch:
            fail(f'{name}: TRUE identity mismatch in {key}, rows={mismatch}; stop for review')
    return True


def aligned(df: pd.DataFrame, reference: pd.DataFrame, name: str):
    a = df.sort_values('stay_id', kind='stable').reset_index(drop=True)
    b = reference.sort_values('stay_id', kind='stable').reset_index(drop=True)
    assert_aligned_identity(a, b, name)
    return a, b


def canonical_matrix_check(a, b, columns, label):
    a, b = aligned(a, b, label)
    for c in columns:
        if c not in a or c not in b: fail(f'{label}: missing {c}')
        if not np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True):
            fail(f'{label}: fixed predictor changed in {c}')


def load_coeff(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 20 or set(rows[0]) != {'term','locked_coefficient'}:fail('coefficient format differs')
    names=[r['term'] for r in rows]
    if names[0] != 'const' or len(set(names))!=20:fail('unexpected coefficient names')
    coefs = np.array([float(r['locked_coefficient']) for r in rows], dtype=np.float64)
    if not np.isfinite(coefs).all():fail('nonfinite coefficients')
    return names[1:], coefs[0], coefs[1:]


def preflight(root: Path) -> dict:
    """Validate local study input tables; no external decision-record dependency."""
    inputs = [COEF, FULL_COHORT, FULL_MATRIX, EVAL_MATRIX, UNAVAILABLE,
              PRE_COHORT, PRE_MATRIX, RAW_MATRIX, FLAGS]
    for rel in inputs:
        path = root / rel
        if not path.is_file():
            fail(f"Missing input: {path}")
    eva = pd.read_parquet(root / EVAL_MATRIX)
    if len(eva) != 1897 or eva['hospital_id'].nunique() != 123:
        fail('Unexpected external evaluation cohort size or hospital count')
    y = eva['hospital_expire_flag']
    if y.isna().any() or not y.isin([0, 1]).all() or int(y.sum()) != 433:
        fail('Unexpected external mortality outcome distribution')
    pred, intercept, beta = load_coeff(root / COEF)
    if len(pred) != PRED_N or not np.isfinite(beta).all() or not np.isfinite(intercept):
        fail('Invalid harmonized model coefficient specification')
    if not set(pred).issubset(eva.columns):
        fail('Model predictors missing in external cohort')
    return {'status': 'PASS', 'primary_evaluable_n': len(eva),
            'events_from_fixed_vector': int(y.sum()),
            'coefficient_sha256': sha_file(root / COEF),
            'execution_script_sha256': sha_file(Path(__file__).resolve())}


def auc_ranks(y,p):
    a=y==1
    npos=int(a.sum()); nneg=len(y)-npos
    if not npos or not nneg:fail('AUROC requires both outcome classes')
    r=rankdata(p,method='average')
    return float((np.sum(r[a])-npos*(npos+1)/2)/(npos*nneg))


def offset_intercept(y,eta):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings('error',category=PerfectSeparationWarning)
        fit=sm.GLM(y,np.ones((len(y),1)),offset=eta,family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,disp=0)
    if not fit.converged or not np.isfinite(fit.params).all():fail('offset calibration intercept GLM failed')
    return float(fit.params[0])


def calib_slope(y,eta):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings('error',category=PerfectSeparationWarning)
        fit=sm.GLM(y,np.column_stack([np.ones(len(y)),eta]),family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,disp=0)
    if not fit.converged or not np.isfinite(fit.params).all():fail('calibration slope GLM failed')
    return float(fit.params[1])


def rcs_design(x,knots):
    x=np.asarray(x,dtype=float)
    k=np.asarray(knots,dtype=float)
    if len(k)!=4 or not np.all(np.diff(k)>1e-10):fail('spline knots not distinct')
    span=k[-1]-k[0]
    bases=[np.ones_like(x),x]
    for j in range(2):
        # Natural cubic spline truncated-power basis: linear tails.
        b=(np.maximum(x-k[j],0)**3 - np.maximum(x-k[-2],0)**3*(k[-1]-k[j])/(k[-1]-k[-2])
           +np.maximum(x-k[-1],0)**3*(k[-2]-k[j])/(k[-1]-k[-2]))/(span*span)
        bases.append(b)
    return np.column_stack(bases)


def spline_curve(y,eta,knots,grid):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('error', category=PerfectSeparationWarning)
        fit=sm.GLM(y,rcs_design(eta,knots),family=sm.families.Binomial()).fit(maxiter=100,tol=1e-10,disp=0)
    if not fit.converged or not np.isfinite(fit.params).all():fail('calibration spline GLM failed')
    curve=expit(rcs_design(grid,knots) @ fit.params)
    if not np.isfinite(curve).all():fail('nonfinite spline curve')
    return curve


def scalar_metrics(y,eta,p):
    if not(np.isfinite(eta).all() and np.isfinite(p).all()):fail('nonfinite predictions')
    if not(np.all(p>0) and np.all(p<1)):fail('fixed probability rounded to 0/1')
    if not np.isin(y,[0,1]).all():fail('nonbinary outcome')
    E=float(p.sum()); O=int(y.sum())
    if E<=0:fail('expected events 0')
    return {'auroc':auc_ranks(y,p),'brier':float(np.mean((y-p)**2)),
            'calibration_intercept':offset_intercept(y,eta), 'calibration_slope':calib_slope(y,eta),
            'observed_over_expected':float(O/E)}


def primary_run(root,pre):
    import statsmodels

    dest=root/RESULT_DIR
    if dest.exists():fail('primary performance directory already exists; cannot overwrite')
    eva=pd.read_parquet(root/EVAL_MATRIX)
    raw=pd.read_parquet(root/RAW_MATRIX)
    flags=pd.read_parquet(root/FLAGS)
    pred,intercept,beta=load_coeff(root/COEF)
    # No fitting: only fixed column lookup and original coefficients.
    X=eva[pred].to_numpy(dtype=np.float64)
    eta=intercept+X@beta
    p=expit(eta)
    y=eva['hospital_expire_flag'].to_numpy(dtype=np.int8)
    if not np.isfinite(eta).all() or not np.isfinite(p).all():fail('nonfinite locked model predictions')
    if not np.all((p>0)&(p<1)):fail('overflow/extreme locked predictions: no clipping allowed')
    points=scalar_metrics(y,eta,p)
    sites=eva['hospital_id'].to_numpy()
    hospitals=np.unique(sites)
    if len(hospitals)!=123:fail('not 123 sites')
    site_indices=[np.flatnonzero(sites==h) for h in hospitals]
    knots=np.quantile(eta,[.05,.35,.65,.95])
    grid=np.linspace(float(np.quantile(eta,.01)),float(np.quantile(eta,.99)),GRID_N)
    point_curve=spline_curve(y,eta,knots,grid)
    rng=np.random.Generator(np.random.PCG64(SEED))
    boot={k:np.full(N_REPS,np.nan,dtype=float) for k in points}
    boot_curves=np.full((N_REPS,len(grid)),np.nan,dtype=float)
    errors=Counter()
    # Cluster draws are fixed once and used for ALL metrics and the calibration curve.
    for t in range(N_REPS):
        chosen=rng.integers(0,len(hospitals),size=len(hospitals))
        idx=np.concatenate([site_indices[j] for j in chosen])
        yy=y[idx]; ee=eta[idx]; pp=p[idx]
        if np.unique(yy).size<2:
            errors['single_outcome_class']+=1
            continue
        # Each statistic's numerical failure is audited independently.
        scalar_jobs={
            'auroc':lambda: auc_ranks(yy,pp),
            'brier':lambda: float(np.mean((yy-pp)**2)),
            'calibration_intercept':lambda: offset_intercept(yy,ee),
            'calibration_slope':lambda: calib_slope(yy,ee),
            'observed_over_expected':lambda: float(yy.sum()/pp.sum())}
        for name,job in scalar_jobs.items():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error',category=RuntimeWarning)
                    value=job()
                if not np.isfinite(value):fail('nonfinite bootstrap statistic')
                boot[name][t]=value
            except Exception as exc:
                errors[name+'_exception_'+type(exc).__name__]+=1
        try:boot_curves[t]=spline_curve(yy,ee,knots,grid)
        except Exception as exc:errors['spline_exception_'+type(exc).__name__]+=1
    validity={k:int(np.isfinite(v).sum()) for k,v in boot.items()}
    curve_valid=int(np.isfinite(boot_curves).all(axis=1).sum())
    if min([*validity.values(),curve_valid])<VALID_MIN:
        # Do not suppress invalid draws or silently change CI method.
        fail(f'BOOTSTRAP RELEASE STOP; scalar_valid={validity}, curve_valid={curve_valid}, failures={dict(errors)}')
    cis={k:[float(v) for v in np.nanpercentile(series,[2.5,97.5])] for k,series in boot.items()}
    low,high=np.nanpercentile(boot_curves,[2.5,97.5],axis=0)
    if not(np.isfinite(low).all() and np.isfinite(high).all()):fail('curve CI invalid')
    # Missingness stats from fixed pre-FCE matrix, restricted by fixed evaluable identity.
    raw_e=raw.set_index('stay_id').loc[eva['stay_id'].tolist()].reset_index()
    flag_e=flags.set_index('stay_id').loc[eva['stay_id'].tolist()].reset_index()
    # Preserve identical identity proof in the reporting step; do not use
    # pandas DataFrame.equals, which additionally requires matching dtypes.
    assert_aligned_identity(raw_e, eva, 'evaluable raw missingness alignment')
    assert_aligned_identity(flag_e, eva, 'evaluable sensitivity flags alignment')
    missingness={'primary_evaluable_n':len(eva),'full_target_n':1918,'predictors':{}}
    for col in pred:
        missingness['predictors'][col]={
            'evaluable_missing_n':int(raw_e[col].isna().sum()),
            'evaluable_missing_pct':float(raw_e[col].isna().mean()*100),
            'target_missing_n':int(raw[col].isna().sum()),
            'target_missing_pct':float(raw[col].isna().mean()*100)}
    missingness['fixed_sensitivity_flags_descriptive_only']={
        'complete_case_17_nonrenal_target_n':int(flags['complete_case_17_nonrenal'].sum()),
        'complete_case_17_nonrenal_evaluable_n':int(flag_e['complete_case_17_nonrenal'].sum()),
        'high_burden_ge7of14_target_n':int(flags['high_imputation_burden_ge7of14'].sum()),
        'high_burden_ge7of14_evaluable_n':int(flag_e['high_imputation_burden_ge7of14'].sum())}
    results={'status':'EXTERNAL_VALIDATION_COMPLETE',
             'created_utc':utc(),'model':'Harmonized model MIMIC Cr/RRT transport logistic, unchanged',
             'outcome':'subsequent in-hospital mortality after +24h landmark',
             'target_n':1918,'target_sites':125,'unavailable_outcome_n':21,
             'evaluable_n':int(len(y)),'evaluable_sites':int(len(hospitals)),
             'events':int(np.sum(y)),'non_events':int((y==0).sum()),
             'observed_event_rate':float(np.mean(y)),
             'expected_event_count':float(p.sum()),'mean_locked_predicted_risk':float(np.mean(p)),
             'metrics':{k:{'point':float(v),'cluster_bootstrap_ci95':cis[k],
                           'valid_bootstrap_replicates':validity[k]} for k,v in points.items()},
             'bootstrap_failures_by_reason':dict(errors),'calibration_curve_valid_bootstraps':curve_valid,
             'calibration_knots_eta':knots.tolist(),
             'calibration_eta_grid_1pct_99pct':[float(grid[0]),float(grid[-1])],
             'execution_script_sha256':pre['execution_script_sha256'],
             'coefficient_sha256':pre['coefficient_sha256'],
             'site_coverage_context':{'source_eligible_hospitals':145,'fixed_target_hospitals':125,'outcome_evaluable_hospitals':123},
             'harmonization_mapping_summary':{'GREEN':9,'AMBER':10,'RED':0,'fixed_amber_deviations':FIXED_AMBER_MAPPING_DEVIATIONS,'interpretation':'Qualitative deviations transcribed from original fixed harmonization matrix; no mappings reclassified'},
             'limitations':['21 source-outcome-unavailable target patients excluded from outcome-requiring metrics; recorded maximum absolute availability SMD 0.515 for respiratory rate',
                            'cluster CI conditions on fixed development coefficients and FCE parameters',
                            'calibration diagnostic GLMs not applied to original locked predictions'],
             'software_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,
                                   'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,
                                   'platform':platform.platform()},
             'no_model_refit':True,'no_predictive_recalibration':True,
             'prespecified_sensitivity_performance_computed':False}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True,exist_ok=False)
    dump_json(dest/'EICU_LOCKED_PRIMARY_PERFORMANCE.json',results)
    dump_json(dest/'EICU_LOCKED_PRIMARY_MISSINGNESS.json',missingness)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],linestyle='--',color='0.4',label='Perfect calibration')
    ax.plot(expit(grid),point_curve,color='navy',label='Locked-model calibration diagnostic')
    ax.fill_between(expit(grid),low,high,color='steelblue',alpha=.22,label='95% pointwise hospital-bootstrap CI')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Locked predicted mortality risk');ax.set_ylabel('Smoothed observed mortality risk')
    ax.legend(loc='best');ax.grid(alpha=.2)
    ax.set_title('eICU locked external calibration | n=1,897')
    fig.tight_layout();fig.savefig(dest/'EICU_LOCKED_PRIMARY_CALIBRATION.png',dpi=180);plt.close(fig)
    # Publication histogram disclosure rule: do not display bins with 1-10
    # records. This merges adjacent bins and does not alter any predictions,
    # metrics, or underlying restricted outputs.
    initial_edges=np.linspace(0.0,1.0,31)
    initial_counts,_=np.histogram(p,bins=initial_edges)
    safe_edges=[float(initial_edges[0])]
    accumulated_count=0
    for j,n in enumerate(initial_counts,1):
        accumulated_count+=int(n)
        if accumulated_count>=11:
            safe_edges.append(float(initial_edges[j]))
            accumulated_count=0
    if len(safe_edges)==1:
        safe_edges=[0.0,1.0]
    elif safe_edges[-1]<1.0:
        # Include the final incomplete (or empty) region in the last safe bin.
        safe_edges[-1]=1.0
    safe_counts,_=np.histogram(p,bins=safe_edges)
    if any(0<n<11 for n in safe_counts):
        raise AssertionError('Histogram includes a disclosive small cell')
    fig,ax=plt.subplots(figsize=(7,3.5))
    ax.hist(p,bins=safe_edges,edgecolor='0.2',color='slategray')
    ax.set_xlabel('Locked predicted mortality risk');ax.set_ylabel('Patients per variable-width bin')
    ax.set_title('Original locked risk distribution | n=1,897')
    fig.tight_layout();fig.savefig(dest/'EICU_LOCKED_PRIMARY_PREDICTED_RISK_DISTRIBUTION.png',dpi=180);plt.close(fig)
    output_names=['EICU_LOCKED_PRIMARY_PERFORMANCE.json','EICU_LOCKED_PRIMARY_MISSINGNESS.json',
                  'EICU_LOCKED_PRIMARY_CALIBRATION.png','EICU_LOCKED_PRIMARY_PREDICTED_RISK_DISTRIBUTION.png']
    output_hashes={n:sha_file(dest/n) for n in output_names}
    dump_json(dest/'EICU_LOCKED_PRIMARY_EXECUTION_RECEIPT.json',{
        'status':'EXTERNAL_VALIDATION_COMPLETE', 'created_utc':utc(),
        'executed_script_sha256':pre['execution_script_sha256'],
        'output_hashes':output_hashes,'data_disclosure':'aggregate-only outputs; no row-level predictions/outcomes/IDs',
        'no_recalibration':True,'no_model_refit':True})
    review_zip=dest/'EICU_LOCKED_PRIMARY_REVIEW.zip'
    with zipfile.ZipFile(review_zip,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in output_names+['EICU_LOCKED_PRIMARY_EXECUTION_RECEIPT.json']:
            z.write(dest/name,arcname=name)
    print('EXTERNAL_VALIDATION_COMPLETE')
    print('REVIEW_ZIP',review_zip)
    print('REVIEW_SHA256',sha_file(review_zip))
    print('NO RECALIBRATION OR MODEL REFIT')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--preflight', action='store_true', help='Validate inputs without scoring')
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    if not root.is_dir():
        fail(f'Missing project directory: {root}')
    pre = preflight(root)
    if args.preflight:
        print('INPUT_VALIDATION_PASS')
    else:
        primary_run(root, pre)


if __name__ == '__main__':
    main()
