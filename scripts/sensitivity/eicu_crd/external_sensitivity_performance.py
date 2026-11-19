#!/usr/bin/env python3
"""Prespecified sensitivity analyses for the eICU-CRD external evaluation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from helpers._study_records import verify_study_file

EXPECTED = {
 'contract': '291d6af54f3c6f1cb7c4160f2c6bdfc05f3421a844e0ff9fa17cf4f1504542e1',
 'primary_zip': '854ff683c0b02746c307b76c2fa4c84a81746b072619b63e8393a4b4a0f50b07',
 'primary_lock': '26cecac9f74b0225536fb38186cdbfbce5985c174c33ecd1bc510a521d58c2a6',
 'methods': '99623ea444964a4de25cc5c3fa7e5045efe5cac10ae355af40411c6c27c61ff5',
 'readiness_zip': '63e38ea6aec79cff86f0bd333dffe7a024aca63263a544f50374379634f3a6b0',
 'readiness_script': 'b0f527da822ffc2425b3f7373befc8bf3e504f17fcf4c9133cf96875ca002de9',
 'primary_script': '4ecb75ab2ce6e4dd88cba761451f627e1fb45216468fad4ae3313cb7cc3f7307',
}
ROOT_SUB = 'outputs/08_external_transport/'
PRE = ROOT_SUB + 'eicu_final_preoutcome/'
ATT = ROOT_SUB + 'eicu_outcome_attachment/'
COEF = ROOT_SUB + 'mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv'
EVALUABLE = ATT + 'eicu_external_validation_evaluable.parquet'
FLAGS = PRE + 'eicu_prespecified_sensitivity_flags_preoutcome.parquet'
PRIMARY_TOOLKIT = 'analyses/external_performance'
READINESS_TOOLKIT = 'restricted/analysis_inputs/eicu_sensitivity_readiness_toolkit'
PRIMARY_REVIEW = ROOT_SUB+'eicu_locked_external_validation/EICU_LOCKED_PRIMARY_REVIEW.zip'
PRIMARY_LOCK = 'restricted/analysis_inputs/EICU_PRIMARY_PERFORMANCE_FINAL_LOCK.zip'
READY_EVIDENCE = 'restricted/analysis_inputs/EICU_SENSITIVITY_READINESS_EVIDENCE.zip'
METHODS = 'quality_control/eicu_external_validation/eicu_locked_performance/EICU_LOCKED_METHODS_LOCK_RECEIPT.json'
NEW_METHOD_DIR = 'quality_control/eicu_external_validation/eicu_locked_performance/sensitivity_methods'
NEW_RECEIPT = 'EICU_SENSITIVITY_PERFORMANCE_METHODS_RECEIPT.json'
DEST = ROOT_SUB+'eicu_locked_external_validation/prespecified_sensitivities'
REVIEW_ZIP = 'restricted/analysis_inputs/EICU_SENSITIVITY_PERFORMANCE_REVIEW.zip'
SENSITIVITY_INPUTS = {
    "readiness_script": Path(READINESS_TOOLKIT) / "eicu_sensitivity_readiness.py",
    "primary_lock": Path(PRIMARY_LOCK),
    "readiness_zip": Path(READY_EVIDENCE),
    "methods": Path(METHODS),
}
B = 2000
SEED = 20261009
MIN_VALID = 1980
TARGET_N = 1918
EVALUABLE_N = 1897
METRICS = ('auroc','brier','calibration_intercept','calibration_slope','observed_over_expected')


def require(condition, message):
    if not condition: raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    path=Path(path)
    require(path.is_file(), 'Missing required artifact: '+str(path))
    digest=sha(path)
    require(digest==expected, 'SHA mismatch: '+str(path))
    return digest


def read_json_record(path, matches):
    """Select the unique JSON document with the expected study fields."""
    records = []
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, 'ZIP integrity failure')
        for member in archive.infolist():
            if not member.is_dir() and Path(member.filename).suffix == ".json":
                record = json.loads(archive.read(member))
                if isinstance(record, dict) and matches(record):
                    records.append(record)
    require(len(records) == 1, f"Expected one matching study record in {path.name}")
    return records[0]


def loadmodule(path, key):
    spec=importlib.util.spec_from_file_location(key,str(path))
    require(spec and spec.loader,'No module loader for '+str(path))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight(root, package, inputs=None):
    # Read-only: no score/prediction/performance computation.
    own_script = Path(__file__).resolve()
    note=package/'EICU_SENSITIVITY_PERFORMANCE_METHODS.md'
    require(note.is_file(),'Missing methods note')
    contract=root/PRIMARY_TOOLKIT/'EICU_LOCKED_PERFORMANCE_INPUT_CONTRACT.json'
    primary_code=root/PRIMARY_TOOLKIT/'external_performance.py'
    primary_zip = root / PRIMARY_REVIEW
    selected = SENSITIVITY_INPUTS if inputs is None else {**SENSITIVITY_INPUTS, **inputs}
    verified = {
        key: verify_study_file(root / path, EXPECTED[key])
        for key, path in selected.items()
    }
    ready_code = verified["readiness_script"]
    fixed_zip = verified["primary_lock"]
    evidence_zip = verified["readiness_zip"]
    for name, file in [('contract',contract),('primary_script',primary_code),('primary_zip',primary_zip)]:
        verify(file,EXPECTED[name])
    c=json.loads(contract.read_text(encoding='utf-8-sig'))
    require(len(c['validated_files'])==31,'Fixed contract must have 31 inputs')
    for item in c['validated_files']:
        rel=Path(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in item['path'],'unsafe contract path')
        p=root/rel
        verify(p,item['expected_sha256'])
        require(p.stat().st_size==int(item['size_bytes']),'fixed input size mismatch')
    lock = read_json_record(
        fixed_zip,
        lambda record: isinstance(record.get("anchor_sha256"), dict)
        and {"input_contract", "methods_receipt", "primary_zip"} <= record["anchor_sha256"].keys(),
    )
    require(lock['anchor_sha256']['input_contract']==EXPECTED['contract'],'primary contract mismatch')
    require(lock['anchor_sha256']['methods_receipt']==EXPECTED['methods'],'primary methods mismatch')
    require(lock['anchor_sha256']['primary_zip']==EXPECTED['primary_zip'],'primary ZIP mismatch')
    ready = read_json_record(
        evidence_zip,
        lambda record: {"input_sha_checked", "anchors", "complete_case",
                        "exclude_high_missing_burden", "uf_fixed_membership"} <= record.keys(),
    )
    require(ready['status']=='SENSITIVITY_MEMBERSHIP_READINESS_PASS_NO_PERFORMANCE','readiness not passed')
    require(ready['input_sha_checked']==31,'readiness contract drift')
    require(ready['anchors']['methods_receipt_sha256']==EXPECTED['methods'],'readiness methods mismatch')
    require(ready['complete_case']['evaluable']['n']==1057,'complete-case count drift')
    require(ready['uf_fixed_membership']['evaluable_n']==EVALUABLE_N,'UF fixed denominator drift')
    require(ready['uf_fixed_membership']['model_risk_rescoring_authorized'] is False,'UF rescore prohibition drift')
    # Recompute the prespecified subgroup records from the verified inputs.
    ready_mod=loadmodule(ready_code,'governed_eicu_readiness')
    data=ready_mod.build(root,contract,primary_zip,fixed_zip)
    require(data['status']==ready['status'],'source readiness status drift')
    for field in ('complete_case', 'exclude_high_missing_burden',
                  'high_burden_group_descriptive_only', 'uf_fixed_membership'):
        require(data[field] == ready[field],
                f'Membership differs from authenticated readiness evidence: {field}')
    return {'ready':data,'note_sha':sha(note),'script_sha':sha(own_script),'verified_input_count':31,
            'primary_code':primary_code,'contract_sha':EXPECTED['contract'], 'readiness_zip_sha':EXPECTED['readiness_zip']}


def record_methods(root,package,pre):
    target=root/NEW_METHOD_DIR
    require(not target.exists(),'Methods ratification path already exists; refusing overwrite')
    require(not (root/DEST).exists(),'Sensitivity output already exists; refusing methods ratification')
    receipt={'status':'RATIFIED_POST_PRIMARY_PRE_SENSITIVITY_PERFORMANCE',
             'created_utc':datetime.now(timezone.utc).isoformat(),
             'note_sha256':pre['note_sha'],'script_sha256':pre['script_sha'],
             'contract_sha256':EXPECTED['contract'],'readiness_zip_sha256':EXPECTED['readiness_zip'],
             'primary_lock_zip_sha256':EXPECTED['primary_lock'],
             'primary_methods_sha256':EXPECTED['methods'],
             'sensitivity_definitions':'fixed CC, high-burden exclusion and UF attribution only',
             'calculation_methods':'five primary scalar methods; 2000 hospital cluster bootstrap each subset; 95% percentile CI; seed 20261009',
             'ratified_before_sensitivity_performance':True,
             'post_primary_result_access_acknowledged':True,
             'UF_rescoring_authorized':False,'recalibration_authorized':False}
    # Append only into a new directory; source note is copied without rewriting.
    target.mkdir(parents=True, exist_ok=False)
    (target/'EICU_SENSITIVITY_PERFORMANCE_METHODS.md').write_bytes((package/'EICU_SENSITIVITY_PERFORMANCE_METHODS.md').read_bytes())
    p=target/NEW_RECEIPT
    p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f"Methods record written to {p}")


def check_ratification(root,pre):
    directory=root/NEW_METHOD_DIR
    note=directory/'EICU_SENSITIVITY_PERFORMANCE_METHODS.md'
    verify(note,pre['note_sha'])
    receipt_path=directory/NEW_RECEIPT
    require(receipt_path.is_file(),'Methods receipt missing; run --ratify before --run')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    require(receipt['status']=='RATIFIED_POST_PRIMARY_PRE_SENSITIVITY_PERFORMANCE','invalid ratification status')
    bindings = {
        'note_sha256': pre['note_sha'],
        'script_sha256': pre['script_sha'],
        'contract_sha256': EXPECTED['contract'],
        'readiness_zip_sha256': EXPECTED['readiness_zip'],
        'primary_lock_zip_sha256': EXPECTED['primary_lock'],
        'primary_methods_sha256': EXPECTED['methods'],
    }
    for field, expected in bindings.items():
        require(receipt.get(field) == expected, f'Ratification binding mismatch for {field}')
    require(receipt['UF_rescoring_authorized'] is False and receipt['recalibration_authorized'] is False,
        'unauthorized model changes in ratification')
    return sha(receipt_path)


def evaluate_subset(y,eta,p,sites,metric_module):
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

    y=np.asarray(y,dtype=np.int8);eta=np.asarray(eta,dtype=float);p=np.asarray(p,dtype=float);sites=np.asarray(sites)
    require(np.isin(y,[0,1]).all() and np.unique(y).size==2,'subset outcomes insufficient')
    require(np.isfinite(eta).all() and np.isfinite(p).all() and ((p>0)&(p<1)).all(),'nonfinite/extreme locked probability')
    pts=metric_module.scalar_metrics(y,eta,p)
    hospital_ids=np.unique(sites)
    indices=[np.flatnonzero(sites==hospital) for hospital in hospital_ids]
    nsite=len(hospital_ids)
    rng=np.random.Generator(np.random.PCG64(SEED))
    reps={metric:np.full(B,np.nan,dtype=float) for metric in METRICS}
    errors=Counter()
    for i in range(B):
        take=rng.integers(0,nsite,size=nsite)
        rows=np.concatenate([indices[k] for k in take])
        yy=y[rows];ee=eta[rows];pp=p[rows]
        if np.unique(yy).size<2:
            errors['single_outcome_class']+=1
            continue
        tasks={
            'auroc':lambda:metric_module.auc_ranks(yy,pp),
            'brier':lambda:float(np.mean((yy-pp)**2)),
            'calibration_intercept':lambda:metric_module.offset_intercept(yy,ee),
            'calibration_slope':lambda:metric_module.calib_slope(yy,ee),
            'observed_over_expected':lambda:float(np.sum(yy)/np.sum(pp)),
        }
        for metric,task in tasks.items():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error',category=RuntimeWarning)
                    warnings.filterwarnings('error',category=PerfectSeparationWarning)
                    v=float(task())
                if not np.isfinite(v): raise ValueError('not finite')
                reps[metric][i]=v
            except Exception as e:
                errors[f'{metric}_{type(e).__name__}']+=1
    valid={name:int(np.isfinite(v).sum()) for name,v in reps.items()}
    require(min(valid.values())>=MIN_VALID, f'bootstrap release fail closed: valid={valid} errors={dict(errors)}')
    ci={name:[float(x) for x in np.nanpercentile(v,[2.5,97.5])] for name,v in reps.items()}
    return {'metrics':{name:{'point':float(pts[name]),'ci95':ci[name],'valid_bootstrap_replicates':valid[name]} for name in METRICS},
            'bootstrap_failure_counts':dict(errors),'hospitals':nsite}


def run_sensitivities(root,package,pre):
    import statsmodels

    method_sha=check_ratification(root,pre)
    output_dir=root/DEST
    review_zip=root/REVIEW_ZIP
    require(not output_dir.exists() and not review_zip.exists(),'Sensitivity result already exists; refusing overwrite')
    model=loadmodule(pre['primary_code'],'locked_eicu_primary_metrics')
    eva=pd.read_parquet(root/EVALUABLE)
    flags=pd.read_parquet(root/FLAGS)
    require(len(eva) == EVALUABLE_N and len(flags) == TARGET_N, 'Sensitivity cohort size mismatch')
    # Recover fixed membership by stay IDs only; no inferred extra exclusions.
    require(eva['stay_id'].is_unique and flags['stay_id'].is_unique,'duplicate fixed stays')
    flag_e=flags.set_index('stay_id').loc[eva['stay_id'].tolist()].reset_index()
    model.assert_aligned_identity(flag_e,eva,'sensitivity membership identity')
    cc=(flag_e['complete_case_17_nonrenal'].astype(int).to_numpy()==1)
    hb=(flag_e['high_imputation_burden_ge7of14'].astype(int).to_numpy()==1)
    ready=pre['ready']
    retained=ready['exclude_high_missing_burden']['evaluable_retained']
    require(int(cc.sum())==1057 and int((~hb).sum())==retained['n'],'subset denominator drift')
    require(int(eva.loc[cc,'hospital_id'].nunique())==92,'complete-case sites drift')
    # Test known fixed event totals internally; do not export complementary tiny subgroup event totals.
    y=eva['hospital_expire_flag'].to_numpy(dtype=np.int8)
    require(int(np.sum(y))==433 and int(y[cc].sum())==242,'observed outcome total drift')
    features,intercept,beta=model.load_coeff(root/COEF)
    require(len(features)==19,'not 19 original predictors')
    X=eva[features].to_numpy(dtype=np.float64)
    require(np.isfinite(X).all(),'nonfinite fixed predictor matrix')
    eta=intercept+X@beta
    pred=expit(eta)
    require(np.isfinite(eta).all() and (pred>0).all() and (pred<1).all(),'invalid locked probabilities')
    sites=eva['hospital_id'].to_numpy()
    groups={'complete_case':cc,'high_burden_excluded':~hb}
    out={}
    for name,m in groups.items():
        metrics=evaluate_subset(y[m],eta[m],pred[m],sites[m],model)
        out[name]={'n':int(m.sum()),'hospitals':metrics.pop('hospitals'),**metrics}
        if name=='complete_case':
            out[name]['events']=int(y[m].sum());out[name]['non_events']=int(np.sum(m)-y[m].sum())
        else:
            out[name]['events']='WITHHELD_COMPLEMENTARY_SUPPRESSION'
            out[name]['non_events']='WITHHELD_COMPLEMENTARY_SUPPRESSION'
    report={'status':'COMPUTED','verification_scope':'calculation_only',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'methods_receipt_sha256':method_sha,
            'methods_note_sha256':pre['note_sha'],
            'analysis_script_sha256':pre['script_sha'],
            'input_contract_sha256':EXPECTED['contract'],
            'primary_lock_sha256':EXPECTED['primary_lock'],
            'readiness_evidence_sha256':EXPECTED['readiness_zip'],
            'model_coefficient_sha256':'3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513',
            'bootstrap':{'repeats':B,'seed':SEED,'generator':'NumPy Generator(PCG64)',
                         'unit':'hospital per sensitivity subset','CI':'95% percentile', 'min_valid_per_metric':MIN_VALID},
            'sensitivity_metrics':out,
            'isolated_uf':{'evaluability_population_n':EVALUABLE_N,'target_n':TARGET_N,
                           'isolated_uf_evaluable':ready['uf_fixed_membership']['isolated_uf_evaluable'],
                           'stage_states_suppressed':ready['uf_fixed_membership']['states_among_isolated_uf'],
                           'interpretation':'attribution/stage-state only; no model risk rescoring',
                           'new_risks_computed':False},
            'data_disclosure':'controlled aggregate review, not public release; complementary small-cell counts suppressed for HBX',
            'no_predictive_model_refit':True,'no_recalibration':True,'no_imputer_refit':True,
            'primary_results_unchanged':True,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                        'scipy':scipy.__version__,'statsmodels':statsmodels.__version__}}
    # Stage all content in memory, never release partial artifacts if metrics fail.
    body=json.dumps(report,indent=2,sort_keys=True,allow_nan=False).encode('utf-8')+b'\n'
    receipt={'status':'CREATED','verification_scope':'calculation_only',
             'methods_receipt_sha256':method_sha,'script_sha256':pre['script_sha'],
             'source_input_contract_sha256':EXPECTED['contract'],'primary_lock_sha256':EXPECTED['primary_lock'],
             'readiness_zip_sha256':EXPECTED['readiness_zip'],
             'file_sha256':{'EICU_SENSITIVITY_PERFORMANCE_RESULT.json':hashlib.sha256(body).hexdigest()},
             'no_recalibration':True,'no_uf_risk_rescore':True,'no_model_refit':True}
    receipt_bytes=json.dumps(receipt,indent=2,sort_keys=True).encode('utf-8')+b'\n'
    review_zip.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    files={'EICU_SENSITIVITY_PERFORMANCE_RESULT.json':body,
           'EICU_SENSITIVITY_PERFORMANCE_EXECUTION_RECEIPT.json':receipt_bytes}
    for fname,b in files.items():
        with (output_dir/fname).open('xb') as f:f.write(b)
    with zipfile.ZipFile(review_zip,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for fname,b in files.items():z.writestr(fname,b)
    print(f"Restricted sensitivity results written to {review_zip}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--package-root',type=Path,required=True)
    ap.add_argument("--readiness-source", type=Path, default=SENSITIVITY_INPUTS["readiness_script"])
    ap.add_argument("--primary-evaluation-zip", type=Path, default=SENSITIVITY_INPUTS["primary_lock"])
    ap.add_argument("--readiness-evidence-zip", type=Path, default=SENSITIVITY_INPUTS["readiness_zip"])
    ap.add_argument("--primary-methods-receipt", type=Path, default=SENSITIVITY_INPUTS["methods"])
    modes=ap.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight',action='store_true')
    modes.add_argument('--ratify',action='store_true')
    modes.add_argument('--run',action='store_true')
    args=ap.parse_args()
    root=args.project_root.resolve();package=args.package_root.resolve()
    pre = preflight(root, package, {
        "readiness_script": args.readiness_source,
        "primary_lock": args.primary_evaluation_zip,
        "readiness_zip": args.readiness_evidence_zip,
        "methods": args.primary_methods_receipt,
    })
    print(f"Verified {pre['verified_input_count']} study inputs.")
    if args.preflight:
        return
    if args.ratify:
        record_methods(root,package,pre);return
    run_sensitivities(root,package,pre)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f"Sensitivity analysis error: {exc}", file=sys.stderr)
        sys.exit(1)
