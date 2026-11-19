# Analysis and reproducibility guide

Run the commands from the repository root using the
[documented Python environment](../README.md#installation). Study workflows
require credentialed data and the corresponding authorized inputs. Keep
restricted inputs and generated outputs in the designated local data,
output, and analysis directories.

## Analysis entry points

| Stage | Entry point | Inputs and outputs |
| --- | --- | --- |
| MIMIC-IV cohort and landmark | [`verify_raw_data.py`](../scripts/cohort/mimic_iv/verify_raw_data.py), [`build_landmark_population.py`](../scripts/cohort/mimic_iv/build_landmark_population.py) | MIMIC-IV v3.1 `hosp/` and `icu/` tables. Set `MIMIC_ROOT` to their parent directory. The landmark table is written under `outputs/01_cohort/`. |
| Phenotype and predictors | [HF and chronic KRT](../scripts/cohort/mimic_iv/identify_heart_failure_and_chronic_krt.py), [AKI ascertainment](../scripts/preprocessing/aki/), [primary cohort](../scripts/cohort/mimic_iv/build_primary_cohort.py), [predictor extraction](../scripts/preprocessing/predictors/predictor_inventory.py) | Source tables, landmark population, and outcome-blind AKI adjudication. Produces the outcome-linked cohort and predictor matrices under `outputs/01_cohort/` and `outputs/03_predictors/`. |
| Canonical model development | [`fit_canonical_models.py`](../scripts/modeling/mimic_iv/fit_canonical_models.py) | Cohort, predictor matrices, and `outputs/03_predictors/modeling_policy.json`. Fitted model records, including `apparent_predictions.csv` and `apparent_performance.csv`, are written under `outputs/04_modeling/`. |
| Canonical internal validation (Table 2) | [`canonical_internal_validation.py`](../scripts/evaluation/mimic_iv/canonical_internal_validation.py) | Predictor matrices, fitted canonical model records, and the reference `outputs/05_validation/internal_validation/internal_validation_summary.csv`. The post hoc analysis excludes assessment outcomes from predictor imputation. |
| Harmonized Cr/RRT model | [`fit_harmonized_model.py`](../scripts/modeling/harmonized/fit_harmonized_model.py) | MIMIC-IV cohort, phenotype and predictor records, and canonical modeling artifacts. Writes harmonized coefficients, imputation state, and `harmonized_cr_rrt_apparent_predictions.csv` under `outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/`. |
| eICU-CRD pre-outcome preparation | [`final_preoutcome_build.py`](../scripts/cohort/eicu_crd/final_preoutcome_build.py) | eICU-CRD v2.0 tables, study eligible-site list, harmonized coefficients, and development-derived imputation state. Writes the fixed pre-outcome matrices and sensitivity flags. |
| eICU-CRD outcome attachment | [`build_outcome_evaluable_attachment.py`](../scripts/cohort/eicu_crd/build_outcome_evaluable_attachment.py) | Raw eICU-CRD tables, pre-outcome matrices, and the corresponding authorized manifest and verification record. Produces the outcome-evaluable matrix under `outputs/08_external_transport/eicu_outcome_attachment/`. |
| eICU-CRD external evaluation (Table 3 and Figure 2) | [`external_performance.py`](../scripts/evaluation/eicu_crd/external_performance.py) | Outcome-evaluable matrix, fixed coefficients, pre-outcome reference records, and sensitivity flags. Figure 2 shows the calibration curve and its interval. A separate diagnostic risk histogram combines sparse adjacent bins. |
| Model comparisons and sensitivity analyses | [APACHE IV/IVa](../scripts/evaluation/apache_iv/), [decision curves](../scripts/evaluation/decision_curve/), [sensitivity analyses](../scripts/sensitivity/) | Authorized, analysis-specific inputs and methods records as described below. |

The canonical outcome-safe validation uses 500 bootstrap resamples per model and
30 imputations per resample. [`bootstrap_internal_validation.py`](../scripts/evaluation/mimic_iv/bootstrap_internal_validation.py)
implements the comparison procedure that uses the outcome during assessment
imputation. Table 2 uses `canonical_internal_validation.py`.

Set `MIMIC_ROOT` to the credentialed MIMIC-IV directory before running the
MIMIC-IV entry points. Supply the outcome attachment inputs using the
following arguments.

```bash
python scripts/cohort/eicu_crd/build_outcome_evaluable_attachment.py \
  --eicu-root /authorized/eicu-crd-2.0 \
  --preoutcome-lock-manifest /authorized/study-records/preoutcome_manifest.json \
  --preoutcome-verification /authorized/study-records/preoutcome_verification.json
```

For a descriptive cross-database predictor-missingness comparison,
[`external_missingness_audit.py`](../scripts/cohort/eicu_crd/external_missingness_audit.py)
accepts an authorized MIMIC primary-model missingness summary with
`--mimic-missingness`. This input is optional.

## Input verification

The analysis and verification scripts check the SHA-256 values of their
specified study inputs. Use `--outcome-manifest` and `--performance-manifest`
to supply the authorized eICU outcome and performance records for the APACHE
IV/IVa and decision-curve analyses. The source tests use synthetic inputs.

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/modeling/harmonized/audit_harmonized_model_lock.py
```

The cross-analysis integrity check requires the authorized scientific-invariant
contract (JSON). Supply it at `config/accepted_scientific_invariants.json` or
specify its location explicitly:

```bash
python scripts/evaluation/mimic_iv/scientific_integrity_checks.py \
  --invariants /authorized/study-records/accepted_scientific_invariants.json
```

The contract is a study-specific source of accepted values and is required
alongside the corresponding analysis outputs. The check does not infer or
replace absent scientific-invariant values.

The optional `--mode restricted` harmonized model audit accepts
`--reference-source`, `--manifest-path`, and `--mimic-root` for the
corresponding authorized study records and MIMIC-IV data directory.
Harmonized internal validation accepts `--apparent-predictions` for the
prediction table recorded in the model manifest.

### APACHE IV/IVa and decision curves

The APACHE IV/IVa benchmark requires the eICU APACHE source table, outcome-evaluable
matrix, harmonized coefficients, primary evaluation records, and the common-cohort
membership record `evaluation/apache_iv/EICU_APACHE_COMMON_COHORT_MEMBERSHIP_LOCK.json`.
The independent verifier additionally uses the benchmark protocol and restricted
aggregate results.

The decision-curve analysis requires the fixed external evaluation inputs,
the approved methods record `DCA_SOFA_FEASIBILITY_AND_ANALYSIS_PLAN.md`, and
the study-specific test source
`evaluation/decision_curve/EICU_DCA_SYNTHETIC_TESTS.py`. Supply both files from the authorized study records at the indicated paths.
The analysis authenticates their contents by SHA-256. The synthetic tests
shipped in the repository cover selected safeguards and can be run separately. The verifier
also requires the disclosure assessment and restricted aggregate results.
The independent verifiers for both analyses accept `--project-root` and
`--run-sensitive-reproduction` for authorized local execution.

### eICU-CRD sensitivity analyses

[`external_sensitivity_performance.py`](../scripts/sensitivity/eicu_crd/external_sensitivity_performance.py)
requires the primary input contract and scoring implementation, primary evaluation
ZIP, methods receipt, sensitivity readiness source and evidence ZIP, and the
authorized inputs named in the contract. Supply the methods document through
`--package-root` and the study files through the input options below.

```bash
python scripts/sensitivity/eicu_crd/external_sensitivity_performance.py \
  --project-root /authorized/study-root \
  --package-root /authorized/sensitivity-methods \
  --readiness-source /authorized/study-records/readiness.py \
  --primary-evaluation-zip /authorized/study-records/primary_evaluation.zip \
  --readiness-evidence-zip /authorized/study-records/readiness_evidence.zip \
  --primary-methods-receipt /authorized/study-records/methods_receipt.json \
  --preflight
```

The script verifies the SHA-256 values of the study files and compares the
recomputed subgroup records with the readiness evidence. The `--ratify`
option records the methods and binds them to the analysis source and study
inputs. The `--run` option verifies that record and writes restricted
aggregate sensitivity results.

### Outcome-free imputation sensitivity analysis

For the `evaluate` phase of
[`outcome_free_imputation_sensitivity.py`](../scripts/sensitivity/posthoc/outcome_free_imputation_sensitivity.py),
the approval manifest must set `preoutcome_lock_independently_approved` and
`outcome_access_approved` to `true` and provide
`preoutcome_lock_manifest_path` and `preoutcome_lock_manifest_sha256` for the
approved pre-outcome manifest.

## Canonical AUROC intervals

The approximate 95% AUROC intervals in Table 2 use the 2.5th and 97.5th
percentiles of apparent AUROC across 500 resamples per model. Both percentiles
are computed by linear interpolation and shifted by subtracting mean bootstrap
optimism. The interval treats the optimism estimate as fixed and therefore
does not fully propagate uncertainty from estimating optimism or multiple
imputation. The corrected point estimate comes from the validation summary.

| Canonical model | Optimism-corrected AUROC | Approximate 95% interval |
| --- | ---: | ---: |
| Primary | 0.712 | 0.696–0.736 |
| Secondary | 0.718 | 0.703–0.744 |

Use the [interval calculation tool](../tools/derive_canonical_auroc_intervals.py)
with the authorized aggregate summary and resample-level AUROC records.

```bash
python tools/derive_canonical_auroc_intervals.py \
  --summary /authorized/validation/internal_validation_summary.csv \
  --primary /authorized/validation/primary_bootstrap_replicates.csv \
  --secondary /authorized/validation/secondary_bootstrap_replicates.csv
```
