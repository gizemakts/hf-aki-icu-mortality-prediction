# Mortality Prediction in Critically Ill Patients With Heart Failure and Acute Kidney Injury

Research code for the manuscript *Mortality Prediction in Critically Ill Patients
With Heart Failure and Acute Kidney Injury: Model Development, Internal
Validation, and Multicenter External Evaluation*.

## Study overview

This retrospective study estimated subsequent in-hospital mortality at a
24-hour ICU landmark. The canonical full-KDIGO models use creatinine, urine
output, and renal replacement therapy (RRT) to define AKI. They were developed
and internally validated in MIMIC-IV. A separate model developed with a
creatinine/RRT definition was evaluated in eICU-CRD using the fitted MIMIC-IV
coefficients and development-derived imputation procedure.

| Model and evaluation | Patients | Deaths | AUROC |
| --- | ---: | ---: | ---: |
| Canonical primary, MIMIC-IV internal validation | 3,457 | 707 | 0.712 |
| Canonical secondary, MIMIC-IV internal validation | 3,457 | 707 | 0.718 |
| Harmonized Cr/RRT, MIMIC-IV internal validation | 2,098 | 537 | 0.705 |
| Harmonized Cr/RRT, eICU-CRD external evaluation | 1,897 | 433 | 0.752 |

Internal AUROCs are optimism-corrected estimates. The eICU-CRD evaluation
spanned 123 hospitals and yielded a Brier score of 0.148, a calibration slope
of 1.021, and an observed-to-expected mortality ratio of 0.891. See the
[analysis guide](docs/ANALYSIS_GUIDE.md#canonical-auroc-intervals) for the
derivation and interpretation of the approximate Table 2 intervals.

## Repository layout

| Path | Content |
| --- | --- |
| [`scripts/cohort/`](scripts/cohort/) | Eligibility, landmark construction, and outcome ascertainment |
| [`scripts/preprocessing/`](scripts/preprocessing/) | AKI definitions and predictor extraction |
| [`scripts/modeling/`](scripts/modeling/) | Canonical and harmonized model development |
| [`scripts/evaluation/`](scripts/evaluation/) | Internal validation, external evaluation, and model comparisons |
| [`scripts/sensitivity/`](scripts/sensitivity/) | Sensitivity and post hoc analyses |
| [`tools/`](tools/) | Calculation of approximate canonical AUROC intervals |
| [`tests/`](tests/) | Source and calculation checks with synthetic inputs |

## Installation

Use Python 3.11–3.13 and install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

## Data access and reproducibility

Obtain individual access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
and [eICU-CRD v2.0](https://physionet.org/content/eicu-crd/2.0/) through PhysioNet.
Patient-level data, fitted model parameters, predictions, and study-specific
records are maintained separately under the applicable access restrictions.
Reproducing the reported analyses requires the authorized study inputs listed in
the [analysis guide](docs/ANALYSIS_GUIDE.md), including the reference records
authenticated by SHA-256. The guide distinguishes the tests shipped with the repository from study-specific
verification that requires additional records.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) to cite the software. Cite the accompanying
manuscript when available and acknowledge the source databases in derived work.

The research code is released under the [MIT License](LICENSE). The
[third-party notices](THIRD_PARTY_NOTICES.md) give attribution for the
MIT-LCP Glasgow Coma Scale implementation. The databases are governed by
their respective PhysioNet data-use agreements.
