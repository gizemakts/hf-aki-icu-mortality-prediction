#!/usr/bin/env python3
"""03 Urine Output Alternatives.

Build landmark-safe urine-output variants.

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
    import sys, hashlib
    import pandas as pd
    import duckdb
    CONFIRMED_MIMIC_ROOT = Path('data/raw/mimic-iv-3.1')
    CONFIRMED_PROJECT_ROOT = _REPOSITORY_ROOT
    MIMIC_ROOT = Path(os.environ.get('MIMIC_ROOT', CONFIRMED_MIMIC_ROOT))
    PROJECT_ROOT = Path(os.environ.get('HF_AKI_PROJECT_ROOT', CONFIRMED_PROJECT_ROOT))
    COHORT_DIR = PROJECT_ROOT / 'outputs' / '01_cohort'
    PHENOTYPE_DIR = PROJECT_ROOT / 'outputs' / '02_phenotype'
    URINE_OUTPUT_DIR = PHENOTYPE_DIR / 'urine_output'
    PHENOTYPE_QC_DIR = PROJECT_ROOT / 'quality_control' / 'phenotype'
    for d in [COHORT_DIR, PHENOTYPE_DIR, URINE_OUTPUT_DIR, PHENOTYPE_QC_DIR]:
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

    def sha256_file(p: Path, block=1024 * 1024):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while True:
                b = f.read(block)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    print('MIMIC_ROOT :', MIMIC_ROOT)
    print('PROJECT_ROOT:', PROJECT_ROOT)
    print('URINE_OUTPUT_DIR:', URINE_OUTPUT_DIR)
    print('Python     :', sys.version.split()[0])
    print('pandas     :', pd.__version__)
    print('duckdb     :', duckdb.__version__)
    con = duckdb.connect(database=':memory:')
    LANDMARK_PATH = require_file(COHORT_DIR / 'landmark_population.parquet')
    CE_PATH = require_file(MIMIC_ROOT / 'icu' / 'chartevents.csv.gz')
    OE_PATH = require_file(MIMIC_ROOT / 'icu' / 'outputevents.csv.gz')
    PAT_PATH = require_file(MIMIC_ROOT / 'hosp' / 'patients.csv.gz')
    OMR_PATH = MIMIC_ROOT / 'hosp' / 'omr.csv.gz'
    con.execute(f"CREATE VIEW landmark_eligible AS SELECT * FROM read_parquet('{qpath(LANDMARK_PATH)}')")
    con.execute(f"CREATE VIEW chartevents AS SELECT * FROM read_csv_auto('{qpath(CE_PATH)}',header=true,sample_size=200000)")
    con.execute(f"CREATE VIEW outputevents AS SELECT * FROM read_csv_auto('{qpath(OE_PATH)}',header=true,sample_size=200000)")
    con.execute(f"CREATE VIEW patients AS SELECT * FROM read_csv_auto('{qpath(PAT_PATH)}',header=true,sample_size=200000)")
    if OMR_PATH.exists():
        con.execute(f"CREATE VIEW omr AS SELECT * FROM read_csv_auto('{qpath(OMR_PATH)}',header=true,sample_size=200000)")
        HAS_OMR = True
    else:
        HAS_OMR = False
        print('WARNING: omr.csv.gz not found; IBW will use ICU height only.')
    ABW_MIN_KG, ABW_MAX_KG = (25.0, 300.0)
    GAP_VARIANTS = [1.0, 2.0, 3.0, 4.0, 6.0, 24.0]
    FULL_EPS = 1e-06
    UO_PRIMARY = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567)
    UO_OFFICIAL = (226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558, 227488, 227489)
    COMPARTMENT_CASE = "\nCASE\n WHEN oe.itemid IN (226559,226560,226561,226563,226567) THEN 'BLADDER'\n WHEN oe.itemid=226584 THEN 'ILEAL_CONDUIT'\n WHEN oe.itemid=226564 THEN 'NEPHROSTOMY_R'\n WHEN oe.itemid=226565 THEN 'NEPHROSTOMY_L'\n ELSE NULL END\n"
    print('GAP_VARIANTS:', GAP_VARIANTS)
    con.execute(f'\nCREATE OR REPLACE TEMP TABLE selected_abw AS\nSELECT stay_id,charttime weight_time_used,storetime weight_storetime_used,valuenum admission_weight_kg\nFROM (\n SELECT l.stay_id,ce.charttime,ce.storetime,ce.valuenum,\n ROW_NUMBER() OVER(PARTITION BY l.stay_id ORDER BY ce.charttime,ce.storetime) rn\n FROM landmark_eligible l JOIN chartevents ce USING(stay_id)\n WHERE ce.itemid=226512 AND ce.valuenum BETWEEN {ABW_MIN_KG} AND {ABW_MAX_KG}\n   AND ce.charttime>=l.admittime AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time\n) x WHERE rn=1\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE icu_height AS\nSELECT stay_id,charttime height_time_used,storetime height_storetime_used,height_cm,'ICU_CHARTEVENTS' height_source\nFROM (\n SELECT l.stay_id,ce.charttime,ce.storetime,\n CASE WHEN ce.itemid=226730 THEN ce.valuenum WHEN ce.itemid=226707 THEN ce.valuenum*2.54 END height_cm,\n ROW_NUMBER() OVER(PARTITION BY l.stay_id ORDER BY ce.charttime,ce.storetime) rn\n FROM landmark_eligible l JOIN chartevents ce USING(stay_id)\n WHERE ce.itemid IN (226730,226707) AND ce.valuenum IS NOT NULL\n   AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time\n   AND ((ce.itemid=226730 AND ce.valuenum>120 AND ce.valuenum<230) OR (ce.itemid=226707 AND ce.valuenum*2.54>120 AND ce.valuenum*2.54<230))\n) x WHERE rn=1\n")
    if HAS_OMR:
        con.execute("\n    CREATE OR REPLACE TEMP TABLE prior_omr_height AS\n    SELECT stay_id,height_cm,chartdate height_chartdate,'PRIOR_OMR' height_source\n    FROM (\n      SELECT l.stay_id,o.chartdate,\n        TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 AS height_cm,\n        ROW_NUMBER() OVER(PARTITION BY l.stay_id ORDER BY o.chartdate DESC,o.seq_num DESC) rn\n      FROM landmark_eligible l JOIN omr o USING(subject_id)\n      WHERE o.result_name IN ('Height (Inches)','Height')\n        AND o.chartdate < CAST(l.admittime AS DATE)\n        AND TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 >120\n        AND TRY_CAST(TRIM(o.result_value) AS DOUBLE)*2.54 <230\n    ) x WHERE rn=1\n    ")
    else:
        con.execute('CREATE OR REPLACE TEMP TABLE prior_omr_height AS SELECT CAST(NULL AS BIGINT) stay_id,CAST(NULL AS DOUBLE) height_cm,CAST(NULL AS DATE) height_chartdate,CAST(NULL AS VARCHAR) height_source WHERE FALSE')
    con.execute("\nCREATE OR REPLACE TEMP TABLE weight_height AS\nSELECT l.stay_id,a.admission_weight_kg,a.weight_time_used,a.weight_storetime_used,\n COALESCE(i.height_cm,o.height_cm) height_cm,\n CASE WHEN i.height_cm IS NOT NULL THEN i.height_source WHEN o.height_cm IS NOT NULL THEN o.height_source ELSE NULL END height_source,\n i.height_time_used,i.height_storetime_used,o.height_chartdate,\n p.gender,\n CASE WHEN COALESCE(i.height_cm,o.height_cm) IS NULL THEN NULL\n      WHEN UPPER(p.gender)='M' THEN 50.0+2.3*((COALESCE(i.height_cm,o.height_cm)/2.54)-60.0)\n      WHEN UPPER(p.gender)='F' THEN 45.5+2.3*((COALESCE(i.height_cm,o.height_cm)/2.54)-60.0)\n      ELSE NULL END ibw_kg\nFROM landmark_eligible l LEFT JOIN selected_abw a USING(stay_id) LEFT JOIN icu_height i USING(stay_id)\nLEFT JOIN prior_omr_height o USING(stay_id) LEFT JOIN patients p USING(subject_id)\n")
    height_qc = con.execute("SELECT COUNT(*) n,SUM(admission_weight_kg IS NOT NULL) with_abw,SUM(height_cm IS NOT NULL) with_height,SUM(ibw_kg IS NOT NULL AND ibw_kg>0) with_ibw,SUM(height_source='ICU_CHARTEVENTS') height_icu,SUM(height_source='PRIOR_OMR') height_prior_omr,MIN(height_cm) min_height,MAX(height_cm) max_height,MIN(ibw_kg) min_ibw,MAX(ibw_kg) max_ibw FROM weight_height").df()
    display_table(height_qc)
    con.execute(f'\nCREATE OR REPLACE TEMP TABLE uo_events AS\nSELECT l.stay_id,{COMPARTMENT_CASE} compartment,oe.charttime,MAX(oe.storetime) storetime,SUM(oe.value) volume_ml\nFROM landmark_eligible l JOIN outputevents oe USING(stay_id)\nWHERE oe.itemid IN {UO_PRIMARY} AND oe.value IS NOT NULL AND oe.value BETWEEN 0 AND 5000\n AND oe.charttime>=l.intime AND oe.charttime<=l.landmark_time AND oe.storetime IS NOT NULL AND oe.storetime<=l.landmark_time\nGROUP BY l.stay_id,compartment,oe.charttime HAVING compartment IS NOT NULL\n')
    con.execute('\nCREATE OR REPLACE TEMP TABLE uo_interval_raw AS\nSELECT stay_id,compartment,charttime interval_end,storetime,volume_ml,\n LAG(charttime) OVER(PARTITION BY stay_id,compartment ORDER BY charttime) interval_start\nFROM uo_events\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE uo_intervals_all AS\nSELECT *,DATE_DIFF('second',interval_start,interval_end)/3600.0 duration_hours\nFROM uo_interval_raw\nWHERE interval_start IS NOT NULL AND interval_end>interval_start\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE hour_grid AS\nSELECT l.stay_id,l.intime,l.landmark_time,h.range::INTEGER hour_index,\n l.intime+h.range*INTERVAL '1 hour' hour_start,l.intime+(h.range+1)*INTERVAL '1 hour' hour_end\nFROM landmark_eligible l CROSS JOIN range(0,24) h\n")
    def build_gap_variant(gap: float):
        tag = ('g' + str(gap).replace('.', 'p')).replace('p0', '')
        con.execute(f'CREATE OR REPLACE TEMP TABLE valid_{tag} AS SELECT stay_id,interval_start,interval_end,volume_ml,duration_hours FROM uo_intervals_all WHERE duration_hours<={gap}')
        con.execute(f'\n    CREATE OR REPLACE TEMP TABLE marked_{tag} AS\n    SELECT *,MAX(interval_end) OVER(PARTITION BY stay_id ORDER BY interval_start,interval_end ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM valid_{tag}\n    ')
        con.execute(f'\n    CREATE OR REPLACE TEMP TABLE groups_{tag} AS\n    SELECT *,SUM(CASE WHEN prev_end IS NULL OR interval_start>prev_end THEN 1 ELSE 0 END) OVER(PARTITION BY stay_id ORDER BY interval_start,interval_end ROWS UNBOUNDED PRECEDING) grp FROM marked_{tag}\n    ')
        con.execute(f'CREATE OR REPLACE TEMP TABLE union_{tag} AS SELECT stay_id,grp,MIN(interval_start) interval_start,MAX(interval_end) interval_end FROM groups_{tag} GROUP BY stay_id,grp')
        con.execute(f"\n    CREATE OR REPLACE TEMP TABLE hv_{tag} AS\n    SELECT g.stay_id,g.hour_index,SUM(i.volume_ml*(DATE_DIFF('second',GREATEST(i.interval_start,g.hour_start),LEAST(i.interval_end,g.hour_end))/3600.0)/i.duration_hours) observed_volume_ml\n    FROM hour_grid g JOIN valid_{tag} i ON i.stay_id=g.stay_id AND i.interval_end>g.hour_start AND i.interval_start<g.hour_end\n    GROUP BY g.stay_id,g.hour_index\n    ")
        con.execute(f"\n    CREATE OR REPLACE TEMP TABLE hc_{tag} AS\n    SELECT g.stay_id,g.hour_index,SUM(DATE_DIFF('second',GREATEST(c.interval_start,g.hour_start),LEAST(c.interval_end,g.hour_end))/3600.0) coverage_hours\n    FROM hour_grid g JOIN union_{tag} c ON c.stay_id=g.stay_id AND c.interval_end>g.hour_start AND c.interval_start<g.hour_end GROUP BY g.stay_id,g.hour_index\n    ")
        con.execute(f'\n    CREATE OR REPLACE TEMP TABLE hourly_{tag} AS\n    SELECT g.*,COALESCE(v.observed_volume_ml,0.0) vol,LEAST(1.0,COALESCE(c.coverage_hours,0.0)) cov\n    FROM hour_grid g LEFT JOIN hv_{tag} v USING(stay_id,hour_index) LEFT JOIN hc_{tag} c USING(stay_id,hour_index)\n    ')
        con.execute(f'\n    CREATE OR REPLACE TEMP TABLE rolling_{tag} AS\n    SELECT *,\n      SUM(vol) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) vol6,\n      SUM(cov) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) cov6,\n      SUM(vol) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) vol12,\n      SUM(cov) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) cov12,\n      SUM(vol) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) vol24,\n      SUM(cov) OVER(PARTITION BY stay_id ORDER BY hour_index ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) cov24\n    FROM hourly_{tag}\n    ')
        con.execute(f'\n    CREATE OR REPLACE TEMP TABLE stage_{tag} AS\n    SELECT r.*,wh.admission_weight_kg,wh.ibw_kg,\n      CASE WHEN wh.admission_weight_kg IS NULL THEN NULL\n        WHEN hour_index>=23 AND cov24>=24-{FULL_EPS} AND vol24/(wh.admission_weight_kg*24)<0.3 THEN 3\n        WHEN hour_index>=11 AND cov12>=12-{FULL_EPS} AND vol12=0 THEN 3\n        WHEN hour_index>=11 AND cov12>=12-{FULL_EPS} AND vol12/(wh.admission_weight_kg*12)<0.5 THEN 2\n        WHEN hour_index>=5 AND cov6>=6-{FULL_EPS} AND vol6/(wh.admission_weight_kg*6)<0.5 THEN 1 ELSE 0 END stage_abw_hour,\n      CASE WHEN wh.ibw_kg IS NULL OR wh.ibw_kg<=0 THEN NULL\n        WHEN hour_index>=23 AND cov24>=24-{FULL_EPS} AND vol24/(wh.ibw_kg*24)<0.3 THEN 3\n        WHEN hour_index>=11 AND cov12>=12-{FULL_EPS} AND vol12=0 THEN 3\n        WHEN hour_index>=11 AND cov12>=12-{FULL_EPS} AND vol12/(wh.ibw_kg*12)<0.5 THEN 2\n        WHEN hour_index>=5 AND cov6>=6-{FULL_EPS} AND vol6/(wh.ibw_kg*6)<0.5 THEN 1 ELSE 0 END stage_ibw_hour\n    FROM rolling_{tag} r LEFT JOIN weight_height wh USING(stay_id)\n    ')
        return con.execute(f"\n    SELECT l.stay_id,'gap{gap:g}_full' variant_id,{gap}::DOUBLE max_gap_hours,\n      MAX(CASE WHEN s.stage_abw_hour>=1 THEN s.stage_abw_hour ELSE NULL END) positive_stage_abw,\n      MAX(CASE WHEN s.stage_ibw_hour>=1 THEN s.stage_ibw_hour ELSE NULL END) positive_stage_ibw,\n      MAX(CASE WHEN s.hour_index=23 THEN s.cov24 ELSE NULL END) coverage24_hours,\n      MIN(CASE WHEN s.hour_index>=5 THEN s.cov6 ELSE NULL END) min_mature_cov6,\n      MIN(CASE WHEN s.hour_index>=11 THEN s.cov12 ELSE NULL END) min_mature_cov12,\n      CASE WHEN wh.admission_weight_kg IS NOT NULL AND MAX(CASE WHEN s.hour_index=23 THEN s.cov24 ELSE NULL END)>=24-{FULL_EPS}\n        AND MIN(CASE WHEN s.hour_index>=5 THEN s.cov6 ELSE NULL END)>=6-{FULL_EPS}\n        AND MIN(CASE WHEN s.hour_index>=11 THEN s.cov12 ELSE NULL END)>=12-{FULL_EPS} THEN 1 ELSE 0 END absence_evaluable_abw,\n      CASE WHEN wh.ibw_kg IS NOT NULL AND wh.ibw_kg>0 AND MAX(CASE WHEN s.hour_index=23 THEN s.cov24 ELSE NULL END)>=24-{FULL_EPS}\n        AND MIN(CASE WHEN s.hour_index>=5 THEN s.cov6 ELSE NULL END)>=6-{FULL_EPS}\n        AND MIN(CASE WHEN s.hour_index>=11 THEN s.cov12 ELSE NULL END)>=12-{FULL_EPS} THEN 1 ELSE 0 END absence_evaluable_ibw\n    FROM landmark_eligible l LEFT JOIN stage_{tag} s USING(stay_id) LEFT JOIN weight_height wh USING(stay_id)\n    GROUP BY l.stay_id,wh.admission_weight_kg,wh.ibw_kg\n    ").df()
    parts = []
    for gap in GAP_VARIANTS:
        print('Building study-specific alternative gap=', gap)
        parts.append(build_gap_variant(gap))
    variants = pd.concat(parts, ignore_index=True)
    variants['uo_positive_abw'] = variants.positive_stage_abw.fillna(0).ge(1).astype('int8')
    variants['uo_positive_ibw'] = variants.positive_stage_ibw.fillna(0).ge(1).astype('int8')
    print('alternative rows:', len(variants))
    con.execute("\nCREATE OR REPLACE TEMP TABLE dyn_w0 AS\nSELECT l.stay_id,l.intime,l.landmark_time,ce.charttime,ce.storetime,ce.valuenum weight,\n CASE WHEN ce.itemid=226512 THEN 'admit' ELSE 'daily' END weight_type,\n ROW_NUMBER() OVER(PARTITION BY l.stay_id,CASE WHEN ce.itemid=226512 THEN 'admit' ELSE 'daily' END ORDER BY ce.charttime) rn\nFROM landmark_eligible l JOIN chartevents ce USING(stay_id)\nWHERE ce.itemid IN (226512,224639) AND ce.valuenum>0 AND ce.valuenum<1500\n AND ce.charttime<=l.landmark_time AND ce.storetime IS NOT NULL AND ce.storetime<=l.landmark_time\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE dyn_w1 AS\nSELECT stay_id,intime,landmark_time,weight,weight_type,\n CASE WHEN weight_type='admit' AND rn=1 THEN intime-INTERVAL '2 hours' ELSE charttime END starttime\nFROM dyn_w0\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE dyn_w2 AS\nSELECT *,COALESCE(LEAD(starttime) OVER(PARTITION BY stay_id ORDER BY starttime),landmark_time+INTERVAL '2 hours') endtime\nFROM dyn_w1\n")
    con.execute('\nCREATE OR REPLACE TEMP TABLE first_w AS SELECT * FROM dyn_w2 QUALIFY ROW_NUMBER() OVER(PARTITION BY stay_id ORDER BY starttime)=1\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE weight_durations_like AS\nSELECT stay_id,starttime,endtime,weight,weight_type FROM dyn_w2\nUNION ALL\nSELECT stay_id,intime-INTERVAL '2 hours',starttime,weight,weight_type FROM first_w WHERE intime<starttime\n")
    con.execute(f'\nCREATE OR REPLACE TEMP TABLE off_uo AS\nSELECT l.stay_id,oe.charttime,SUM(CASE WHEN oe.itemid=227488 AND oe.value>0 THEN -oe.value ELSE oe.value END) urineoutput\nFROM landmark_eligible l JOIN outputevents oe USING(stay_id)\nWHERE oe.itemid IN {UO_OFFICIAL} AND oe.value IS NOT NULL\n AND oe.charttime>=l.intime AND oe.charttime<=l.landmark_time\n AND oe.storetime IS NOT NULL AND oe.storetime<=l.landmark_time\nGROUP BY l.stay_id,oe.charttime\n')
    con.execute("\nCREATE OR REPLACE TEMP TABLE off1 AS\nSELECT u.*,l.intime,\n COALESCE(DATE_DIFF('second',LAG(u.charttime) OVER(PARTITION BY u.stay_id ORDER BY u.charttime),u.charttime)/3600.0,1.0) hours_since_previous_row\nFROM off_uo u JOIN landmark_eligible l USING(stay_id)\n")
    con.execute("\nCREATE OR REPLACE TEMP TABLE off2 AS\nSELECT *,\n SUM(urineoutput) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND CURRENT ROW) uo6,\n SUM(urineoutput) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '12 hours' PRECEDING AND CURRENT ROW) uo12,\n SUM(urineoutput) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) uo24,\n SUM(hours_since_previous_row) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND CURRENT ROW) tm6,\n SUM(hours_since_previous_row) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '12 hours' PRECEDING AND CURRENT ROW) tm12,\n SUM(hours_since_previous_row) OVER(PARTITION BY stay_id ORDER BY charttime RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW) tm24\nFROM off1\n")
    con.execute('\nCREATE OR REPLACE TEMP TABLE off_stage_events AS\nSELECT o.*,w.weight,\n CASE WHEN w.weight IS NULL THEN NULL\n      WHEN o.tm24>=24 AND o.uo24/w.weight/o.tm24<0.3 THEN 3\n      WHEN o.tm12>=12 AND o.uo12=0 THEN 3\n      WHEN o.tm12>=12 AND o.uo12/w.weight/o.tm12<0.5 THEN 2\n      WHEN o.tm6>=6 AND o.tm6<12 AND o.uo6/w.weight/o.tm6<0.5 THEN 1\n      WHEN o.tm6>=6 THEN 0 ELSE NULL END official_like_stage_event\nFROM off2 o LEFT JOIN weight_durations_like w ON o.stay_id=w.stay_id AND o.charttime>=w.starttime AND o.charttime<w.endtime\n')
    con.execute('\nCREATE OR REPLACE TEMP TABLE official_like_stay AS\nSELECT l.stay_id,MAX(e.official_like_stage_event) official_like_uo_stage_dynamic_abw,\n MAX(e.charttime) official_like_max_charttime,\n MAX(CASE WHEN e.official_like_stage_event IS NOT NULL THEN 1 ELSE 0 END) official_like_evaluable\nFROM landmark_eligible l LEFT JOIN off_stage_events e USING(stay_id) GROUP BY l.stay_id\n')
    base = con.execute('\nSELECT l.stay_id,wh.admission_weight_kg,wh.weight_time_used,wh.weight_storetime_used,\n wh.height_cm,wh.height_source,wh.height_time_used,wh.height_storetime_used,wh.height_chartdate,wh.gender,wh.ibw_kg,\n o.official_like_uo_stage_dynamic_abw,o.official_like_evaluable,o.official_like_max_charttime,\n t.max_uo_charttime_used,t.max_uo_storetime_used\nFROM landmark_eligible l LEFT JOIN weight_height wh USING(stay_id) LEFT JOIN official_like_stay o USING(stay_id)\nLEFT JOIN (SELECT stay_id,MAX(charttime) max_uo_charttime_used,MAX(storetime) max_uo_storetime_used FROM uo_events GROUP BY stay_id) t USING(stay_id)\n').df()
    lm = pd.read_parquet(LANDMARK_PATH)[['stay_id', 'landmark_time']]
    chk = base.merge(lm, on='stay_id', how='left', validate='1:1')
    for c in ['max_uo_charttime_used', 'max_uo_storetime_used', 'weight_time_used', 'weight_storetime_used', 'height_time_used', 'height_storetime_used', 'official_like_max_charttime']:
        if c in chk.columns:
            n = int((pd.to_datetime(chk[c], errors='coerce') > pd.to_datetime(chk.landmark_time)).fillna(False).sum())
            print(c, 'after landmark:', n)
            assert n == 0
    assert len(base) == len(lm)
    assert len(variants) == len(lm) * len(GAP_VARIANTS)
    variant_summary = variants.groupby('variant_id', as_index=False).agg(n=('stay_id', 'size'), abw_positive=('uo_positive_abw', 'sum'), ibw_positive=('uo_positive_ibw', 'sum'), abw_absence_evaluable=('absence_evaluable_abw', 'sum'), ibw_absence_evaluable=('absence_evaluable_ibw', 'sum'), abw_kdigo_stage_3_n=('positive_stage_abw', lambda x: int((x == 3).sum())), ibw_kdigo_stage_3_n=('positive_stage_ibw', lambda x: int((x == 3).sum())))
    display_table(variant_summary)
    display_table(base[['height_source', 'stay_id']].groupby('height_source', dropna=False).count().rename(columns={'stay_id': 'n'}).reset_index())
    off_qc = pd.DataFrame([{'n_landmark': len(base), 'official_like_evaluable': int(base.official_like_evaluable.fillna(0).sum()), 'official_like_positive': int(base.official_like_uo_stage_dynamic_abw.fillna(0).ge(1).sum()), 'official_like_stage3': int(base.official_like_uo_stage_dynamic_abw.eq(3).sum())}])
    display_table(off_qc)
    PBASE = URINE_OUTPUT_DIR / 'urine_output_base.parquet'
    PVAR = URINE_OUTPUT_DIR / 'urine_output_variants.parquet'
    base.to_parquet(PBASE, index=False)
    variants.to_parquet(PVAR, index=False)
    variant_summary.to_csv(PHENOTYPE_QC_DIR / 'urine_output_variant_summary.csv', index=False)
    height_qc.to_csv(PHENOTYPE_QC_DIR / 'height_weight_summary.csv', index=False)
    off_qc.to_csv(PHENOTYPE_QC_DIR / 'urine_output_reference_summary.csv', index=False)
    print('Saved:', PBASE)
    print('Saved:', PVAR)
    print('URINE OUTPUT ALTERNATIVES: PASS')


if __name__ == "__main__":
    main()
