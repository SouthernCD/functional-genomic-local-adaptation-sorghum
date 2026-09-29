"""Real-data example. Run from the repository root with python -m examples.run_feature_coverage."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_coverage import FEATURE_COVERAGE_JOB

MAF_STRATA = [
    'maf_0.05_0.1', 'maf_0.1_0.2', 'maf_0.2_0.3',
    'maf_0.3_0.4', 'maf_0.4_0.5',
]

def load_example_data(data_dir, env_id_list, maf_stratify=MAF_STRATA):
    data_dir = Path(data_dir)
    input_data_df = pd.read_hdf(
        data_dir / 'results/site_contexts.genome.maf0.05.h5', key='df')[
            ['ID', 'CHROM', 'POS', 'ACR'] + list(maf_stratify)].copy()
    input_data_df = input_data_df.rename(columns={'ACR': 'foreground'})
    if input_data_df.ID.isna().any() or input_data_df.ID.duplicated().any():
        raise ValueError('Site IDs must be present and unique')
    for env in env_id_list:
        path = (data_dir / 'results/gwas' / env / f'{env}.gemma'
                / f'{env}.gemma.gwas_df.h5')
        env_df = pd.read_hdf(path, key='df')[['ID', 'P_WALD']]
        if env_df.ID.duplicated().any():
            raise ValueError(f'Duplicate GWAS ID in {env}')
        p = pd.to_numeric(env_df.P_WALD, errors='coerce')
        score = pd.Series(-np.log10(p.clip(lower=np.finfo(float).tiny)).to_numpy(),
                          index=env_df.ID)
        input_data_df[env] = score.reindex(input_data_df.ID).to_numpy()
    return input_data_df


if __name__ == '__main__':
    data_dir = Path('/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum/data')
    env_id_list = ['bio18', 'gsl', 'al', 'striga']
    input_data_df = load_example_data(data_dir, env_id_list)
    output_dir = Path(os.environ.get(
        'FEATURE_COVERAGE_OUTPUT_DIR',
        data_dir / 'results/ld_weighted_acr_enrichment_maf_v2'))
    job = FEATURE_COVERAGE_JOB(
        input_data_df=input_data_df,
        input_vcf_file=data_dir / 'results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz',
        output_dir=output_dir,
        env_id_list=env_id_list, maf_stratify=MAF_STRATA,
        top_percent=0.05, ld_window_size=15000,
        block_r2_threshold=0.5, score_min_r2=0.25,
        permutations=1000, pool_num=20,
        permutation_pool_num=4, silence=False,
    )
    job.run()
