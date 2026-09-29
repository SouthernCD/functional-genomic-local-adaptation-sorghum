"""Run the real-data example from src/simulate.py on dobby's rock01.

The name differs from the earlier e1 validation run so this example starts
a fresh run. Change name or seed for another independent run.
"""

from src.simulate import SIMULATION_JOB


PROJECT = "/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum"

base_config = {
    "output_dir": f"{PROJECT}/data/results/simulate",
    "genome_fasta_file": f"{PROJECT}/data/source/Sbicolor_730_v5.0.fa.gz",
    "site_info_df_h5": f"{PROJECT}/data/results/site_contexts.genome.maf0.05.h5",
    "input_vcf_file": f"{PROJECT}/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz",
    "kinship_ibs_file": (
        f"{PROJECT}/data/source/"
        "landraces_snp_pruned.win15000.miss0.5.maf0.01.het0.01.rq0.5.ibs.matrix"
    ),
    "foreground_col": "genic_2kb",
    "ld_window_size": 15000,
    "r2_threshold": 0.5,
    "pool_num": 40,
    "causal_var": 0.3,
    "kinship_var": 0.5,
    "residual_var": 0.2,
}

# Test1
name = "five"
causal_site_num = 5
enrichment_ratio = 1.0
seed = 42

config = base_config.copy()
config.update({
    "name": name,
    "causal_site_num": causal_site_num,
    "enrichment_ratio": enrichment_ratio,
    # "pruned_site_ids_file": f"{PROJECT}/data/results/simulate/pruned_site_ids.txt.gz",
    "seed": seed,
})

result = SIMULATION_JOB(config).run()

# Test2
name = "s500"
causal_site_num = 500
causal_foreground_fraction = 0.7
seed = 42

config = base_config.copy()
config.update({
    "name": name,
    "causal_site_num": causal_site_num,
    "causal_foreground_fraction": causal_foreground_fraction,
    "enrichment_ratio": None,
    "pruned_site_ids_file": f"{PROJECT}/data/results/simulate/pruned_site_ids.txt.gz",
    "seed": seed,
})

result = SIMULATION_JOB(config).run()

# To make 70% of causal SNPs foreground instead of using an enrichment ratio:
# config["enrichment_ratio"] = None
# config["causal_foreground_fraction"] = 0.7


# Test set

causal_foreground_fractions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
seeds = range(42, 42 + 30)

set_results = []

for causal_foreground_fraction in causal_foreground_fractions:
    print(f"Running simulation for causal_foreground_fraction={causal_foreground_fraction}")
    for seed in seeds:
        seed = seed * 100
        print(f"Running simulation for seed={seed}")
        name = f"s5000f{int(causal_foreground_fraction*100)}"
        causal_site_num = 5000

        config = base_config.copy()
        config.update({
            "name": name,
            "causal_site_num": causal_site_num,
            "causal_foreground_fraction": causal_foreground_fraction,
            "enrichment_ratio": None,
            "pruned_site_ids_file": f"{PROJECT}/data/results/simulate/pruned_site_ids.txt.gz",
            "seed": seed,
        })

        result = SIMULATION_JOB(config).run(plot=False)
        pval = result['enrichment_result']['stat'][name]['pval']
        obs = result['enrichment_result']['stat'][name]['obs_value']
        set_results.append({
            "name": name,
            "causal_foreground_fraction": causal_foreground_fraction,
            "enrichment_ratio": None,
            "seed": seed,
            "pval": pval,
            "obs": obs,
        })
        
import pandas as pd
set_results_df = pd.DataFrame(set_results)
set_results_df.to_csv("simulation_results.csv", index=False)
