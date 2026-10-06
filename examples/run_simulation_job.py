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
    "contrasts_settings_json_file": f"{PROJECT}/config/contrasts_settings.json",
    "ld_window_size": 15000,
    "r2_threshold": 0.5,
    "pool_num": 40,
    "causal_var": 0.3,
    "kinship_var": 0.5,
    "residual_var": 0.2,
}


# genic_2kb_vs_genome
contrast_id = "genic_2kb_vs_genome"
causal_site_nums = [50,500,5000]
causal_foreground_fractions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
seeds = range(42, 42 + 30)

# The first run generates a tagged ID list and provenance metadata. Reuse
# its returned path only after SIMULATION_JOB validates the candidate universe.
pruned_site_ids_file = None
set_results = []

for causal_site_num in causal_site_nums:
    for causal_foreground_fraction in causal_foreground_fractions:
        print(f"Running simulation for causal_foreground_fraction={causal_foreground_fraction}")
        for seed in seeds:
            seed = seed * 100
            print(f"Running simulation for seed={seed}")
            name = f"{contrast_id}.c{causal_site_num}.f{int(causal_foreground_fraction*100)}.s{seed}"

            config = base_config.copy()
            config.update({
                "name": name,
                "contrast_id": contrast_id,
                "causal_site_num": causal_site_num,
                "causal_foreground_fraction": causal_foreground_fraction,
                "enrichment_ratio": None,
                "pruned_site_ids_file": pruned_site_ids_file,
                "seed": seed,
            })

            if seed == 4200:
                result = SIMULATION_JOB(config).run(plot=True)
            else:
                result = SIMULATION_JOB(config).run(plot=False)

            if pruned_site_ids_file is None:
                pruned_site_ids_file = str(result["pruned_site_ids_file"])

            pval = result['enrichment_result']['stat'][name]['pval']
            obs = result['enrichment_result']['stat'][name]['obs_value']
            set_results.append({
                "name": name,
                "contrast_id": contrast_id,
                "causal_site_num": causal_site_num,
                "causal_foreground_fraction": causal_foreground_fraction,
                "seed": seed,
                "pval": pval,
                "obs": obs,
            })

import pandas as pd
set_results_df = pd.DataFrame(set_results)
set_results_df.to_csv(f"{PROJECT}/data/results/simulate/f{contrast_id}_simulation_results.csv", index=False)



# proximal_ACR_vs_proximal
contrast_id = "proximal_ACR_vs_proximal"
causal_site_nums = [50,500,5000]
causal_foreground_fractions = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
seeds = range(42, 42 + 30)

# The first run generates a tagged ID list and provenance metadata. Reuse
# its returned path only after SIMULATION_JOB validates the candidate universe.
pruned_site_ids_file = None
set_results = []

for causal_site_num in causal_site_nums:
    for causal_foreground_fraction in causal_foreground_fractions:
        print(f"Running simulation for causal_foreground_fraction={causal_foreground_fraction}")
        for seed in seeds:
            seed = seed * 100
            print(f"Running simulation for seed={seed}")
            name = f"{contrast_id}.c{causal_site_num}.f{int(causal_foreground_fraction*100)}.s{seed}"

            config = base_config.copy()
            config.update({
                "name": name,
                "contrast_id": contrast_id,
                "causal_site_num": causal_site_num,
                "causal_foreground_fraction": causal_foreground_fraction,
                "enrichment_ratio": None,
                "pruned_site_ids_file": pruned_site_ids_file,
                "seed": seed,
            })

            if seed == 4200:
                result = SIMULATION_JOB(config).run(plot=True)
            else:
                result = SIMULATION_JOB(config).run(plot=False)

            if pruned_site_ids_file is None:
                pruned_site_ids_file = str(result["pruned_site_ids_file"])

            pval = result['enrichment_result']['stat'][name]['pval']
            obs = result['enrichment_result']['stat'][name]['obs_value']
            set_results.append({
                "name": name,
                "contrast_id": contrast_id,
                "causal_site_num": causal_site_num,
                "causal_foreground_fraction": causal_foreground_fraction,
                "seed": seed,
                "pval": pval,
                "obs": obs,
            })

import pandas as pd
set_results_df = pd.DataFrame(set_results)
set_results_df.to_csv(f"{PROJECT}/data/results/simulate/f{contrast_id}_simulation_results.csv", index=False)
