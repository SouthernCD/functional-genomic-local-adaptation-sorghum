#!/usr/bin/env bash
# [REMOTE COMPUTE / SCHEDULER JOB] 仅计算 CSV，不绘图；不要在 cn04 登录节点执行。
# 手动运行：bash src/calculate_multiscale_ld_decay.sh > multiscale_ld.log 2>&1
# 也可使用已有 SGE 作业资源设置提交；workers 自动使用已分配的 NSLOTS。
# 与 notebook 02 当前 multiscale cell 保持一致；参数在此集中修改。
set -euo pipefail
ld_project=/lustre/home/dobby/Work/Project/functional-genomic-local-adaptation-sorghum
ld_vcf="$ld_project/data/results/landraces_qc_miss0.5_maf0.05_het0.05.vcf.gz"
ld_stat="$ld_vcf.var_stat.h5"
ld_csv="$ld_vcf.multiscale_ld_decay.csv.gz"
ld_min_maf=0.05
ld_max_het=0.05
ld_full_distance=200000
ld_max_pairs=50000
ld_seed=42
ld_pool_num=${NSLOTS:-8}
ld_silence=false
ld_force_recompute=false
# 临时标准化数组每条染色体约 8*SNP数*样本数 bytes；留空使用系统临时目录。
ld_scratch_dir=""
source /lustre/home/dobby/Program/agent_program/activate_dobby-base.sh
cd "$ld_project"
# 防止每个 worker 再开启多个 BLAS 线程。
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -u - "$ld_stat" "$ld_vcf" "$ld_csv" "$ld_min_maf" "$ld_max_het" \
  "$ld_full_distance" "$ld_max_pairs" "$ld_seed" "$ld_pool_num" \
  "$ld_silence" "$ld_force_recompute" "$ld_scratch_dir" <<'PY'
import multiprocessing as mp
import sys
from src.ld import calculate_multiscale_ld_decay

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    (stat_file, vcf_file, csv_file, min_maf, max_het, distance, cap, seed,
     workers, silence, recompute, scratch) = sys.argv[1:]
    calculate_multiscale_ld_decay(
        stat_file, vcf_file, min_maf=float(min_maf), max_het=float(max_het),
        full_distance=int(distance), max_pairs_per_bin=int(cap), random_seed=int(seed),
        pool_num=int(workers), silence=silence.lower() == 'true',
        cache_file=csv_file, force_recompute=recompute.lower() == 'true',
        scratch_dir=scratch or None,
    )
PY
