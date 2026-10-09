#!/bin/zsh
# The spherical-harmonic degree-range experiment (docs/experiment-sh-degree-range.md), as it was run.
#   ./run_degree_experiment.sh stage1   ~100 km cells (regions/australia): LCS-1 and MLI 2E, degrees 16-100/133/185,
#                                       lowest degree 20 or 25, synthetic degrees 1-133
#   ./run_degree_experiment.sh stage2   ~50 km cells (regions/australia_fine): real data to degree 300
#   ./run_degree_experiment.sh stage3   ~50 km cells: the future-data test (synthetic, low noise at high degrees)
# Stages 2-3 need build_region.py regions/australia_fine/config.toml first; stage 3 needs stage 2's first build.
# Lightly: one job at a time, 3 workers, nice 15, BLAS threads 1. Logs: '###' lines per step, 'FAILED' on errors.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 N_WORKERS=3 PYTHONPATH=..:$PYTHONPATH
PY="nice -n 15 conda run --no-capture-output -n pygmt17 python"
step() { echo "### $(date) $1"; }

stage1() {
  local R=regions/australia/config.toml L=regions/australia/cache/bands
  mkdir -p $L
  band() {   # band, cases
    step "linear models $1"
    BAND=$1 CASES=$2 ${=PY} run_linear_models.py $R > $L/$1/linear_models.log 2>&1 || echo "FAILED linear models $1"
    step "bounds $1"
    BAND=$1 ROUGH=1 ${=PY} run_bounds.py $R > $L/$1/bounds.log 2>&1 || echo "FAILED bounds $1"
  }
  step "build bands to degree 133"
  RUNS=lcs1_16-133,2e_16-133,lcs1_16-100,2e_16-100,syn_16-133,syn_1-133,lcs1_20-133,lcs1_25-133,lcs1_16-133_mf7 \
    ${=PY} build_bands.py $R > $L/build_first.log 2>&1 || echo "FAILED build"
  step "check: the first build again (GP, real data)"
  BAND=lcs1_16-133_mf7 CASES=real PARTS=gp ${=PY} run_linear_models.py $R > $L/lcs1_16-133_mf7/linear_models.log 2>&1 || echo "FAILED check"
  for b in lcs1_16-133 2e_16-133 lcs1_16-100 2e_16-100 lcs1_20-133 lcs1_25-133; do band $b cov_truth,hast,hast_var,real; done
  for b in syn_16-133 syn_1-133; do band $b cov_truth,hast,hast_var; done
  step "cell columns to 185, build bands to degree 185"
  RUNS=lcs1_16-185,2e_16-185 ${=PY} build_bands.py $R > $L/build_185.log 2>&1 || echo "FAILED build 185"
  for b in lcs1_16-185 2e_16-185; do band $b cov_truth,hast,hast_var,real; done
  ${=PY} compare_bands.py $R > $L/compare.log 2>&1 || echo "FAILED compare"
}

stage2() {
  local R=regions/australia_fine/config.toml C=regions/australia_fine/cache
  local MERGE="MODES=refine TAG=_sweep LAM_RELS=1,0.333333 GAMMAS=0.0025,0.00375,0.005,0.0075,0.01 BOOT=1,0.005"
  mkdir -p $C/bands
  step "build bands"
  RUNS=lcs1_16-185,2e_16-185,2e_16-300 ${=PY} build_bands.py $R > $C/bands/build.log 2>&1 || echo "FAILED build bands"
  step "covariates"
  ${=PY} run_covariates.py $R > $C/covariates.log 2>&1 || echo "FAILED covariates"
  step "merging sweep, first build (gamma halved for ~50 km edges)"
  env ${=MERGE} ${=PY} run_merge.py $R > $C/merge_points_sweep.log 2>&1 || echo "FAILED merge"
  step "realistic synthetic, first build (the same province values as the ~100 km cells)"
  TRUTH_FROM=regions/australia/cache/realistic_synthetic.npz MERGE_MODE=refine GAMMA_REL=0.005 \
    ${=PY} run_realistic_synthetic.py $R > $C/realistic_synthetic.log 2>&1 || echo "FAILED realistic synthetic"
  # PARTS=gp: regression, classes and hypotheses are too slow on 5,358 cells
  step "linear models, first build"
  PARTS=gp MERGE_MODE=refine ${=PY} run_linear_models.py $R > $C/linear_models.log 2>&1 || echo "FAILED linear models"
  step "bounds, first build"
  ROUGH=1 ${=PY} run_bounds.py $R > $C/bounds.log 2>&1 || echo "FAILED bounds"
  for b in 2e_16-300 2e_16-185 lcs1_16-185; do
    step "linear models $b"
    BAND=$b PARTS=gp MERGE_MODE=refine ${=PY} run_linear_models.py $R > $C/bands/$b/linear_models.log 2>&1 || echo "FAILED linear models $b"
    step "bounds $b"
    BAND=$b ROUGH=1 ${=PY} run_bounds.py $R > $C/bands/$b/bounds.log 2>&1 || echo "FAILED bounds $b"
  done
  step "merging at degree 300 (real and synthetic)"
  BAND=2e_16-300 env ${=MERGE} ${=PY} run_merge.py $R > $C/bands/2e_16-300/merge_points_sweep.log 2>&1 || echo "FAILED merge 300"
  BAND=2e_16-300 TRUTH_FROM=$C/realistic_synthetic.npz MERGE_MODE=refine GAMMA_REL=0.005 \
    ${=PY} run_realistic_synthetic.py $R > $C/bands/2e_16-300/realistic_synthetic.log 2>&1 || echo "FAILED realistic synthetic 300"
  step "compare"
  ${=PY} compare_bands.py $R > $C/bands/compare.log 2>&1 || echo "FAILED compare"
}

stage3() {
  local R=regions/australia_fine/config.toml C=regions/australia_fine/cache
  local BANDS=(syn_16-133 syn_f10_16-133 syn_16-185 syn_f10_16-185 syn_f10_16-300 syn_f30_16-300)
  step "multiscale truth"
  ${=PY} make_multiscale_truth.py $R > $C/multiscale_truth.log 2>&1 || echo "FAILED truth"
  step "build the synthetic bands"
  RUNS=${(j:,:)BANDS} ${=PY} build_bands.py $R > $C/bands/build_future.log 2>&1 || echo "FAILED build"
  for b in $BANDS; do
    step "linear models $b"
    BAND=$b CASES=cov_truth,hast,hast_var,hast_ms PARTS=gp,gp2 ${=PY} run_linear_models.py $R > $C/bands/$b/linear_models.log 2>&1 || echo "FAILED linear models $b"
    step "merging calibration $b"
    BAND=$b MODES=refine TAG=_sweep LAM_RELS=1 GAMMAS=0.005 BOOT=none ${=PY} run_merge.py $R > $C/bands/$b/merge_points_sweep.log 2>&1 || echo "FAILED merge $b"
    step "merging synthetic $b"
    BAND=$b TRUTH_FROM=$C/realistic_synthetic.npz MERGE_MODE=refine GAMMA_REL=0.005 N_BOOT=5 \
      ${=PY} run_realistic_synthetic.py $R > $C/bands/$b/realistic_synthetic.log 2>&1 || echo "FAILED realistic synthetic $b"
  done
  step "compare and maps"
  ${=PY} compare_future.py $R > $C/bands/compare_future.log 2>&1 || echo "FAILED compare"
  ${=PY} regions/australia_fine/plot_future.py > $C/plot_future.log 2>&1 || echo "FAILED maps"
}

for s in "$@"; do $s; done
step "all done"
