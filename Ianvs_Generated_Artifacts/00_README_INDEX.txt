============================================
  Ianvs Benchmark Evidence Files Index
============================================

This folder contains all evidence files from the
Ianvs Cloud-Edge LLM Benchmark runs.

Files:
------
01_gpqa_benchmark_output.log      - Full GPQA benchmark terminal output with leaderboard
02_mmlu_benchmark_output.log      - Full MMLU benchmark terminal output with leaderboard
03_docker_environment_info.txt    - Docker container status, image info, environment details
04_edge_model_fix.txt             - Patched edge_model.py showing LadeSpecDecLLM import fix
05_test_queryrouting_config.txt   - Final test_queryrouting.yaml configuration
06_testenv_config.txt             - Final testenv.yaml with dataset paths
07_benchmarkingjob_config.txt     - Final benchmarkingjob.yaml with workspace setting
08_Dockerfile.txt                 - Complete Dockerfile used for the Docker image build

How to use with RUNLOG.md:
--------------------------
Each file is referenced in RUNLOG.md with a placeholder like:
  Screenshot placeholder: -> 01_gpqa_benchmark_output.log

You can take screenshots of these files and insert your own
screenshot images at those placeholder locations in the doc.

Date: 2026-02-15
