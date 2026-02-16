#!/bin/bash
set -e

cd /ianvs

# Step 1: Update testenv.yaml to point to MMLU dataset
TESTENV="examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml"
sed -i 's|./dataset/gpqa/train_data/data.json|./dataset/mmlu-5-shot/train_data/data.json|g' "$TESTENV"
sed -i 's|./dataset/gpqa/test_data/metadata.json|./dataset/mmlu-5-shot/test_data/metadata.json|g' "$TESTENV"
echo "Updated testenv.yaml for MMLU dataset"

# Step 2: Update benchmarkingjob.yaml to use workspace-mmlu
BENCHJOB="examples/cloud-edge-collaborative-inference-for-llm/benchmarkingjob.yaml"
sed -i 's|workspace-gpqa|workspace-mmlu|g' "$BENCHJOB"
echo "Updated benchmarkingjob.yaml for MMLU workspace"

# Step 3: Update test_queryrouting.yaml to match MMLU cache configs
source activate ianvs-experiment

cat > /ianvs/update_mmlu_config.py << 'PYEOF'
import yaml
import sys

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

for module in config['algorithm']['modules']:
    if module.get('type') == 'edgemodel':
        module['hyperparameters'] = [
            {'model': {'values': ['Qwen/Qwen2.5-7B-Instruct']}},
            {'backend': {'values': ['vllm']}},
            {'temperature': {'values': [0]}},
            {'top_p': {'values': [0.8]}},
            {'max_tokens': {'values': [512]}},
            {'repetition_penalty': {'values': [1.05]}},
            {'tensor_parallel_size': {'values': [4]}},
            {'gpu_memory_utilization': {'values': [0.9]}},
            {'use_cache': {'values': [True]}},
        ]
    if module.get('type') == 'cloudmodel':
        module['hyperparameters'] = [
            {'api_provider': {'values': ['openai']}},
            {'model': {'values': ['gpt-4o-mini']}},
            {'api_key_env': {'values': ['OPENAI_API_KEY']}},
            {'api_base_url': {'values': ['OPENAI_BASE_URL']}},
            {'temperature': {'values': [0]}},
            {'top_p': {'values': [0.8]}},
            {'max_tokens': {'values': [512]}},
            {'repetition_penalty': {'values': [1.05]}},
            {'use_cache': {'values': [True]}},
        ]
    if module.get('type') == 'hard_example_mining':
        module['name'] = 'EdgeOnly'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("test_queryrouting.yaml updated for MMLU cache")
PYEOF

python update_mmlu_config.py examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml

echo "All MMLU configs ready!"
