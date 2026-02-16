#!/bin/bash
set -e

EDGE_MODEL="/ianvs/examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/edge_model.py"

# Fix 1: Make LadeSpecDecLLM import optional
sed -i 's/from models import HuggingfaceLLM, APIBasedLLM, VllmLLM, EagleSpecDecModel, LadeSpecDecLLM/from models import HuggingfaceLLM, APIBasedLLM, VllmLLM, EagleSpecDecModel\ntry:\n    from models import LadeSpecDecLLM\nexcept ImportError:\n    LadeSpecDecLLM = None/' "$EDGE_MODEL"

echo "Fix 1 applied: LadeSpecDecLLM import made optional"

# Fix 2: Update config to match GPQA cache
source activate ianvs-experiment
cd /ianvs
python update_config.py examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml gpqa

echo "Fix 2 applied: Config updated for GPQA cache"
echo "All fixes applied successfully!"
