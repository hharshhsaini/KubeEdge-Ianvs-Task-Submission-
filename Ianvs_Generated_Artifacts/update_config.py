import yaml
import sys

config_path = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "gpqa"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

if mode == "gpqa":
    # Match the GPQA cache Entry 1: NousResearch/Llama-2-7b-chat-hf with huggingface backend
    # Cache config: {"model": "NousResearch/Llama-2-7b-chat-hf", "backend": "huggingface", 
    #                "draft_model": "yuhuili/EAGLE-llama2-chat-7B", "temperature": 1e-07,
    #                "top_p": 0.9, "max_tokens": 1024, "repetition_penalty": 1, "use_cache": true}
    for module in config['algorithm']['modules']:
        if module.get('type') == 'edgemodel':
            module['hyperparameters'] = [
                {'model': {'values': ['NousResearch/Llama-2-7b-chat-hf']}},
                {'backend': {'values': ['huggingface']}},
                {'draft_model': {'values': ['yuhuili/EAGLE-llama2-chat-7B']}},
                {'temperature': {'values': [0.0000001]}},
                {'top_p': {'values': [0.9]}},
                {'max_tokens': {'values': [1024]}},
                {'repetition_penalty': {'values': [1]}},
                {'use_cache': {'values': [True]}},
            ]
            
        if module.get('type') == 'cloudmodel':
            module['hyperparameters'] = [
                {'api_provider': {'values': ['openai']}},
                {'model': {'values': ['gpt-4o-mini']}},
                {'api_key_env': {'values': ['OPENAI_API_KEY']}},
                {'api_base_url': {'values': ['OPENAI_BASE_URL']}},
                {'temperature': {'values': [0.9]}},
                {'top_p': {'values': [0.9]}},
                {'max_tokens': {'values': [1024]}},
                {'repetition_penalty': {'values': [1.05]}},
                {'use_cache': {'values': [True]}},
            ]

        if module.get('type') == 'hard_example_mining':
            module['name'] = 'EdgeOnly'

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Config updated for {mode} mode")
