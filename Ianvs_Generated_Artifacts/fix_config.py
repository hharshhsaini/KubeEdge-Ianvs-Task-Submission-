import yaml
import sys

config_path = sys.argv[1]

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

for module in config['algorithm']['modules']:
    if module.get('type') == 'edgemodel':
        for hp in module.get('hyperparameters', []):
            if 'backend' in hp:
                hp['backend']['values'] = ['api']

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Config updated successfully - backend changed to 'api'")
