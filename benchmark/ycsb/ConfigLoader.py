import yaml
from typing import Dict, Any
import os

def __load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def __merge_configs(global_config: Dict[str, Any], workload_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge global and workload configurations, with workload taking precedence."""
    merged = global_config.copy()
    merged.update(workload_config)
    return merged


def get_available_workloads():
    config_path = os.path.join(os.path.dirname(__file__), "..", "workload_config.yaml")
    config = __load_config(config_path)
    return list(config['workloads'].keys())

def get_workload_config(workload_name: str=None):
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "workload_config.yaml")
    config = __load_config(config_path)
    
    if workload_name is None:
        return config['global']
    
    if workload_name not in config['workloads']:
        raise ValueError(f"Workload {workload_name} not found in config")
    
    # Merge configurations
    final_config = __merge_configs(config['global'], config['workloads'][workload_name])
    
    # Display final configuration
    print(f"Selected Configuration: {workload_name}")
    print("-" * 50)
    print(f"Workload: {config['workloads'][workload_name]['name']}")
    print("\nFinal Settings:")
    for key, value in final_config.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    return final_config


def get_binding_config(binding_name: str):
    config_path = os.path.join(os.path.dirname(__file__), "..", "binding_config.yaml")
    config = __load_config(config_path)
    return config[binding_name]