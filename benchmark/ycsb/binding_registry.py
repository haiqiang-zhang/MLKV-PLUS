from typing import Dict, Type
from bindings_py.BindingBase import BindingBase
import yaml
import importlib
import os

class BindingRegistry:
    def __init__(self):
        # bindings.yaml is in the bindings_py directory
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'bindings_py', 'bindings.yaml')
        self._load_config()
        
    def _load_config(self):
        """Load binding configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def get_binding_class(self, binding_name: str) -> BindingBase:
        """Get the binding class by name"""
        if binding_name not in self.config['bindings']:
            raise ValueError(f"Unknown binding: {binding_name}")
            
        class_path = self.config['bindings'][binding_name]['class']
        module_name, class_name = class_path.rsplit('.', 1)
        
        # Import the module and get the class
        module = importlib.import_module(f'benchmark.bindings_py.{module_name}')
        return getattr(module, class_name)
        
    def get_available_bindings(self) -> Dict[str, Dict]:
        """Get all available bindings with their descriptions"""
        return self.config['bindings'].copy()

# Create a singleton instance
registry = BindingRegistry()

# Convenience functions
get_binding_class = registry.get_binding_class
get_available_bindings = registry.get_available_bindings 