"""
Python wrapper for YCSB benchmark binding.

This module provides a convenient Python interface to the YCSB benchmark
implemented in C++/CUDA.
"""

import numpy as np
from typing import List

try:
    from ycsb_binding import ycsb_binding
except (ModuleNotFoundError, ImportError) as e:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    import ycsb_binding
except:
    ycsb_binding = None
    
    
class Operation:
    def __init__(self, op: str, keys: List[int | np.ndarray], values: List[float | np.ndarray] = []):

        self.op = op
        self.keys = keys
        self.values = values
    
class YCSBBridge:
    def __init__(self, key_type: str, value_type: str, binding: str):
        self._bridge = None
        self._operation_class = None
        
        # check if binding is cuda or cpu
        available_bindings = self.get_available_bindings()
        if binding not in available_bindings["cuda"] and binding not in available_bindings["cpu"]:
            raise ValueError(f"Binding {binding} is not available")
        self._isCuda = binding in available_bindings["cuda"]
        
        self._typed_bridge_factory(key_type, value_type, self._isCuda)
        self._bridge = self._bridge(binding)
        
    def initialize_cuda(self, gpu_init_capacity: int, gpu_max_capacity: int, dim: int, hbm_gb: int, gpu_ids: List[int], max_batch_size: int, binding_config: str=""):
        if not self._isCuda:
            raise ValueError("Bridge is not CUDA")
        self._bridge.initialize(gpu_init_capacity, gpu_max_capacity, dim, hbm_gb, gpu_ids, max_batch_size, binding_config)
        
    def initialize_cpu(self, dim: int, max_batch_size: int, binding_config: str=""):
        if self._isCuda:
            raise ValueError("Bridge is CUDA")
        self._bridge.initialize(dim, max_batch_size, binding_config)
        
    def multiset(self, batch_size: int, keys: np.ndarray, values: np.ndarray):
        self._bridge.multiset(batch_size, keys, values)
        
    def run_benchmark_cuda(self, ops: List[Operation], num_streams: int=1, data_integrity: str="YCSB"):
        if not self._isCuda:
            raise ValueError("Bridge is not CUDA")
        ops = [self._operation_class(op.op, op.keys, op.values) for op in ops]
        return self._bridge.run_benchmark(ops, num_streams, data_integrity)
    
    def run_benchmark_cpu(self, ops: List[Operation], data_integrity: str="YCSB"):
        ops = [self._operation_class(op.op, op.keys, op.values) for op in ops]
        return self._bridge.run_benchmark(ops, data_integrity)
        
    def cleanup(self):
        self._bridge.cleanup()
        
    def _typed_bridge_factory(self, key_type: str, value_type: str, isCuda: bool=False):
        """
        Factory function to create a typed bridge.
        """
        if key_type == "uint64" and value_type == "double":
            self._bridge = ycsb_binding.YCSBBridgeCUDA_uint64_double if isCuda else ycsb_binding.YCSBBridgeCPU_uint64_double
            self._operation_class = ycsb_binding.Operation_uint64_double
        else:
            raise ValueError(f"Unsupported key-value type combination: {key_type}, {value_type}")
        
        
    @staticmethod
    def get_available_bindings():
        if ycsb_binding is not None:
            cuda_bindings = ycsb_binding.YCSBBridgeCUDA_uint64_double.get_available_bindings()
            cpu_bindings = ycsb_binding.YCSBBridgeCPU_uint64_double.get_available_bindings()
            return {"cuda": cuda_bindings, "cpu": cpu_bindings}
        else:
            return {}
        
        
        
if __name__ == "__main__":
    available_bindings = YCSBBridge.get_available_bindings()
    print(available_bindings)
    # bridge = YCSBBridge("uint64", "double", "rocksdb")