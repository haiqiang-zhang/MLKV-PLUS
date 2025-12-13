import torch
import torch.distributed as dist
import numpy as np
import json
import gc
from functools import partial
from typing import Dict, Any, Tuple, Optional, List
from benchmark import WorkloadGenerator
from .DistBindingBase import DistBindingBase
import multiprocessing as mp

import mlkv_plus
from mlkv_plus import communication

class DistMLKVPlusBinding(DistBindingBase):
    def __init__(self, workload_gen: WorkloadGenerator, 
                 max_batch_size: int,
                 num_records: int, 
                 dim: int,
                 distribution: str = 'zipfian', 
                 scan_size: int = 100,
                 binding_config: dict = {}):
        self.num_records = num_records
        self.scan_size = scan_size
        self.distribution = distribution
        self.fields = []
        self.workload_gen = workload_gen
        self.dim = dim
        self.binding_config = binding_config
        self.max_batch_size = max_batch_size
        self.max_hbm_for_vectors_gb = self.binding_config.get('max_hbm_for_vectors_gb', 2)
        self.gpu_init_capacity = self.binding_config.get('gpu_init_capacity', 1000000)
        self.gpu_max_capacity = self.binding_config.get('gpu_max_capacity', 10000000)
        
        

        mlkv_plus.init(comm_tool="torch_dist")
        
        # Create MLKVPlusDB instance
        self.db = mlkv_plus.MLKVPlusDB(
            dim=self.dim,
            max_hbm_for_vectors_gb=self.max_hbm_for_vectors_gb,
            hkv_io_by_cpu=False,
            gpu_id=torch.cuda.current_device(),
            create_if_missing=True,
            gpu_init_capacity=self.gpu_init_capacity,
            gpu_max_capacity=self.gpu_max_capacity,
            max_batch_size=self.max_batch_size,
            rocksdb_path=f"/tmp/mlkv_plus_rocksdb_{communication.rank()}"
        )

    def load_data(self, num_records, num_processes=1) -> None:
        
        print(f"Rank {communication.rank()}: Loading data...")
        
        if self.rank() == 0:
            keys, values = self.workload_gen.generate_load_data(num_records, num_processes=num_processes)
            
            # convert to torch
            keys_tensor = torch.tensor(keys, dtype=torch.int64, device=torch.device('cuda'))
            values_tensor = torch.tensor(values, dtype=torch.float32, device=torch.device('cuda'))
        else:
            keys_tensor = torch.zeros(num_records, dtype=torch.int64, device=torch.device('cuda'))
            values_tensor = torch.zeros((num_records, self.dim), dtype=torch.float32, device=torch.device('cuda'))
        
        self.broadcast(keys_tensor, root=0)
        self.broadcast(values_tensor, root=0)
        
        # Convert to tensors and insert into MLKVPlus
        batch_size = 10000
        for i in range(0, keys_tensor.shape[0], batch_size):
            end_idx = min(i + batch_size, keys_tensor.shape[0])
            batch_keys = keys_tensor[i:end_idx]
            batch_values = values_tensor[i:end_idx]
            
            # Use distributed assign
            mlkv_plus.dist_assign(self.db, batch_keys, batch_values)
            
            print(f"Rank {communication.rank()}: Inserted {end_idx} records into the MLKVPlus table...")
            
            
        print("Rank {communication.rank()}: Rank {communication.rank()} finished loading data...")
        
        communication.barrier()


    def multiget(self, keys, stream=None):
        """Perform distributed multiget operation"""
        # Convert keys to tensor if needed
        if not isinstance(keys, torch.Tensor):
            keys_tensor = torch.tensor(keys, dtype=torch.int64,
                                     device=torch.device('cuda', torch.cuda.current_device()))
        else:
            keys_tensor = keys
        
        # Perform all2all dense embedding lookup
        result = mlkv_plus.all2all_dense_embedding(self.db, keys_tensor)

        
        return 0, 0  # Return dummy timing values for compatibility
    
    def multiset(self, key_list, value_list):
        """Perform distributed multiset operation"""
        # Convert to tensors if needed
        if not isinstance(key_list, torch.Tensor):
            keys_tensor = torch.tensor(key_list, dtype=torch.int64,
                                     device=torch.device('cuda', torch.cuda.current_device()))
        else:
            keys_tensor = key_list
            
        if not isinstance(value_list, torch.Tensor):
            values_tensor = torch.tensor(value_list, dtype=torch.float32,
                                       device=torch.device('cuda', torch.cuda.current_device()))
        else:
            values_tensor = value_list
        
        # Reshape values if needed
        if values_tensor.dim() == 1:
            values_tensor = values_tensor.view(-1, self.dim)
        
        # Perform distributed assign
        mlkv_plus.dist_assign(self.db, keys_tensor, values_tensor)
        
        return 0, 0  # Return dummy timing values for compatibility
            

    def cleanup(self):
        """Clean up GPU memory resources"""
        if hasattr(self, 'db'):
            del self.db
        
        torch.cuda.empty_cache()
        
        # Force complete cleanup
        gc.collect()

    def insert(self, key: str, values: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        """Single insert operation - not implemented for distributed case"""
        pass

    def read(self, key: str) -> Tuple[float, float, Dict[str, Any]]:
        """Single read operation - not implemented for distributed case"""
        pass

    def update(self, key: str, fieldkey: str, value: Any) -> Tuple[float, float, Dict[str, Any]]:
        """Single update operation - not implemented for distributed case"""
        pass

    def scan(self, start_key: str) -> Tuple[float, float, Dict[str, Any]]:
        """Scan operation - not implemented for distributed case"""
        pass

    def rank(self):
        return communication.rank()
    
    def num_ranks(self):
        return communication.num_ranks()
    
    def broadcast(self, tensor, root=0):
        return communication.broadcast(tensor, root)

    def barrier(self):
        communication.barrier()
    
    def allreduce(self, tensor, op):
        return communication.allreduce(tensor, op)
    
    
    def broadcast_object_list(self, obj, root=0):
        return dist.broadcast_object_list(obj, root)