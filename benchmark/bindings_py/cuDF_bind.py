import rmm
import cudf
import cupy as cp
import numpy as np
import time

import json
import gc
import rmm
from functools import partial
from typing import Dict, Any, Tuple, Optional, List
from benchmark import WorkloadGenerator
from .base_binding import BaseBinding
from benchmark import get_spawn_ctx, get_fork_ctx
from typing import overload



def generate_records_chunk(chunk_size, start_idx, workload_gen):
    """Worker function to generate a chunk of records"""
    records = []
    for i in range(start_idx, start_idx + chunk_size):
        key = workload_gen.build_key_name(i, 8)
        data = workload_gen.build_values()
        record = {'key': key} | data
        records.append(record)
    return records

class CuDFBinding(BaseBinding):
    def __init__(self, workload_gen: WorkloadGenerator, num_records: int = 1_000_000, 
                 distribution: str = 'zipfian', 
                 scan_size: int = 100,
                 gpu_device: List[int] = [0]):
        self.num_records = num_records
        self.scan_size = scan_size
        self.distribution = distribution
        self.fields = []
        self.workload_gen = workload_gen

        
        
        self.gpu_device_id = gpu_device[0]
        self.device = cp.cuda.Device(self.gpu_device_id)
        self.device.use()
        
        self.cudf_ins = cudf.DataFrame()

    def load_data(self, start_key: int=None, end_key: int=None, load_data_output_file: Optional[str] = None, num_processes: int = 1) -> None:
        # Load data into the table
        
        if load_data_output_file:
            load_data_list = []
        
        if start_key is not None and end_key is not None:
            assert end_key - start_key == self.num_records
        
        # Calculate chunk size for each process
        chunk_size = self.num_records // num_processes
        remaining = self.num_records % num_processes
        
        
        if num_processes == 1:
            records = []
            for i in range(self.num_records):
                key = self.workload_gen.build_key_name(i, 8)
                data = self.workload_gen.build_values()
                record = [key] + list(data.values())
                records.append(record)
                
                if load_data_output_file:
                    load_data_list.append(record)
                
                # Print progress every 100,000 records
                if (i + 1) % 100000 == 0:
                    print(f"Loaded {i + 1} records")
        else:
            # Create a pool of workers
            with get_fork_ctx().Pool(processes=num_processes) as pool:
                # Create partial function with fixed arguments
                generate_chunk = partial(generate_records_chunk, workload_gen=self.workload_gen)
                
                # Generate chunks in parallel
                chunk_args = []
                if start_key is not None and end_key is not None:
                    current_idx = start_key
                else:
                    current_idx = 0
                for _ in range(num_processes):
                    chunk_args.append((chunk_size, current_idx))
                    current_idx += chunk_size
                
                # Use starmap instead of map to unpack the arguments
                chunks = pool.starmap(generate_chunk, chunk_args)
            
            # Process chunks and create DataFrame
            for i, chunk in enumerate(chunks):
                if load_data_output_file:
                    load_data_list.extend(chunk)
                    
                batch_df = cudf.DataFrame(chunk)
                self.cudf_ins = cudf.concat([self.cudf_ins, batch_df])
                del batch_df
                gc.collect()
        
            # Process remaining records
            if remaining > 0:
                record = generate_records_chunk(remaining, current_idx, self.workload_gen)
                if load_data_output_file:
                    load_data_list.extend(record)
                batch_df = cudf.DataFrame(record)
                self.cudf_ins = cudf.concat([self.cudf_ins, batch_df])
                del batch_df
                gc.collect()
            
            
            
        print(f"cuDF DataFrame loaded with {self.cudf_ins.shape[0]} records")
        
        if load_data_output_file: 
            with open(load_data_output_file, 'w') as f:
                json.dump(load_data_list, f, indent=4)
                
                

    def insert(self, key: str, values: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        op_detail = {
            'operation_type': 'insert',
            'key': key,
            'values': values
        }
        
        self.cudf_ins = cudf.concat([
            self.cudf_ins,
            cudf.DataFrame({'key': key} | values)
        ])
        self.num_records += 1
        
        return 0, 0, op_detail

    def read(self, key: str) -> Tuple[float, float, Dict[str, Any]]:
        op_detail = {
            'operation_type': 'read',
            'key': key
        }
        
        self.cudf_ins[self.cudf_ins['key'] == key]
        
        return 0, 0, op_detail

    def update(self, key: str, fieldkey: str, value: Any) -> Tuple[float, float, Dict[str, Any]]:
        op_detail = {
            'operation_type': 'update',
            'key': key,
            'field': fieldkey,
            'value': value
        }
        
        self.cudf_ins.loc[self.cudf_ins['key'] == key, fieldkey] = value
        
        return 0, 0, op_detail

    def scan(self, start_key: str) -> Tuple[float, float, Dict[str, Any]]:
        op_detail = {
            'operation_type': 'scan',
            'start_key': start_key,
            'scan_size': self.scan_size
        }
        
        # Get the next scan_size keys after start_key
        df = self.cudf_ins[self.cudf_ins['key'] >= start_key].head(self.scan_size)
        
        return 0, 0, op_detail

    def multiget(self, keys: np.ndarray, stream: cp.cuda.Stream=None) -> cudf.DataFrame:
        # if stream is not None:
        #     with stream:
        #         mask = self.cudf_ins.iloc[:, 0].isin(keys)
        #         return self.cudf_ins[mask]
        # else:
        #     mask = self.cudf_ins.iloc[:, 0].isin(keys)
        #     return self.cudf_ins[mask]
        return self.cudf_ins.take(keys)

    def multiset(self, key_list: np.ndarray, value_list):
        # Create a new DataFrame with the updated records
        new_records = cudf.DataFrame(value_list)
        new_records['key'] = key_list
        
        # drop duplicate keys
        new_records = new_records.drop_duplicates(subset=['key'], keep='last')
        
        # Update the DataFrame with new records
        self.cudf_ins.update(new_records)
    
    
    
    def groupby(self, groupby_key: str, groupby_value: str) -> cudf.DataFrame:
        return self.cudf_ins.groupby(groupby_key)[groupby_value].sum()
    
    def init_each_thread(self):
        self.device.use()

    def cleanup(self):
        """Clean up GPU memory resources"""
        if hasattr(self, 'cudf_ins'):
            del self.cudf_ins
        
        cp.get_default_memory_pool().free_all_blocks()
        
        # Force complete cleanup
        gc.collect()


