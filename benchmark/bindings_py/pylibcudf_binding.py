import rmm
import cupy as cp
import numpy as np
import time
import json
import gc
from functools import partial
from typing import Dict, Any, Tuple, Optional, List
from benchmark import WorkloadGenerator
from .base_binding import BaseBinding
from benchmark import get_spawn_ctx, get_fork_ctx
import pyarrow as pa
import pylibcudf as plc


def generate_records_chunk(chunk_size, start_idx, workload_gen):
    """Worker function to generate a chunk of records"""
    records = []
    for i in range(start_idx, start_idx + chunk_size):
        key = workload_gen.build_key_name(i, 8)
        data = workload_gen.build_values(key)
        record = {
            'key': key,
        }
        record = record | data
        records.append(record)
    return records


class PyLibCuDFBinding(BaseBinding):
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

        
        rmm.reinitialize(pool_allocator=True,
                         devices=self.gpu_device_id)
        
        self.table: plc.Table = None

    def load_data(self, start_key: int=None, end_key: int=None, load_data_output_file: Optional[str] = None, num_processes: int = 1) -> None:
        if load_data_output_file:
            load_data_list = []
        
        if start_key is not None and end_key is not None:
            assert end_key - start_key == self.num_records
        
        # Calculate chunk size for each process
        chunk_size = self.num_records // num_processes
        remaining = self.num_records % num_processes
        
        records = []
        if num_processes == 1:
            
            records = generate_records_chunk(self.num_records, 0, self.workload_gen)

        else:
            with get_fork_ctx().Pool(processes=num_processes) as pool:
                generate_chunk = partial(generate_records_chunk, workload_gen=self.workload_gen)
                
                chunk_args = []
                current_idx = 0
                if start_key is not None and end_key is not None:
                    current_idx = start_key
                    
                for _ in range(num_processes):
                    chunk_args.append((chunk_size, current_idx))
                    current_idx += chunk_size
                
                chunks = pool.starmap(generate_chunk, chunk_args)
            
                records = [record for chunk in chunks for record in chunk]
        
            if remaining > 0:
                record = generate_records_chunk(remaining, current_idx, self.workload_gen)
                if load_data_output_file:
                    load_data_list.extend(record)
                
                records.extend(record)
                
                
        
        # load data to pyarrow table
        arrow_tbl = pa.Table.from_pylist(records)
        self.table = plc.interop.from_arrow(arrow_tbl)
            
        print(f"LibCuDF Table loaded with {self.table.num_rows()} records")
        
        if load_data_output_file: 
            with open(load_data_output_file, 'w') as f:
                json.dump(load_data_list, f, indent=4)

    def multiget(self, keys: np.ndarray):
        return plc.batch_ops.multiget(self.table, keys, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    
    
    def multiset(self, key_list, value_list):
        self.table = plc.batch_ops.multiset(self.table, key_list, value_list)
    
    def results_postprocess(self, results):
        pa_results = plc.interop.to_arrow(results)
        pd_results = pa_results.to_pandas()
        pd_results.columns = ["key"] + self.workload_gen.fields
        return pd_results
    
    def check_data_integrity(self, keys, results):
        # convet keys to pyarrow array, then to plc Column
        keys_col = plc.interop.from_arrow(pa.array(keys))
        base_result = plc.copying.gather(self.table, keys_col, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
        base_result_arrow = plc.interop.to_arrow(base_result)
        
        result_arrow = plc.interop.to_arrow(results)
        integrity_check = base_result_arrow.equals(result_arrow)
        del base_result, base_result_arrow, result_arrow
        return integrity_check
            
    def init_each_thread(self):
        self.device.use()

    def cleanup(self):
        """Clean up GPU memory resources"""
        if hasattr(self, 'table'):
            del self.table
        
        cp.get_default_memory_pool().free_all_blocks()
        
        # Force complete cleanup
        gc.collect()


    def insert(self, key: str, values: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
        pass

    def read(self, key: str) -> Tuple[float, float, Dict[str, Any]]:
        pass

    def update(self, key: str, fieldkey: str, value: Any) -> Tuple[float, float, Dict[str, Any]]:
        pass

    def scan(self, start_key: str) -> Tuple[float, float, Dict[str, Any]]:
        pass

    


    
    