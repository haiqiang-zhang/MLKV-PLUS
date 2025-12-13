import cupy as cp
import numpy as np

import json
import gc
from functools import partial
from typing import Dict, Any, Tuple, Optional, List
from benchmark import WorkloadGenerator
from .base_binding import BaseBinding
import multiprocessing as mp

import merlin_hashtable_python.merlin_hashtable_python as mh


def pack_strings(str_list, width=10):
    """
    Pack a 2D array of strings into a uint8 numpy array
    
    Args:
        str_list: 2D list of strings, shape (n_rows, n_cols)
        width: maximum width for each string (will be padded/truncated to this size)
    
    Returns:
        numpy array of shape (n_rows, n_cols * width) with dtype uint8
    """
    if not str_list:
        return np.array([], dtype=np.uint8)
    
    # Handle both 1D and 2D cases
    if isinstance(str_list[0], str):
        # 1D case - convert to 2D with single row
        str_list = [str_list]
    
    n_rows = len(str_list)
    n_cols = len(str_list[0]) if str_list else 0
    
    # Create output array
    result = np.zeros((n_rows, n_cols * width), dtype=np.uint8)
    
    for i, row in enumerate(str_list):
        for j, s in enumerate(row):
            # Convert string to bytes if needed
            if isinstance(s, str):
                s_bytes = s.encode('utf-8')
            else:
                s_bytes = s
            
            # Pad or truncate to width
            s_padded = s_bytes.ljust(width, b'\0')[:width]
            
            # Copy to result array
            start_idx = j * width
            end_idx = start_idx + width
            result[i, start_idx:end_idx] = np.frombuffer(s_padded, dtype=np.uint8)
    
    return result

def unpack_strings(uint8_arr, width=10):
    """
    Unpack a uint8 numpy array back to 2D list of strings
    
    Args:
        uint8_arr: numpy array of shape (n_rows, n_cols * width) with dtype uint8
        width: width of each packed string
    
    Returns:
        2D list of strings
    """
    if uint8_arr.size == 0:
        return []
    
    n_rows, total_width = uint8_arr.shape
    n_cols = total_width // width
    
    result = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            start_idx = j * width
            end_idx = start_idx + width
            string_bytes = uint8_arr[i, start_idx:end_idx].tobytes()
            # Remove null padding and decode
            string_val = string_bytes.rstrip(b'\0').decode('utf-8')
            row.append(string_val)
        result.append(row)
    
    return result


def generate_records_chunk(chunk_size, start_idx, workload_gen):
    """Worker function to generate a chunk of records"""
    keys = []
    values = []
    for i in range(start_idx, start_idx + chunk_size):
        key = workload_gen.build_key_name(i, 8)
        data = workload_gen.build_values(key)
        keys.append(key)
        temp_values = []
        for k, v in data.items():
            temp_values.append(v)
        values.append(temp_values)
        
    return keys, values


class HKVBinding(BaseBinding):
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
        
        options = mh.HashTableOptions()
        options.dim = self.workload_gen.field_count
        options.init_capacity = 12 * 1024 * 1024
        options.max_capacity = 12 * 1024 * 1024
        options.max_hbm_for_vectors = 12 * 1024 * 1024 * 1024
        options.io_by_cpu = False
        options.device_id = self.gpu_device_id
        
        self.options = options

        self.table = mh.HashTable()
        self.table.init(self.options)
        
        
        self.d_values_out = cp.empty((self.num_records,self.options.dim*self.workload_gen.max_field_length), dtype=cp.int8)
        self.d_found_out = cp.empty((self.num_records,self.options.dim*self.workload_gen.max_field_length), dtype=cp.bool_)

    def load_data(self, start_key: int=None, end_key: int=None, load_data_output_file: Optional[str] = None, num_processes: int = 1) -> None:
        if load_data_output_file:
            load_data_list = []
        
        if start_key is not None and end_key is not None:
            assert end_key - start_key == self.num_records
        
        # Calculate chunk size for each process
        chunk_size = self.num_records // num_processes
        remaining = self.num_records % num_processes
        
        total_keys = []
        total_values = []
        if num_processes == 1:
            
            keys, values = generate_records_chunk(self.num_records, 0, self.workload_gen)
            total_keys.extend(keys)
            total_values.extend(values)

        else:
            with mp.get_context('fork').Pool(processes=num_processes) as pool:
                generate_chunk = partial(generate_records_chunk, workload_gen=self.workload_gen)
                
                chunk_args = []
                current_idx = 0
                if start_key is not None and end_key is not None:
                    current_idx = start_key
                    
                for _ in range(num_processes):
                    chunk_args.append((chunk_size, current_idx))
                    current_idx += chunk_size
                
                chunks = pool.starmap(generate_chunk, chunk_args)
            
                keys, values = [key for chunk in chunks for key in chunk[0]], [value for chunk in chunks for value in chunk[1]]
                total_keys.extend(keys)
                total_values.extend(values)
        
            if remaining > 0:
                keys, values = generate_records_chunk(remaining, current_idx, self.workload_gen)
                if load_data_output_file:
                    load_data_list.extend(keys)
                
                total_keys.extend(keys)
                total_values.extend(values)
                
        assert len(total_keys) == len(total_values)
        assert len(total_keys) == self.num_records
        
        print(f"Inserting {self.num_records} records into the HKV table...")
        
        
        # batch insert into the table
        for i in range(0, self.num_records, 1000):
            d_keys = cp.array(total_keys[i:i+1000], dtype=cp.int64)
            d_values = cp.asarray(pack_strings(total_values[i:i+1000]), blocking=True)
            self.table.insert_or_assign(
                len(d_keys),
                d_keys.data.ptr,
                d_values.data.ptr
            )
            cp.cuda.Stream.null.synchronize()
            print(f"Inserted {i+1000} records into the HKV table...")
        
        
        if load_data_output_file: 
            with open(load_data_output_file, 'w') as f:
                json.dump(load_data_list, f, indent=4)

    def multiget(self, keys, stream: cp.cuda.Stream=None):
        d_keys = cp.asarray(keys, order='C', blocking=True)
        
        self.table.find(
            len(keys),
            d_keys.data.ptr,
            self.d_values_out.data.ptr,
            self.d_found_out.data.ptr
        )
        
        cp.cuda.Stream.null.synchronize()
        
        return 0, 0
    
    def multiset(self, key_list, value_list):
        pass
    
    # def results_postprocess(self, results):
    #     pa_results = plc.interop.to_arrow(results)
    #     pd_results = pa_results.to_pandas()
    #     pd_results.columns = ["key"] + self.workload_gen.fields
    #     return pd_results
    
    # def check_data_integrity(self, keys, results):
    #     # convet keys to pyarrow array, then to plc Column
    #     keys_col = plc.interop.from_arrow(pa.array(keys))
    #     base_result = plc.copying.gather(self.table, keys_col, plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    #     base_result_arrow = plc.interop.to_arrow(base_result)
        
    #     result_arrow = plc.interop.to_arrow(results)
    #     integrity_check = base_result_arrow.equals(result_arrow)
    #     del base_result, base_result_arrow, result_arrow
    #     return integrity_check
            
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

    


    
    