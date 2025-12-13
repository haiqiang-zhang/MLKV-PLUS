import cupy as cp
import random
import json
import pandas as pd
from .binding_registry import get_binding_class
from .WorkloadGenerator import WorkloadGenerator, DataIntegrity
from concurrent.futures import ProcessPoolExecutor, as_completed as as_completed_process
import multiprocessing as mp
from benchmark.ycsb.YCSBController import YCSBController, check_rows_integrity_ycsb
from typing import override
import torch


class DistYCSBController(YCSBController):
    @override
    def __init__(self, num_records=1_000_000, 
                 operations=[('read', 0.5), ('update', 0.5)], 
                 workload_name='',
                 distribution='zipfian', 
                 zipfian_theta=0.99,
                 scan_size=100, 
                 orderedinserts=False,
                 data_integrity="NOT_CHECK",
                 target_qps=None,
                 min_field_length=128,
                 max_field_length=512,
                 field_count=5,
                 value_type: str = "float",
                 binding_name: str = "DistMLKVPlusBinding",
                 output_file=None,
                 load_data_output_file=None,
                 operations_file=None,
                 binding_config={},
                 generator_num_processes=8):
        self.num_records = num_records
        self.operations = operations
        self.workload_name = workload_name
        self.scan_size = scan_size
        self.distribution = distribution
        self.zipfian_theta = zipfian_theta
        self.output_file = output_file
        self.load_data_output_file = load_data_output_file
        self.operations_file = operations_file
        self.orderedinserts = orderedinserts
        self.data_integrity = DataIntegrity[data_integrity]
        self.min_field_length = min_field_length
        self.max_field_length = max_field_length
        self.value_type = value_type
        self.binding_name = binding_name
        self.binding_config = binding_config
        self.field_count = field_count
        self.max_batch_size = max(param[1] if isinstance(param, list) else 1 for op, param in self.operations.items())
        self.generator_num_processes = generator_num_processes
        
        if target_qps is not None and target_qps < 0:
            raise ValueError("Target QPS must be greater than 0 or 0 for no QPS control")
        elif target_qps == 0:
            self.target_qps = None
        else:
            self.target_qps = target_qps
            
        self.workload_gen = WorkloadGenerator(data_integrity=self.data_integrity, 
                                              distribution=self.distribution, 
                                              zipfian_theta=self.zipfian_theta,
                                              orderedinserts=self.orderedinserts,
                                              min_field_length=self.min_field_length,
                                              max_field_length=self.max_field_length,
                                              field_count=self.field_count,
                                              value_type=self.value_type)
        
        binding_class = get_binding_class(binding_name)
        
        self.bind_instance = binding_class(
            workload_gen=self.workload_gen,
            num_records=self.num_records,
            distribution=self.distribution,
            scan_size=self.scan_size,
            dim=self.field_count,
            max_batch_size=self.max_batch_size,
            binding_config=self.binding_config
        )

        num_processes = mp.get_context('fork').cpu_count() - self.bind_instance.num_ranks() - 1
        self._load_data(num_processes=num_processes)


    @override
    def _load_data(self, num_processes=1):
        self.bind_instance.load_data(self.num_records, num_processes=num_processes)
            


    @override
    def run(self, num_ops=10_000, num_streams=16, save_ops_details=False, keys_type='np_int64'):
        total_time = 0
        
        # Calculate total operations including existing ones
        total_ops = 0
        for op, param in self.operations.items():
            op_count = int(param[0] * num_ops) if isinstance(param, list) else int(param * num_ops)
            if isinstance(param, list):
                total_ops += (op_count * param[1])
            else:
                total_ops += op_count
        
        total_time, integrity, integrity_accuracy = self.run_py(num_ops, num_streams, save_ops_details, keys_type)
        
        
        self.bind_instance.barrier()
        

        # Gather timing results from all ranks
        rank = self.bind_instance.rank()
        world_size = self.bind_instance.num_ranks()
        
        # Convert to tensors for all_reduce
        total_time_tensor = torch.tensor(total_time, device=f'cuda:{torch.cuda.current_device()}')
        total_ops_tensor = torch.tensor(total_ops, dtype=torch.float32, device=f'cuda:{torch.cuda.current_device()}')
        
        # All-reduce to get average timing across all ranks
        self.bind_instance.allreduce(total_time_tensor, op="sum")
        self.bind_instance.allreduce(total_ops_tensor, op="sum")
        
        # Calculate global averages
        avg_total_time = total_time_tensor.item() / world_size
        global_total_ops = total_ops_tensor.item()
        
        throughput = global_total_ops / avg_total_time
        avg_latency_per_query = avg_total_time / (global_total_ops / world_size) * 1000
        avg_latency_per_batch_op = avg_total_time / num_ops * 1000
        
        # Only rank 0 prints and saves results
        if rank == 0:
            print(f"\nDistributed Results (World Size: {world_size}):")
            print(f"Average total time across ranks: {avg_total_time:.4f}s")
            print(f"Global total operations: {global_total_ops}")

        
         # Prepare results
        results = {
            'workload': self.workload_name,
            'distribution': self.distribution,
            'num_records': self.num_records,
            'num_batch_ops': num_ops,
            'avg_latency_per_query_ms': avg_latency_per_query,
            'avg_latency_per_batch_op_ms': avg_latency_per_batch_op,
            'throughput': throughput,
            'total_time': total_time,
            'binding': self.bind_instance.__class__.__name__,
            'num_streams': num_streams,
            'min_field_length': self.min_field_length,
            'max_field_length': self.max_field_length,
            'orderedinserts': self.orderedinserts,
            'data_integrity': self.data_integrity.name,
            'integrity': integrity,
            'integrity_accuracy': integrity_accuracy,
            'distributed': True,
            'world_size': self.bind_instance.num_ranks()
        }
        
        if self.distribution == 'zipfian':
            results['zipfian_theta'] = self.zipfian_theta
        
        for op, param in self.operations.items():
            if isinstance(param, list):
                results[f"{op}_prob"] = param[0]
                results[f'{op}_batch_size'] = param[1]
            else:
                results[f"{op}_prob"] = param
                
        
        if self.bind_instance.rank() == 0:
            print(f"\nWorkload {self.workload_name} Results:")
            print(f"Distribution: {self.distribution}")
            print(f"Total Operations: {total_ops}")
            print(f"Number of batches: {num_ops}")
            print(f"Average Latency per query: {results['avg_latency_per_query_ms']} ms")
            print(f"Average Latency per batch op: {results['avg_latency_per_batch_op_ms']} ms")
            print(f"Throughput: {results['throughput']} ops/sec")
            print(f"Integrity: {integrity}")
            print(f"Integrity Accuracy: {integrity_accuracy}")
            print(f"Total Time: {results['total_time']} seconds")
            print(f"World Size: {self.bind_instance.num_ranks()}")
            
        # Write results to file if specified (only rank 0 in distributed mode)
        if self.output_file and self.bind_instance.rank() == 0:
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=4)
                print(f"\nResults have been written to {self.output_file}")
                
                
                
        self.bind_instance.cleanup()
        
        return results 
        
    @override
    def run_py(self, num_ops=10_000, num_streams=16, save_ops_details=False, keys_type='int'):
        print(f"Starting workload {self.operations} with {num_ops} operations...")
        if self.target_qps:
            print(f"Target QPS: {self.target_qps}")
        # Try to load operations from file
        loaded_operation_data = self._load_operations()
        
        # generate operations through one rank
        if self.bind_instance.rank() == 0:
            all_operation_data, all_op_list = self._generate_operations(loaded_operation_data, num_ops, keys_type=keys_type)
            random.shuffle(all_op_list)
            payload = {
                'all_operation_data': all_operation_data,
                'all_op_list': all_op_list
            }
            
        else:
            payload = {}         

        payload = [payload]
        self.bind_instance.broadcast_object_list(payload, 0)
        
        payload = payload[0]
        all_operation_data = payload['all_operation_data']
        all_op_list = payload['all_op_list']
        
        
        # Calculate total operations including existing ones
        total_ops = 0
        for op, param in self.operations.items():
            op_count = int(param[0] * num_ops) if isinstance(param, list) else int(param * num_ops)
            if isinstance(param, list):
                total_ops += (op_count * param[1])
            else:
                total_ops += op_count
        
        

        
        op_counters = {op: 0 for op in all_operation_data.keys()}
        
        tasks = []
        
        # stream = cp.cuda.Stream()
        
        for op in all_op_list:
            if isinstance(op, list):
                op_name, _ = op
                data = all_operation_data[op_name][op_counters[op_name]]
                
                if op_name == 'multiget':
                    tasks.append((self.bind_instance.multiget, 
                                  (data['keys'], ), 
                                  (op_name, op_counters[op_name])))
                elif op_name == 'multiset':
                    tasks.append((self.bind_instance.multiset, 
                                  (data['key_list'], data['value_list']),
                                  (op_name, op_counters[op_name])))
                else:
                    data = all_operation_data[op][op_counters[op]]
                    op_counters[op] += 1
                    if op == 'insert':
                        tasks.append((self.bind_instance.insert, (data['key'], data['values'])))
                    elif op == 'read':
                        tasks.append((self.bind_instance.read, (data['key'])))
                    elif op == 'update':
                        tasks.append((self.bind_instance.update, (data['key'], data['fieldkey'], 
                                               data['value'])))   
                    elif op == 'scan':
                        tasks.append((self.bind_instance.scan, (data['start_key'])))
                    else:
                        raise ValueError(f"Invalid operation: {op}")
                op_counters[op_name] += 1
            
        
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        binding_return_data = []
        
        self.bind_instance.barrier()
            
        print("Starting benchmark...")    
        # Start timing
        if self.data_integrity == DataIntegrity.CUSTOMIZED:
            start_event.record()
            for fn, args, validation in tasks:
                binding_return_data.append((validation, fn(*args)))
        elif self.data_integrity == DataIntegrity.YCSB:
            start_event.record()
            for fn, args, validation in tasks:
                binding_return_data.append((validation, fn(*args)))
        else:
            start_event.record()
            for fn, args, validation in tasks:
                binding_return_data.append((validation, fn(*args)))

        cp.cuda.Stream.null.synchronize()
        end_event.record()
        end_event.synchronize()
        
        total_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000
        
        integrity = True
        
        # check data integrity
        print(f"Checking data integrity...")
        if self.data_integrity == DataIntegrity.CUSTOMIZED:
            for i, (validation, data) in enumerate(binding_return_data):
                keys = all_operation_data[validation[0]][validation[1]]['keys']
                if not self.bind_instance.check_data_integrity(keys, data):
                    integrity = False
                if i % 100 == 0:
                    print(f"{i}/{num_ops} data integrity check passed")
            print(f"Data integrity check passed")
        elif self.data_integrity == DataIntegrity.YCSB:
            cpu_count = mp.get_context('fork').cpu_count()
            executor = ProcessPoolExecutor(mp_context=mp.get_context('fork'), max_workers=cpu_count)
            futures = []
            for i, data in enumerate(binding_return_data):
                test_result: pd.DataFrame = self.bind_instance.results_postprocess(data)
                
                each_worker_check_size = test_result.shape[0] // cpu_count
                remainder = test_result.shape[0] % cpu_count
                
                current_idx = 0
                for j in range(cpu_count):
                    start_idx = current_idx
                    current_idx += each_worker_check_size
                    if j < remainder:
                        current_idx += 1
                    futures.append(executor.submit(check_rows_integrity_ycsb, test_result.iloc[start_idx:current_idx], self.workload_gen))

                if i % 10 == 0:
                    print(f"{i}/{num_ops} data integrity check submitted")
            i = 0
            for future in as_completed_process(futures):
                if not future.result():
                    integrity = False
                i += 1
                if i % 100 == 0:
                    print(f"{i}/{len(futures)} data integrity check passed")
            executor.shutdown(wait=True)
            print(f"Data integrity check passed")
        else:
            print(f"Data integrity type: {self.data_integrity} do not need to check")

        return total_time, integrity, -1
        