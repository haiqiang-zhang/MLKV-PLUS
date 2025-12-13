import json
import time
import os
import glob
from typing import List, Dict, Any, Type
from benchmark import YCSBController
from benchmark.ycsb.ConfigLoader import get_workload_config, get_available_workloads
import multiprocessing as mp
import gc
import psutil

class BenchmarkRunner:
    def __init__(self):
        self.results_dir = get_workload_config()['results_folder']
        self.available_results = self.get_available_results()
        self.current_file = None
        self.status_file = "/tmp/mlkvplus_benchmark_status.json"
        self.current_status = None
        self.results = None
        self.active_processes = []  # Track active processes
        self.reset_status()
        
        
    def reset_status(self):
        self.current_status = None
        self._save_status(None)

    def _load_status(self):
        """Load benchmark status from file"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_status(self, status):
        """Save benchmark status to file"""
        if status is None:
            if os.path.exists(self.status_file):
                os.remove(self.status_file)
            return
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=4)
        self.current_status = status
        
        
        
    def get_available_results(self):
        """Get list of available result files"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            return []
            
        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        return [os.path.basename(f) for f in result_files]

    def load_results_from_file(self, file_path: str):
        """Load results from a specific file"""
        try:
            with open(file_path, 'r') as f:
                self.results = json.load(f)
            self.current_file = file_path
            return True
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return False
        
    def generate_new_results_file(self, filename: str):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        filepath = os.path.join(self.results_dir, filename)
        
        if os.path.exists(filepath):
            raise Exception(f"File {filepath} already exists")
        
        self.current_file = filepath
        
        with open(filepath, 'w') as f:
            json.dump([], f, indent=4)
        
        return True
            
    def check_current_file(self):
        if self.current_file:
            if os.path.exists(self.current_file):
                results = self.load_results_from_file(self.current_file)
                if results == True:
                    return True
        return False

    def _save_results(self, result):
        """Save results to a JSON file"""
        results = []
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        if self.current_file:
            filepath = self.current_file
            
            self.load_results_from_file(filepath)
            self.results = self.results + [result]
            
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            self.current_file = filepath
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
            
        print(f"Saving results to {filepath}")
            
        self.available_results = self.get_available_results()

    def _run_benchmark_process(self, num_records, workload_name, workload_config, gpu_device, repeat, repeat_count, result_pipe, binding_name, binding_type, comments, num_streams, operations_file, zipfian_theta, binding_config):
        """Run a single benchmark in a separate process and send results back through pipe"""
        # try:
        print(f"Process: Running benchmark with {num_records:,} records, {workload_name}, GPU {gpu_device}, Repeat {repeat + 1}/{repeat_count}")
        
        benchmark = YCSBController(
            num_records=num_records,
            operations=workload_config['operations'],
            distribution=workload_config['distribution'],
            scan_size=workload_config['scan_size'],
            orderedinserts=workload_config['orderedinserts'],
            data_integrity=workload_config['data_integrity'],
            target_qps=workload_config['target_qps'],
            min_field_length=workload_config['min_field_length'],
            max_field_length=workload_config['max_field_length'],
            field_count=workload_config['field_count'],
            gpu_device=gpu_device,
            output_file=None,
            load_data_output_file=None,
            binding_type=binding_type,
            binding_name=binding_name,
            operations_file=operations_file,
            zipfian_theta=workload_config['zipfian_theta'],
            binding_config=binding_config
        )
        
        result = benchmark.run(num_ops=workload_config['ops'], num_streams=num_streams, save_ops_details=False)
        result['num_records'] = num_records
        result['workload'] = workload_name
        result['gpu_device'] = gpu_device
        result['repeat'] = repeat + 1
        result['repeat_count'] = repeat_count
        result['binding'] = binding_name
        result['binding_type'] = binding_type
        result['comments'] = comments
        benchmark.cleanup()
        del benchmark
        gc.collect()
        
        # Send result back through pipe
        result_pipe.send(result)
        result_pipe.close()
        print(f"Process: Sent result back to main process")
            
        # except Exception as e:
        #     print(f"Process: Error in benchmark: {str(e)}")
        #     error_result = {
        #         'error': str(e)
        #     }
        #     result_pipe.send(error_result)
        #     result_pipe.close()

    def run_batch_benchmark(self, 
                          record_sizes: List[int],
                          workload_names: List[str],
                          binding_name: str,
                          binding_type: str,
                          distribution: str = 'zipfian',
                          zipfian_theta: float = 0.99,
                          scan_size: int = 100,
                          ops: int = 10_000,
                          target_qps: int = None,
                          gpu_device: int = 0,
                          repeat_count: int = 1,
                          comments: str = None,
                          num_streams: List[int] | None = None,
                          operation_file: str = None,
                          binding_config: dict = None):
        """Run batch benchmarks with different configurations"""
        
        try:
            
            if not self.check_current_file():
                raise Exception("file is not loaded")

            # Cleanup any existing processes before starting new ones
            self._cleanup_processes()
            
            if num_streams is None:
                num_streams = [None]

            total_combinations = len(record_sizes) * len(workload_names) * len(num_streams)
            current_combination = 0
            
            status = {
                'running': True,
                'total_combinations': total_combinations,
                'current_combination': 0,
                'record_sizes': record_sizes,
                'workload_names': workload_names,
                'distribution': distribution,
                'zipfian_theta': zipfian_theta,
                'binding': binding_name,
                'binding_type': binding_type,
                'scan_size': scan_size,
                'ops': ops,
                'target_qps': target_qps,
                'gpu_device': gpu_device,
                'repeat_count': repeat_count,
                'comments': comments,
                'num_streams': num_streams,
                'start_time': time.time(),
                'current_record_size': None,
                'current_workload': None,
                'current_num_stream': None,
                'current_repeat': 0,
                'active_processes': [],
                'operation_file': operation_file,
                'binding_config': binding_config
            }
            self._save_status(status)
            
            for num_records in record_sizes:
                for workload_name in workload_names:
                    for num_stream in num_streams:
                        current_combination += 1
                        print(f"Running {current_combination}/{total_combinations}: {num_records:,} records, {workload_name}")
                        
                        status['current_combination'] = current_combination
                        status['current_record_size'] = num_records
                        status['current_workload'] = workload_name
                        status['current_num_stream'] = num_stream
                        workload_config = get_workload_config(workload_name)
                        
                        # Run benchmark multiple times
                        for repeat in range(repeat_count):
                            status['current_repeat'] = repeat + 1
                            self._save_status(status)
                            
                            print(f"Repeat {repeat + 1}/{repeat_count}")
                            
                            # Create a pipe for communication
                            parent_conn, child_conn = mp.get_context('spawn').Pipe()
                            
                            # Create and start a process for each benchmark
                            process = mp.get_context('spawn').Process(
                                target=self._run_benchmark_process,
                                args=(
                                    num_records,
                                    workload_name,
                                    workload_config,
                                    gpu_device,
                                    repeat,
                                    repeat_count,
                                    child_conn,
                                    binding_name,
                                    binding_type,
                                    comments,
                                    num_stream,
                                    operation_file,
                                    zipfian_theta,
                                    binding_config
                                )
                            )
                            
                            process.start()
                            status['active_processes'].append(process.pid)
                            self._save_status(status)
                            
                            # Receive result from the process
                            result = parent_conn.recv()
                            
                            # Wait for process to complete
                            process.join()
                            
                            # Save result in main process
                            if not result.get('error'):
                                self._save_results(result)
                            else:
                                raise Exception(f"Error in benchmark: {result.get('error')}")
                            
                            
                            self._cleanup_processes()
                            time.sleep(2)  # Give time for system cleanup
            
            status['running'] = False
            status['end_time'] = time.time()
            status['duration'] = status['end_time'] - status['start_time']
            self._save_status(status)
            
        except Exception as e:
            status['running'] = False
            status['error'] = str(e)
            status['end_time'] = time.time()
            status['duration'] = status['end_time'] - status['start_time']
            self._save_status(status)
            raise e
        finally:
            # Reset benchmark running status and cleanup processes
            self.current_status['running'] = False
            self._cleanup_processes()

    def is_benchmark_running(self):
        if self.current_status:
            return self.current_status['running']
        return False

    def __del__(self):
        """Cleanup method to ensure all processes are terminated"""
        self._cleanup_processes()



    def stop_benchmark(self):
        """Stop benchmark"""
        self._cleanup_processes()
        self.current_status['running'] = False
        self.current_status['stopped'] = True
        self._save_status(self.current_status)

    def _cleanup_processes(self):
        """Terminate all active processes"""
        if self.current_status and self.current_status['active_processes'] and len(self.current_status['active_processes']) > 0:
            for process_pid in self.current_status['active_processes']:
                print(f"Terminating active process: {process_pid}")
                try:
                    parent = psutil.Process(process_pid)
                    for child in parent.children(recursive=True):
                        child.terminate()
                    parent.terminate()
                    self.current_status['active_processes'].remove(process_pid)
                    self._save_status(self.current_status)
                except Exception as e:
                    print(f"{process_pid}: {e}")
            print("Active processes cleaned up")
        else:
            print("No active processes to cleanup")