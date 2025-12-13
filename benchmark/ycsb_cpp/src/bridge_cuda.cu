#include "bridge.h"
#include "binding_registry.cuh"
#include "benchmark_util.cuh"
#include <functional>


using benchmark::Timer;


template<typename K, typename V>
YCSBBridgeCUDA<K, V>::YCSBBridgeCUDA(const std::string& binding_name) {
    BindingInfo<K, V>* binding_info = BindingRegistry<K, V>::getInstance().getBindingInfo(binding_name);


    if (!binding_info->isCuda){
        std::cerr << binding_name + " is not CUDA binding." << std::endl;
        throw std::runtime_error(binding_name + " is not CUDA binding.");
    }

    binding_ = binding_info->factory();
}

template<typename K, typename V>
YCSBBridgeCUDA<K, V>::~YCSBBridgeCUDA() {
    std::cout << "YCSBBridgeCUDA destructor" << std::endl;
    
}


template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::initialize(uint64_t gpu_init_capacity, uint64_t gpu_max_capacity, uint32_t dim, uint32_t hbm_gb, std::vector<int> gpu_ids, uint64_t max_batch_size, const std::string& binding_config) {



    InitConfig cfg;
    dim_ = dim;
    hbm_gb_ = hbm_gb;
    gpu_ids_ = gpu_ids;
    max_batch_size_ = max_batch_size;
    binding_config_ = binding_config;

    cfg.dim = dim_;
    cfg.max_batch_size = max_batch_size_;
    cfg.additional_config = binding_config_;
    cfg.use_cuda = true;
    cfg.init_capacity = gpu_init_capacity;
    cfg.max_capacity = gpu_max_capacity;
    cfg.hbm_gb = hbm_gb_;
    cfg.gpu_ids = gpu_ids_;


    bool init_success = binding_->initialize(cfg);

    if (!init_success) {
        std::cerr << "Failed to initialize binding" << std::endl;
        throw std::runtime_error("Failed to initialize binding");
    }
}

template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::multiset(uint32_t batch_size, const K* keys, const V* values, cudaStream_t stream) {
    binding_->multiset(batch_size, keys, values, {stream});
}

template<typename K, typename V>
BenchmarkResult YCSBBridgeCUDA<K, V>::run_benchmark(const std::vector<Operation<K, V>>& ops, uint64_t num_streams, const std::string& data_integrity) {

    std::vector<std::string> supported_data_integrity = {"YCSB", "NOT_CHECK"};
    if (std::find(supported_data_integrity.begin(), supported_data_integrity.end(), data_integrity) == supported_data_integrity.end()) {
        std::cerr << "Unsupported data integrity: " << data_integrity << std::endl;
        throw std::runtime_error("Unsupported data integrity");
    }



    if (gpu_ids_.size() == 1) {
        cudaSetDevice(gpu_ids_[0]);
    }
    
    
    // Create a vector of CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (uint32_t i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    

    std::vector<V*> values_out_list;
    std::vector<bool*> h_found_list;
    
    std::vector<std::function<void()>> workload_batch_fns;
    workload_batch_fns.reserve(ops.size());

    int read_counter = 0;

    for (uint32_t i = 0; i < ops.size(); ++i) {

        uint32_t stream_idx = i % num_streams;

        if (ops[i].op == "multiget") {

            // Fix cudaMalloc syntax - allocate device memory properly
            V* values_out;
            bool* h_found = new bool[ops[i].keys.size()];

     
            cudaError_t err = cudaMalloc(&values_out, ops[i].keys.size() * dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            values_out_list.push_back(values_out);
            h_found_list.push_back(h_found);

            workload_batch_fns.push_back([&, i, read_counter]() {
                binding_->multiget(ops[i].keys.size(), ops[i].keys.data(), values_out_list[read_counter], h_found_list[read_counter], {streams[stream_idx]});
            });
            read_counter++;
        } else if (ops[i].op == "multiset") {
            workload_batch_fns.push_back([&, i]() {
                binding_->multiset(ops[i].keys.size(), ops[i].keys.data(), ops[i].values.data(), {streams[stream_idx]});
            });
        } else if (ops[i].op == "read") {
            V* values_out;
            h_found_list.push_back(new bool[1]);


            cudaError_t err = cudaMalloc(&values_out, dim_ * sizeof(V));
            if (err != cudaSuccess) {
                std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error("cudaMalloc failed");
            }

            values_out_list.push_back(values_out);
            
            workload_batch_fns.push_back([&, i, read_counter]() {
                binding_->get(ops[i].keys.data(), values_out_list[read_counter], h_found_list[read_counter], {streams[stream_idx]});
            });
            read_counter++;
        }
        else {
            std::cerr << "Unsupported operation: " << ops[i].op << std::endl;
            throw std::runtime_error("Unsupported operation");
        }
    }

    std::cout << "start running workload..." << std::endl;

    // ===============================
    // run workload
    // ===============================
    Timer<double> timer;
    timer.start();
    for (const auto& fn : workload_batch_fns) {
        fn();
    }
    timer.end();
    double total_time = timer.getResult();
    std::cout << "workload done" << std::endl;


    // ===============================
    // verify integrity
    // ===============================
    std::pair<bool, double> result;
    if (data_integrity == "YCSB") {
        result = verify_integrity_ycsb(ops, values_out_list, h_found_list);
    } else if (data_integrity == "NOT_CHECK") {
        result = std::make_pair(true, 1.0);
    } else {
        std::cerr << "Unsupported data integrity: " << data_integrity << std::endl;
        throw std::runtime_error("Unsupported data integrity");
    }

    // clean the d_values_out_list and h_found_list
    for (auto& values_out : values_out_list) {
        cudaFree(values_out);
    }
    for (auto& h_found : h_found_list) {
        delete[] h_found;
    }

    return BenchmarkResult{total_time, result.first, result.second};
}

// GPU kernel to verify the correctness of retrieved values
template<typename K, typename V>
__global__ void verify_values_kernel(const K* keys, const V* values, bool* found, 
                                    uint32_t batch_size, uint32_t dim, bool* verification_results) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size && found[idx]) {
        // Generate expected value using the same logic as build_deterministic_value
        // For double type: (key + fieldkeyIndex) % 2^53
        // fieldkeyIndex is the dimension index (0 to dim-1)
        
        bool all_correct = true;
        for (uint32_t field_idx = 0; field_idx < dim; ++field_idx) {
            V expected_value;
            if constexpr (std::is_same_v<V, double>) {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            } else if constexpr (std::is_same_v<V, float>) {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            } else {
                expected_value = static_cast<V>((keys[idx] + field_idx) % (1ULL << 53));
            }
            
            V actual_value = values[idx * dim + field_idx];
            V diff = actual_value - expected_value;
            if constexpr (std::is_floating_point_v<V>) {
                if (fabs(diff) > 1e-9) {
                    all_correct = false;
                    break;
                }
            } else {
                if (diff != 0) {
                    all_correct = false;
                    break;
                }
            }
        }
        verification_results[idx] = all_correct;
    } else {
        verification_results[idx] = true; // Not found keys are considered correct for this check
    }
}

template<typename K, typename V>
std::pair<bool, double> YCSBBridgeCUDA<K, V>::verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& h_found_list) {


    if (gpu_ids_.size() == 1) {
        cudaSetDevice(gpu_ids_[0]);
    } else {
        std::cerr << "Data integrity check is not supported for multiple GPUs" << std::endl;
        throw std::runtime_error("Data integrity check is not supported for multiple GPUs");
    }
    
    bool overall_result = true;
    size_t read_op_idx = 0;
    double overall_accurate_count = 0;
    double overall_found_count = 0;
    double overall_op_count = 0;

    std::cout << "Starting integrity verification..." << std::endl;
    
    for (size_t op_idx = 0; op_idx < stored_ops.size(); ++op_idx) {
        if (stored_ops[op_idx].op == "multiget" || stored_ops[op_idx].op == "read") {
            V* values_out = d_values_out_list[read_op_idx];
            bool* h_found = h_found_list[read_op_idx];
            uint32_t batch_size = stored_ops[op_idx].keys.size();
            overall_op_count += batch_size;
            
            // std::cout << "Verifying multiget operation " << multiget_op_idx 
            //           << " with batch size " << batch_size << std::endl;
            
            K* d_keys;
            bool* d_verification_results;
            bool* d_found;
            
            cudaMalloc(&d_keys, batch_size * sizeof(K));
            cudaMalloc(&d_verification_results, batch_size * sizeof(bool));
            cudaMalloc(&d_found, batch_size * sizeof(bool));
            
            // Copy data to device
            cudaMemcpy(d_keys, stored_ops[op_idx].keys.data(), 
                        batch_size * sizeof(K), cudaMemcpyHostToDevice);
            cudaMemcpy(d_found, h_found, batch_size * sizeof(bool), cudaMemcpyHostToDevice);
            
            // Launch verification kernel
            int blockSize = 256;
            int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            verify_values_kernel<<<numBlocks, blockSize>>>(
                d_keys, values_out, d_found, batch_size, dim_, d_verification_results);
            
            cudaDeviceSynchronize();
            
            // Copy verification results back to host
            bool* h_verification_results = new bool[batch_size];
            cudaMemcpy(h_verification_results, d_verification_results, 
                        batch_size * sizeof(bool), cudaMemcpyDeviceToHost);
            
            // Check results
            int found_count = 0;
            int correct_count = 0;
            for (uint32_t i = 0; i < batch_size; ++i) {
                if (h_found[i]) {
                    found_count++;
                    if (h_verification_results[i]) {
                        correct_count++;
                    } else {
                        std::cout << "Key " << stored_ops[op_idx].keys[i] 
                                    << " has incorrect value!" << std::endl;
                        overall_result = false;
                    }
                } else {
                    // std::cout << "Key " << stored_ops[op_idx].keys[i] 
                    //           << " not found!" << std::endl;
                    overall_result = false;
                }
            }
            overall_accurate_count += correct_count;
            overall_found_count += found_count;
            
            // Cleanup device memory
            cudaFree(d_keys);
            cudaFree(d_verification_results);
            cudaFree(d_found);
            delete[] h_verification_results;

            read_op_idx++;

        }
    }

    std::cout << "Found rate: " << overall_found_count / overall_op_count << std::endl;
    std::cout << "Accurate rate in found keys: " << overall_accurate_count / overall_found_count << std::endl;
    std::cout << "Overall accurate rate: " << overall_accurate_count / overall_op_count << std::endl;
    return std::make_pair(overall_result, overall_accurate_count / overall_op_count);
}

template<typename K, typename V>
void YCSBBridgeCUDA<K, V>::cleanup() {
    binding_->cleanup();
}


template<typename K, typename V>
std::vector<std::string> YCSBBridgeCUDA<K, V>::getAvailableBindings() {
    return BindingRegistry<K, V>::getInstance().getAvailableBindings(true);
}




// Explicit template instantiations for supported types
template class YCSBBridgeCUDA<uint64_t, double>;