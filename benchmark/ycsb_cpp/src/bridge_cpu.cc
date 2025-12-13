#include "bridge.h"
#include "binding_registry.cuh"
#include "benchmark_util.cuh"

#include <functional>


using benchmark::Timer;


template<typename K, typename V>
YCSBBridgeCPU<K, V>::YCSBBridgeCPU(const std::string& binding_name) {
    BindingInfo<K, V>* binding_info = BindingRegistry<K, V>::getInstance().getBindingInfo(binding_name);


    if (binding_info->isCuda){
        std::cerr << binding_name + " is not CPU binding." << std::endl;
        throw std::runtime_error(binding_name + " is not CPU binding.");
    }

    binding_ = binding_info->factory();
}

template<typename K, typename V>
YCSBBridgeCPU<K, V>::~YCSBBridgeCPU() {
    std::cout << "YCSBBridgeCPU destructor" << std::endl;
    
}


template<typename K, typename V>
void YCSBBridgeCPU<K, V>::initialize(uint32_t dim, uint64_t max_batch_size, const std::string& binding_config) {
    

    dim_ = dim;
    max_batch_size_ = max_batch_size;
    binding_config_ = binding_config;

    InitConfig cfg;
    cfg.dim = dim;
    cfg.max_batch_size = max_batch_size;
    cfg.additional_config = binding_config;


    bool init_success = binding_->initialize(cfg);

    if (!init_success) {
        std::cerr << "Failed to initialize binding" << std::endl;
        throw std::runtime_error("Failed to initialize binding");
    }
}

template<typename K, typename V>
void YCSBBridgeCPU<K, V>::multiset(uint32_t batch_size, const K* keys, const V* values) {
    binding_->multiset(batch_size, keys, values);
}

template<typename K, typename V>
BenchmarkResult YCSBBridgeCPU<K, V>::run_benchmark(const std::vector<Operation<K, V>>& ops, const std::string& data_integrity) {

    std::vector<std::string> supported_data_integrity = {"YCSB", "NOT_CHECK"};
    if (std::find(supported_data_integrity.begin(), supported_data_integrity.end(), data_integrity) == supported_data_integrity.end()) {
        std::cerr << "Unsupported data integrity: " << data_integrity << std::endl;
        throw std::runtime_error("Unsupported data integrity");
    }

    

    std::vector<V*> values_out_list;
    std::vector<bool*> h_found_list;
    
    std::vector<std::function<void()>> workload_batch_fns;
    workload_batch_fns.reserve(ops.size());

    int read_counter = 0;

    for (uint32_t i = 0; i < ops.size(); ++i) {

        if (ops[i].op == "multiget") {
            V* values_out;
            bool* h_found = new bool[ops[i].keys.size()];

            values_out = (V *) malloc(ops[i].keys.size() * dim_ * sizeof(V));


            values_out_list.push_back(values_out);
            h_found_list.push_back(h_found);

            workload_batch_fns.push_back([&, i, read_counter]() {
                binding_->multiget(ops[i].keys.size(), ops[i].keys.data(), values_out_list[read_counter], h_found_list[read_counter]);
            });
            read_counter++;
        } else if (ops[i].op == "multiset") {
            workload_batch_fns.push_back([&, i]() {
                binding_->multiset(ops[i].keys.size(), ops[i].keys.data(), ops[i].values.data());
            });
        } else {
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
        free(values_out);
    }
    for (auto& h_found : h_found_list) {
        delete[] h_found;
    }

    return BenchmarkResult{total_time, result.first, result.second};
}

template<typename K, typename V>
std::pair<bool, double> YCSBBridgeCPU<K, V>::verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& h_found_list) {
    
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
            
            // CPU path - verify values on host
            int found_count = 0;
            int correct_count = 0;
            
            for (uint32_t i = 0; i < batch_size; ++i) {
                if (h_found[i]) {
                    found_count++;
                    
                    // Verify each dimension of the value
                    bool all_correct = true;
                    for (uint32_t field_idx = 0; field_idx < dim_; ++field_idx) {
                        V expected_value;
                        if constexpr (std::is_same_v<V, double>) {
                            expected_value = static_cast<V>((stored_ops[op_idx].keys[i] + field_idx) % (1ULL << 53));
                        } else if constexpr (std::is_same_v<V, float>) {
                            expected_value = static_cast<V>((stored_ops[op_idx].keys[i] + field_idx) % (1ULL << 53));
                        } else {
                            expected_value = static_cast<V>((stored_ops[op_idx].keys[i] + field_idx) % (1ULL << 53));
                        }
                        
                        V actual_value = values_out[i * dim_ + field_idx];
                        V diff = actual_value - expected_value;
                        
                        if constexpr (std::is_floating_point_v<V>) {
                            if (std::abs(diff) > 1e-9) {
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
                    
                    if (all_correct) {
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
            
            read_op_idx++;
        }
    }

    std::cout << "Found rate: " << overall_found_count / overall_op_count << std::endl;
    std::cout << "Accurate rate in found keys: " << overall_accurate_count / overall_found_count << std::endl;
    std::cout << "Overall accurate rate: " << overall_accurate_count / overall_op_count << std::endl;
    return std::make_pair(overall_result, overall_accurate_count / overall_op_count);
}

template<typename K, typename V>
void YCSBBridgeCPU<K, V>::cleanup() {
    if (binding_) {
        binding_->cleanup();
    }
}


template<typename K, typename V>
std::vector<std::string> YCSBBridgeCPU<K, V>::getAvailableBindings() {
    return BindingRegistry<K, V>::getInstance().getAvailableBindings(false);
}




// Explicit template instantiations for supported types
template class YCSBBridgeCPU<uint64_t, double>;