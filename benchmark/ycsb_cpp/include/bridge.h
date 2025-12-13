#include <iostream>
#include "binding_interface.cuh"


template<typename K, typename V>
struct Operation {
    std::string op;
    std::vector<K> keys;
    std::vector<V> values;
};

struct BenchmarkResult {
    double time_seconds;
    bool integrity;
    double integrity_accuracy;
};

template<typename K, typename V>
class YCSBBridgeCUDA {
public:
    YCSBBridgeCUDA(const std::string& binding_name);
    ~YCSBBridgeCUDA();

    void initialize(uint64_t gpu_init_capacity, uint64_t gpu_max_capacity, uint32_t dim, uint32_t hbm_gb, std::vector<int> gpu_ids, uint64_t max_batch_size, const std::string& binding_config);

    void multiset(uint32_t batch_size, const K* keys, const V* values, cudaStream_t stream = 0);

    BenchmarkResult run_benchmark(const std::vector<Operation<K, V>>& ops, uint64_t num_streams=1, const std::string& data_integrity="YCSB");

    void cleanup();

    // Static method to get available bindings for this K,V type combination
    static std::vector<std::string> getAvailableBindings();

private:
    std::shared_ptr<IBinding<K, V>> binding_;
    uint32_t dim_;
    uint32_t hbm_gb_;
    std::vector<int> gpu_ids_;
    uint64_t max_batch_size_;
    std::string binding_config_;
    
    std::pair<bool, double> verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& h_found_list);
};


template<typename K, typename V>
class YCSBBridgeCPU {
public:
    YCSBBridgeCPU(const std::string& binding_name);
    ~YCSBBridgeCPU();

    void initialize(uint32_t dim, uint64_t max_batch_size, const std::string& binding_config);

    void multiset(uint32_t batch_size, const K* keys, const V* values);

    BenchmarkResult run_benchmark(const std::vector<Operation<K, V>>& ops, const std::string& data_integrity="YCSB");

    void cleanup();

    // Static method to get available bindings for this K,V type combination
    static std::vector<std::string> getAvailableBindings();

private:
    std::shared_ptr<IBinding<K, V>> binding_;
    uint32_t dim_;
    uint64_t max_batch_size_;
    std::string binding_config_;
    
    std::pair<bool, double> verify_integrity_ycsb(const std::vector<Operation<K, V>>& stored_ops, const std::vector<V*>& d_values_out_list, const std::vector<bool*>& h_found_list);
};