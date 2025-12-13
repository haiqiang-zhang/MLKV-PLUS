#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>



struct InitConfig {
    // Generic
    uint32_t dim = 0;
    uint64_t max_batch_size = 0;
    std::string additional_config = "";

    // CUDA
    bool use_cuda = false;
    uint64_t init_capacity = 0, max_capacity = 0;
    uint32_t hbm_gb = 0;
    std::vector<int> gpu_ids;
};

struct CallContext {
    // CUDA
    cudaStream_t stream = 0;
};


template<typename K, typename V>
class IBinding {
public:
    virtual ~IBinding() = default;

    // Initialize the binding with given parameters
    virtual bool initialize(const InitConfig& config) = 0;
    
    // Clean up resources
    virtual void cleanup() = 0;
    
    // Batch insert operation
    virtual void multiset(uint32_t batch_size,
                         const K* h_keys,
                         const V* h_values,
                         const CallContext& ctx = {}) = 0;
    
    // Batch get operation
    virtual void multiget(uint32_t batch_size,
                         const K* h_keys,
                         V* d_values_out,
                         bool* h_found,
                         const CallContext& ctx = {}) = 0;


    virtual void get(const K* h_keys,
                     V* d_values_out,
                     bool* h_found,
                     const CallContext& ctx = {}) {
        throw std::runtime_error("get is not implemented");
    }
};
