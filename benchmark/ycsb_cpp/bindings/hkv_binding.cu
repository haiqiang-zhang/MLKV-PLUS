#include "hkv_binding.cuh"
#include "binding_registry.cuh"
#include <iostream>
#include <stdexcept>
#include <string>

#define HKV_CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    }                                                                   \
  } while (0)

template<typename K, typename V>
HKVBinding<K, V>::HKVBinding() 
    : table_(nullptr), dim_(0), d_keys_(nullptr), d_values_(nullptr), 
      d_values_out_(nullptr), d_found_(nullptr), max_batch_size_(0) {
}

template<typename K, typename V>
HKVBinding<K, V>::~HKVBinding() {
}

template<typename K, typename V>
bool HKVBinding<K, V>::initialize(const InitConfig& config) {
    try {
        // check if gpu_ids is only one
        if (config.gpu_ids.size() != 1) {
            throw std::runtime_error("HKV binding only supports one GPU");
        }

        int gpu_id = config.gpu_ids[0];

        // set device
        HKV_CUDA_CHECK(cudaSetDevice(gpu_id));
        

        // Store dimension
        dim_ = config.dim;
        
        // Initialize HKV table options
        TableOptions options;
        options.init_capacity = config.init_capacity;
        options.max_capacity = config.max_capacity;
        options.dim = config.dim;
        options.max_hbm_for_vectors = nv::merlin::GB(config.hbm_gb);
        options.io_by_cpu = false;
        options.device_id = gpu_id;

        // print options
        std::cout << "HKV options: " << std::endl;
        std::cout << "  init_capacity: " << options.init_capacity << std::endl;
        std::cout << "  max_capacity: " << options.max_capacity << std::endl;
        std::cout << "  dim: " << options.dim << std::endl;
        std::cout << "  hbm_gb: " << config.hbm_gb << std::endl;
        std::cout << "  io_by_cpu: " << options.io_by_cpu << std::endl;
        std::cout << "  device_id: " << options.device_id << std::endl;
        
        // Create and initialize HKV table
        std::cout << "Creating HKV table" << std::endl;
        table_ = std::make_unique<HKVTable>();
        table_->init(options);
        std::cout << "HKV table created" << std::endl;
        
        // Allocate device buffers for operations
        // Use a reasonable default max batch size
        max_batch_size_ = config.max_batch_size;
        allocate_device_buffers(max_batch_size_);
        
        std::cout << "HKV binding initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "HKV binding initialization failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename K, typename V>
void HKVBinding<K, V>::cleanup() {
    free_device_buffers();
    table_.reset();
    std::cout << "HKV binding cleaned up" << std::endl;
}

template<typename K, typename V>
void HKVBinding<K, V>::multiset(uint32_t batch_size,
                               const K* h_keys,
                               const V* h_values,
                               const CallContext& ctx) {
    
    try {



        
        // Copy keys and values to device
        cudaMemcpyAsync(d_keys_, h_keys, batch_size * sizeof(K), 
                                    cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(d_values_, h_values, batch_size * dim_ * sizeof(V), 
                                    cudaMemcpyHostToDevice, ctx.stream);


        
        
        // Perform insert operation
        table_->insert_or_assign(batch_size, d_keys_, d_values_, nullptr, ctx.stream, false);
        
        cudaStreamSynchronize(ctx.stream);

    } catch (const std::exception& e) {
        throw std::runtime_error("HKV multiset failed: " + std::string(e.what()));
    }
}

template<typename K, typename V>
void HKVBinding<K, V>::multiget(uint32_t batch_size,
                               const K* h_keys,
                               V* h_values_out,
                               bool* h_found,
                               const CallContext& ctx) {

    // Copy keys to device
    cudaMemcpyAsync(d_keys_, h_keys, batch_size * sizeof(K), 
                                cudaMemcpyHostToDevice, ctx.stream);
    
    // Perform find operation
    table_->find(batch_size, d_keys_, d_values_out_, d_found_, nullptr, ctx.stream);
    
    // Copy results back to host
    // HKV_CUDA_CHECK(cudaMemcpyAsync(h_values_out, d_values_out_, 
    //                           batch_size * dim_ * sizeof(V), 
    //                           cudaMemcpyDeviceToHost, ctx.stream));
    // HKV_CUDA_CHECK(cudaMemcpyAsync(h_found, d_found_, batch_size * sizeof(bool), 
    //                           cudaMemcpyDeviceToHost, ctx.stream));
    

    cudaStreamSynchronize(ctx.stream);

}


template<typename K, typename V>
void HKVBinding<K, V>::allocate_device_buffers(uint32_t max_batch_size) {
    try {
        HKV_CUDA_CHECK(cudaMalloc(&d_keys_, max_batch_size * sizeof(K)));
        HKV_CUDA_CHECK(cudaMalloc(&d_values_, max_batch_size * dim_ * sizeof(V)));
        HKV_CUDA_CHECK(cudaMalloc(&d_values_out_, max_batch_size * dim_ * sizeof(V)));
        HKV_CUDA_CHECK(cudaMalloc(&d_found_, max_batch_size * sizeof(bool)));
    } catch (const std::exception& e) {
        free_device_buffers();
        throw;
    }
}

template<typename K, typename V>
void HKVBinding<K, V>::free_device_buffers() {
    if (d_keys_) { cudaFree(d_keys_); d_keys_ = nullptr; }
    if (d_values_) { cudaFree(d_values_); d_values_ = nullptr; }
    if (d_values_out_) { cudaFree(d_values_out_); d_values_out_ = nullptr; }
    if (d_found_) { cudaFree(d_found_); d_found_ = nullptr; }
}

// Register the HKV bindings with the registry using simplified macros
using HKVBinding_u64d = HKVBinding<uint64_t, double>;
REGISTER_CUDA_BINDING(uint64_t, double, HKVBinding_u64d, "hkv");

using HKVBinding_u64f = HKVBinding<uint64_t, float>;
REGISTER_CUDA_BINDING(uint64_t, float, HKVBinding_u64f, "hkv");

using HKVBinding_u64u64 = HKVBinding<uint64_t, uint64_t>;
REGISTER_CUDA_BINDING(uint64_t, uint64_t, HKVBinding_u64u64, "hkv"); 