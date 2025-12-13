#include "mlkvplus_binding.cuh"
#include "binding_registry.cuh"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <string>
#include <nlohmann/json.hpp>

template<typename K, typename V>
MLKVPlusBinding<K, V>::MLKVPlusBinding() 
    : storage_(nullptr), dim_(0), config_{} {
        std::cout << "MLKVPlusBinding constructor" << std::endl;
}

template<typename K, typename V>
MLKVPlusBinding<K, V>::~MLKVPlusBinding() {
}

template<typename K, typename V>
bool MLKVPlusBinding<K, V>::initialize(const InitConfig& config) {
    try {

        // check if gpu_ids is only one
        if (config.gpu_ids.size() != 1) {
            throw std::runtime_error("MLKV+ binding only supports one GPU");
        }

        int gpu_id = config.gpu_ids[0];

        // Store dimension
        dim_ = config.dim;
        
        // Configure MLKV Plus storage
        config_.hkv_init_capacity = config.init_capacity;
        config_.hkv_max_capacity = config.max_capacity;
        config_.dim = config.dim;
        config_.max_hbm_for_vectors_gb = config.hbm_gb;
        config_.hkv_io_by_cpu = false;
        config_.gpu_id = gpu_id;
        config_.create_if_missing = true;
        config_.max_batch_size = config.max_batch_size;
        nlohmann::json config_json = nlohmann::json::parse(config.additional_config);
        // Set RocksDB path based on additional config or use default
        if (config_json.contains("rocksdb_path") && !config_json["rocksdb_path"].is_null()) {
            config_.rocksdb_path = config_json["rocksdb_path"].get<std::string>();
        } else {
            config_.rocksdb_path = "/tmp/mlkvplus_ycsb_db_" + std::to_string(gpu_id);
        }
        config_.enable_gds_log = config_json["enable_gds_log"].get<bool>();
        config_.disableWAL = config_json["disableWAL"].get<bool>();
        config_.enable_gds_get_from_sst = config_json["enable_gds_get_from_sst"].get<bool>();
        config_.force_skip_memtable = config_json["force_skip_memtable"].get<bool>();
        config_.rocksdb_use_direct_reads = config_json["rocksdb_use_direct_reads"].get<bool>();
        
        // Create and initialize storage
        storage_ = std::make_unique<mlkv_plus::DB<K, V>>(config_);
        
        auto result = storage_->initialize();
        if (result != mlkv_plus::OperationResult::SUCCESS) {
            std::cerr << "Failed to initialize MLKV Plus storage" << std::endl;
            return false;
        }
        
        std::cout << "MLKV Plus binding initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MLKV Plus binding initialization failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::cleanup() {
    if (storage_) {
        storage_->cleanup();
        storage_.reset();
    }
    std::cout << "MLKV Plus binding cleaned up" << std::endl;
}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::multiset(uint32_t batch_size,
                                    const K* h_keys,
                                    const V* h_values,
                                    const CallContext& ctx) {
    if (!storage_) {
        throw std::runtime_error("MLKV Plus binding not initialized");
    }
    
    try {
        // Convert input arrays to MLKV Plus format
        K* mlkv_keys;
        V* mlkv_values;

        cudaMalloc(&mlkv_keys, batch_size * sizeof(K));
        cudaMalloc(&mlkv_values, batch_size * dim_ * sizeof(V));
        
        // Copy keys and values
        cudaMemcpy(mlkv_keys, h_keys, batch_size * sizeof(K), cudaMemcpyHostToDevice);
        cudaMemcpy(mlkv_values, h_values, batch_size * dim_ * sizeof(V), cudaMemcpyHostToDevice);
        
        // Perform multiset operation
        auto result = storage_->multiset(mlkv_keys, mlkv_values, batch_size);
        
        // Clean up allocated memory
        cudaFree(mlkv_keys);
        cudaFree(mlkv_values);
        
        if (result != mlkv_plus::OperationResult::SUCCESS) {
            throw std::runtime_error("MLKV Plus multiset operation failed");
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("MLKV Plus multiset failed: " + std::string(e.what()));
    }
}

template<typename K, typename V>
void MLKVPlusBinding<K, V>::multiget(uint32_t batch_size,
                                    const K* h_keys,
                                    V* d_values_out,
                                    bool* h_found,
                                    const CallContext& ctx) {
    
    // Perform multiget operation

    K* d_keys;
    cudaMalloc(&d_keys, batch_size * sizeof(K));
    cudaMemcpy(d_keys, h_keys, batch_size * sizeof(K), cudaMemcpyHostToDevice);


    bool* d_found;
    cudaMalloc(&d_found, batch_size * sizeof(bool));
    cudaMemset(d_found, 0, batch_size * sizeof(bool));

    auto result = storage_->multiget(d_keys, d_values_out, d_found, batch_size);
    
    // if (result != mlkv_plus::OperationResult::SUCCESS) {
    //     // Clean up allocated memory
    //     delete[] mlkv_values;
    //     delete[] mlkv_keys;
    //     delete[] mlkv_found;
    //     throw std::runtime_error("MLKV Plus multiget operation failed");
    // }
    
    // // Copy results back to output arrays
    // memcpy(h_found, mlkv_found, batch_size * sizeof(bool));
    // memcpy(h_values_out, mlkv_values, batch_size * dim_ * sizeof(V));

    cudaMemcpy(h_found, d_found, batch_size * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_keys);
    cudaFree(d_found);

}


template<typename K, typename V>
void MLKVPlusBinding<K, V>::get(const K* h_keys,
                               V* d_values_out,
                               bool* h_found,
                               const CallContext& ctx) {

    K* d_keys;
    MLKV_CUDA_CHECK(cudaMalloc(&d_keys, sizeof(K)));
    MLKV_CUDA_CHECK(cudaMemcpy(d_keys, h_keys, sizeof(K), cudaMemcpyHostToDevice));

    auto result = storage_->get(d_keys, d_values_out);


    if (result == mlkv_plus::OperationResult::KEY_NOT_FOUND) {
        memset(h_found, false,  sizeof(bool));
    } else {
        memset(h_found, true,  sizeof(bool));
    }
    
    cudaFree(d_keys);

}


// Register the MLKV Plus binding with the registry using simplified macro
using MLKVPlusBinding_u64d = MLKVPlusBinding<uint64_t, double>;
REGISTER_CUDA_BINDING(uint64_t, double, MLKVPlusBinding_u64d, "mlkv_plus");