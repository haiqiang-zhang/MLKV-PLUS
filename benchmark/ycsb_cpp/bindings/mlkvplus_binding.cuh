#pragma once

#include "binding_interface.cuh"
#include "mlkv_plus.cuh"
#include <memory>

template<typename K, typename V>
class MLKVPlusBinding : public IBinding<K, V> {
private:
    std::unique_ptr<mlkv_plus::DB<K, V>> storage_;
    mlkv_plus::StorageConfig config_;
    uint32_t dim_;
    
public:
    MLKVPlusBinding();
    ~MLKVPlusBinding() override;
    
    bool initialize(const InitConfig& config) override;
    
    void cleanup() override;
    
    void multiset(uint32_t batch_size,
                 const K* d_keys,
                 const V* d_values,
                 const CallContext& ctx = {}) override;
    
    void multiget(uint32_t batch_size,
                 const K* h_keys,
                 V* d_values_out,
                 bool* h_found,
                 const CallContext& ctx = {}) override;


    void get(const K* h_keys,
             V* d_values_out,
             bool* h_found,
             const CallContext& ctx = {}) override;
    
}; 