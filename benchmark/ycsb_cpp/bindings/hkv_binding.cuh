#pragma once

#include "binding_interface.cuh"
#include "merlin_hashtable.cuh"
#include <memory>

template<typename K, typename V>
class HKVBinding : public IBinding<K, V> {
private:
    using HKVTable = nv::merlin::HashTable<K, V, uint64_t, nv::merlin::EvictStrategy::kLru>;
    using TableOptions = nv::merlin::HashTableOptions;
    
    std::unique_ptr<HKVTable> table_;
    uint32_t dim_;
    
    // Device memory buffers for operations
    K* d_keys_;
    V* d_values_;
    V* d_values_out_;
    bool* d_found_;
    uint32_t max_batch_size_;
    
public:
    HKVBinding();
    ~HKVBinding() override;
    
    bool initialize(const InitConfig& config) override;
    
    void cleanup() override;
    
    void multiset(uint32_t batch_size,
                 const K* h_keys,
                 const V* h_values,
                 const CallContext& ctx = {}) override;
    
    void multiget(uint32_t batch_size,
                 const K* h_keys,
                 V* h_values_out,
                 bool* h_found,
                 const CallContext& ctx = {}) override;
    
private:
    void allocate_device_buffers(uint32_t max_batch_size);
    void free_device_buffers();
}; 