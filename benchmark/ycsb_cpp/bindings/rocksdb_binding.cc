#include "binding_interface.cuh"
#include "binding_registry.cuh"
#include <iostream>
#include <nlohmann/json.hpp>
#include <rocksdb/db.h>
#include <cstring>  // For std::memcpy
#include <omp.h>    // For OpenMP





template<typename K, typename V>
class RocksDBBinding : public IBinding<K, V> {
public:

    RocksDBBinding():table_(nullptr), dim_(0) {
        std::cout << "RocksDBBinding constructor" << std::endl;
    };

    ~RocksDBBinding() {
        std::cout << "RocksDBBinding destructor" << std::endl;
        cleanup();
    }

    bool initialize(const InitConfig& config)  {
        try {

            // Store dimension
            dim_ = config.dim;
            nlohmann::json config_json = nlohmann::json::parse(config.additional_config);
            config_ = config_json;
            // Initialize RocksDB table options
            rocksdb::Options options_;
            options_.create_if_missing = config_["create_if_missing"];
            
            
            // Create and initialize HKV table
            std::cout << "Creating RocksDB table" << std::endl;
            
            rocksdb::Status status = rocksdb::DB::Open(options_, config_json["rocksdb_path"], &table_);
            if (!status.ok()) {
                std::cerr << "Failed to open RocksDB: " << status.ToString() << std::endl;
                return false;
            }

            std::cout << "RocksDB table created" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "RocksDB binding initialization failed: " << e.what() << std::endl;
            return false;
        }
    }

    void cleanup() {
        delete table_;
        std::cout << "RocksDB binding cleaned up" << std::endl;
    }

    void multiset(uint32_t batch_size,
                    const K* keys,
                    const V* values,
                    const CallContext& ctx) override {
        if (!table_ || batch_size == 0) {
            std::cerr << "RocksDB binding not initialized" << std::endl;
            return;
        }
        
        try {
            rocksdb::WriteBatch batch;
            
            // Pre-serialize values in parallel
            std::vector<std::string> value_strings(batch_size);
            #pragma omp parallel for
            for (size_t i = 0; i < batch_size; ++i) {
                value_strings[i] = serialize_value_array(&values[i * dim_]);
            }
            
            // Add to batch sequentially (WriteBatch is not thread-safe)
            for (size_t i = 0; i < batch_size; ++i) {
                rocksdb::Slice key_slice(reinterpret_cast<const char*>(&keys[i]), sizeof(K));
                rocksdb::Slice value_slice(value_strings[i].data(), value_strings[i].size());
                batch.Put(key_slice, value_slice);
            }
            

            rocksdb::WriteOptions write_opts;
            write_opts.disableWAL = config_["disableWAL"];

            rocksdb::Status status = table_->Write(write_opts, &batch);
            if (!status.ok()) {
                std::cerr << "Failed to write batch to RocksDB: " << status.ToString() << std::endl;
                return;
            }
            
            return;
        } catch (const std::exception& e) {
            std::cerr << "RocksDB multiset error: " << e.what() << std::endl;
            return;
        }
    }

    void multiget(uint32_t batch_size,
                    const K* h_keys,
                    V* h_values_out,
                    bool* h_found,
                    const CallContext& ctx) override {
        if (!table_) {
            std::cerr << "RocksDB binding not initialized" << std::endl;
            return;
        }

        // Build key slices
        std::vector<rocksdb::Slice> key_slices(batch_size);
        #pragma omp parallel for
        for (uint32_t i = 0; i < batch_size; ++i) {
            key_slices[i] = rocksdb::Slice(reinterpret_cast<const char*>(&h_keys[i]), sizeof(K));
        }
        
        std::vector<std::string> value_strings(batch_size);

        rocksdb::ReadOptions read_opts;

        // Perform batch get
        std::vector<rocksdb::Status> statuses = table_->MultiGet(
            read_opts,
            key_slices,
            &value_strings
        );
        

        // #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            if (statuses[i].ok()) {
                // Convert string value back to Value type
                V* temp_values = deserialize_value_array(value_strings[i]);
                // Print all values in the array
                memcpy(&h_values_out[i*dim_], temp_values, dim_ * sizeof(V));
                delete[] temp_values;  // Clean up temporary memory
                h_found[i] = true;
            } else {
                h_found[i] = false;
                // #pragma omp critical
                // {
                    std::cerr << "RocksDB multiget error: " << statuses[i].ToString() << std::endl;
                    throw std::runtime_error("RocksDB multiget error: " + statuses[i].ToString());
                // }
            }
        }
       
    }


private:
    nlohmann::json config_;
    rocksdb::DB* table_;
    uint32_t dim_;

    std::string serialize_value_array(const V* values) {
        std::string buffer;
        size_t size = dim_;
        buffer.append(reinterpret_cast<const char*>(&size), sizeof(size));
        buffer.append(reinterpret_cast<const char*>(values), sizeof(V) * size);
        return buffer;
    }

    V* deserialize_value_array(const std::string& raw) {
        size_t size;
        memcpy(&size, raw.data(), sizeof(size));
        V* result = new V[size];
        memcpy(result, raw.data() + sizeof(size), sizeof(V) * size);
        return result;
    }


    
}; 



// Register the RocksDB binding with the registry using simplified macro
using RocksDBBinding_u64d = RocksDBBinding<uint64_t, double>;
REGISTER_CPU_BINDING(uint64_t, double, RocksDBBinding_u64d, "rocksdb"); 