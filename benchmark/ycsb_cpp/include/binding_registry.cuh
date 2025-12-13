#pragma once

#include "binding_interface.cuh"
#include <string>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <functional>
#include <mutex>

// Convenience macros for CUDA and CPU bindings
#define REGISTER_CUDA_BINDING(K, V, BindingType, name) \
    static BindingRegistrar<K, V, BindingType> g_registrar_##BindingType(name, true);

#define REGISTER_CPU_BINDING(K, V, BindingType, name) \
    static BindingRegistrar<K, V, BindingType> g_registrar_##BindingType(name, false);


template<typename K, typename V>
struct BindingInfo {
    std::function<std::shared_ptr<IBinding<K, V>>()> factory;
    bool isCuda;
};

template<typename K, typename V>
class BindingRegistry {
public:
    
    // Get singleton instance
    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }
    
    // Register a binding factory function
    bool registerBinding(const std::string& name, 
                         bool isCUDA, 
                         std::function<std::shared_ptr<IBinding<K, V>>()> factory) {
        if (bindings_.find(name) != bindings_.end()) {
            std::cout << "Warning: Binding '" << name << "' already registered, ignoring duplicate registration" << std::endl;
            return false;
        }

        bindings_[name] = {factory, isCUDA};
        std::cout << "Registered binding: " << name << "( Key: " << typeid(K).name() << ", Value: " << typeid(V).name() << ")" << std::endl;
        return true;
    }


    // Get binding info (for checking CUDA capability etc.)
    BindingInfo<K, V>* getBindingInfo(const std::string& name) {
        auto it = bindings_.find(name);
        if (it == bindings_.end()) {
            return nullptr;
        }
        return &it->second;
    }
    
    // Check if binding is CUDA-compatible
    bool isCudaBinding(const std::string& name) const {
        auto info = getBindingInfo(name);
        return info ? info->isCuda : false;
    }
    
    // Get list of available bindings
    std::vector<std::string> getAvailableBindings(bool isCUDA) const {
        std::vector<std::string> names;
        names.reserve(bindings_.size());
        for (const auto& pair : bindings_) {
            if (pair.second.isCuda == isCUDA) {
                names.push_back(pair.first);
            }
        }
        return names;
    }
    
    // Check if a binding is registered
    bool isRegistered(const std::string& name) const {
        return bindings_.find(name) != bindings_.end();
    }
    
    // Get number of registered bindings
    size_t getBindingCount() const {
        return bindings_.size();
    }
    
private:
    BindingRegistry() = default;
    ~BindingRegistry() = default;
    
    // Disable copy and assignment
    BindingRegistry(const BindingRegistry&) = delete;
    BindingRegistry& operator=(const BindingRegistry&) = delete;
    
    std::unordered_map<std::string, BindingInfo<K, V>> bindings_;
};

// Helper class for automatic registration
template<typename K, typename V, typename BindingType>
class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, bool isCUDA) {
        BindingRegistry<K, V>::getInstance().registerBinding(name, isCUDA, 
            []() -> std::shared_ptr<IBinding<K, V>> {
                return std::make_shared<BindingType>();
            });
    }
};


