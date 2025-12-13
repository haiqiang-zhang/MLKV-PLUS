#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bridge.h"
#include "binding_registry.cuh"
#include <variant>
#include <map>
#include <string>

namespace py = pybind11;


PYBIND11_MODULE(ycsb_binding, m) {
    m.doc() = "YCSB Benchmark Python Bindings";

    py::class_<Operation<uint64_t, double>>(m, "Operation_uint64_double")
        .def(py::init<const std::string&, const std::vector<uint64_t>&, const std::vector<double>&>())
        .def_readwrite("op", &Operation<uint64_t, double>::op)
        .def_readwrite("keys", &Operation<uint64_t, double>::keys)
        .def_readwrite("values", &Operation<uint64_t, double>::values);
    // Bind concrete YCSBBridgeCUDA instantiations
    py::class_<YCSBBridgeCUDA<uint64_t, double>>(m, "YCSBBridgeCUDA_uint64_double")
        .def(py::init<const std::string&>(), py::arg("binding_name"))
        .def("initialize", &YCSBBridgeCUDA<uint64_t, double>::initialize, 
             "Initialize the bridge with specified parameters",
             py::arg("gpu_init_capacity"), py::arg("gpu_max_capacity"), py::arg("dim"), py::arg("hbm_gb"), 
             py::arg("gpu_id"), py::arg("max_batch_size"), py::arg("binding_config"))
        .def("multiset", [](YCSBBridgeCUDA<uint64_t, double>& self, 
                           uint32_t batch_size, 
                           py::array_t<uint64_t> keys, 
                           py::array_t<double> values,
                           py::object stream) {
            // Check array dimensions
            if (keys.size() != batch_size) {
                throw std::runtime_error("Keys array size (" + std::to_string(keys.size()) + 
                                       ") does not match batch_size (" + std::to_string(batch_size) + ")");
            }
            
            // Get pointers to the data
            const uint64_t* keys_ptr = static_cast<const uint64_t*>(keys.data());
            const double* values_ptr = static_cast<const double*>(values.data());

            cudaStream_t stream_ptr = 0;  // Default stream
            if (!stream.is_none()) {
                stream_ptr = reinterpret_cast<cudaStream_t>(stream.cast<std::uintptr_t>());
            }
            
            // Call the C++ method
            self.multiset(batch_size, keys_ptr, values_ptr, stream_ptr);
        }, "Perform a multiset operation",
           py::arg("batch_size"), py::arg("keys"), py::arg("values"), py::arg("stream") = py::none())
        .def("run_benchmark", [](YCSBBridgeCUDA<uint64_t, double>& self, 
                                 std::vector<Operation<uint64_t, double>>& ops, 
                                 uint64_t num_streams,
                                 std::string data_integrity) {
            return self.run_benchmark(ops, num_streams, data_integrity);
        },
             "Run the benchmark",
             py::arg("ops"), py::arg("num_streams") = 1, py::arg("data_integrity") = "YCSB")
        .def("cleanup", &YCSBBridgeCUDA<uint64_t, double>::cleanup,
             "Cleanup the bridge")
        .def_static("get_available_bindings", &YCSBBridgeCUDA<uint64_t, double>::getAvailableBindings,
                   "Get list of available bindings for uint64_t keys and double values");

    // Bind YCSBBridgeCPU instantiations
    py::class_<YCSBBridgeCPU<uint64_t, double>>(m, "YCSBBridgeCPU_uint64_double")
        .def(py::init<const std::string&>(), py::arg("binding_name"))
        .def("initialize", &YCSBBridgeCPU<uint64_t, double>::initialize,
             "Initialize the CPU bridge with specified parameters",
             py::arg("dim"), py::arg("max_batch_size"), py::arg("binding_config"))
        .def("multiset", [](YCSBBridgeCPU<uint64_t, double>& self,
                           uint32_t batch_size,
                           py::array_t<uint64_t> keys,
                           py::array_t<double> values) {
            // Check array dimensions
            if (keys.size() != batch_size) {
                throw std::runtime_error("Keys array size (" + std::to_string(keys.size()) + 
                                       ") does not match batch_size (" + std::to_string(batch_size) + ")");
            }
            
            // Get pointers to the data
            const uint64_t* keys_ptr = static_cast<const uint64_t*>(keys.data());
            const double* values_ptr = static_cast<const double*>(values.data());
            
            // Call the C++ method
            self.multiset(batch_size, keys_ptr, values_ptr);
        }, "Perform a multiset operation on CPU",
           py::arg("batch_size"), py::arg("keys"), py::arg("values"))
        .def("run_benchmark", [](YCSBBridgeCPU<uint64_t, double>& self,
                                 std::vector<Operation<uint64_t, double>>& ops,
                                 std::string data_integrity) {
            return self.run_benchmark(ops, data_integrity);
        },
             "Run the benchmark on CPU",
             py::arg("ops"), py::arg("data_integrity") = "YCSB")
        .def("cleanup", &YCSBBridgeCPU<uint64_t, double>::cleanup,
             "Cleanup the CPU bridge")
        .def_static("get_available_bindings", &YCSBBridgeCPU<uint64_t, double>::getAvailableBindings,
                   "Get list of available CPU bindings for uint64_t keys and double values");

    

    // Expose BenchmarkResult struct
    py::class_<BenchmarkResult>(m, "BenchmarkResult")
        .def(py::init<>())
        .def_readwrite("time_seconds", &BenchmarkResult::time_seconds)
        .def_readwrite("integrity", &BenchmarkResult::integrity)
        .def_readwrite("integrity_accuracy", &BenchmarkResult::integrity_accuracy);
} 