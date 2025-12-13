include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(OpenMP REQUIRED)


# ------------------
# hkv binding
# ------------------
add_library(hkv_binding OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/bindings/hkv_binding.cu)
set_target_properties(hkv_binding PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(hkv_binding PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(hkv_binding PRIVATE nlohmann_json::nlohmann_json)

# ------------------
# mlkv_plus binding
# ------------------
add_library(mlkvplus_binding OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/bindings/mlkvplus_binding.cu)
set_target_properties(mlkvplus_binding PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
target_compile_features(mlkvplus_binding PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(mlkvplus_binding PRIVATE nlohmann_json::nlohmann_json mlkv_plus)


# ------------------
# rocksdb binding
# ------------------
add_library(rocksdb_binding OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/bindings/rocksdb_binding.cc)
set_target_properties(rocksdb_binding PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(rocksdb_binding PUBLIC cxx_std_20 cuda_std_20)
target_link_libraries(rocksdb_binding PRIVATE nlohmann_json::nlohmann_json rocksdb-shared OpenMP::OpenMP_CXX)


add_library(bridge_cpu SHARED src/bridge_cpu.cc)
target_compile_features(bridge_cpu PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(bridge_cpu PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
# ------------------------------------------------------------
# Please link the binding of CPU libraries to the bridge_cpu library
# ------------------------------------------------------------
target_link_libraries(bridge_cpu PUBLIC rocksdb_binding)
# ------------------------------------------------------------

add_library(bridge_cuda SHARED src/bridge_cuda.cu)
target_compile_features(bridge_cuda PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(bridge_cuda PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
# ------------------------------------------------------------
# Please link the binding of GPU libraries to the bridge_cuda library
# ------------------------------------------------------------
target_link_libraries(bridge_cuda PUBLIC hkv_binding mlkvplus_binding)
# ------------------------------------------------------------





# Add pybind11 subdirectory for Python bindings
add_subdirectory(python)
