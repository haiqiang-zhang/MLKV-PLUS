# Find pybind11 - explicitly specify Python version to avoid conflicts
find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Debug: Print Python paths being used (can be removed in production)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_VERSION: ${Python_VERSION}")


set(module_name ycsb_binding)

# Create the pybind11 module
pybind11_add_module(${module_name} ycsb_binding.cpp)

# Set properties
target_compile_features(${module_name} PRIVATE cxx_std_20)
set_target_properties(${module_name} PROPERTIES CUDA_ARCHITECTURES OFF)





# Set RPATH to find the ycsb library and RocksDB
set(EXTRA_RPATH_DIRS
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/benchmark/ycsb_cpp"
)


set_target_properties(${module_name} PROPERTIES
    BUILD_RPATH "${EXTRA_RPATH_DIRS}"
    INSTALL_RPATH "${EXTRA_RPATH_DIRS}"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON)

# Link against the ycsb library and necessary dependencies
target_link_libraries(${module_name} PRIVATE bridge_cpu bridge_cuda)


execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "PYTHON_SITE: ${PYTHON_SITE}")

# Option to install locally instead of system-wide
option(INSTALL_PYTHON_LOCAL "Install Python binding locally" OFF)
if(INSTALL_PYTHON_LOCAL)
  set(PYTHON_INSTALL_DIR "${PROJECT_SOURCE_DIR}/benchmark/ycsb_cpp/python")
else()
  set(PYTHON_INSTALL_DIR "${PYTHON_SITE}/${module_name}")
endif()

# install the module to python site-packages
install(TARGETS ${module_name} 
        COMPONENT ${module_name}
        DESTINATION ${PYTHON_INSTALL_DIR})