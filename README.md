# MLKV+

## Operations workflow

* **Multiset**
![Multiset Workflow](imgs/multiset_workflow.svg)

* **Multiget**
![Multiget Workflow](imgs/multiget_workflow.svg)


## How to build MLKV+(PyTorch)
```bash
# clone submodule
git submodule update --init --recursive

# create conda envs
conda env create -f env.yml
conda activate mlkv_plus

# build MLKV+(PyTorch)
MAX_JOBS=$(($(nproc)-1)) CUDA_SM="86" pip install -e .
```
* Please change `CUDA_SM` to your own [Computer Compacity](https://developer.nvidia.com/cuda-gpus) of GPU.
* You can change `MAX_JOBS` to your wanted number of jobs to compile.

## How to build libmlkvplus
```bash
# clone submodule
git submodule update --init --recursive

# create conda envs
conda env create -f env.yml
conda activate mlkv_plus

# build libmlkvplus
mkdir -p build && ./build
cmake .. -Dsm=86
make -j$(($(nproc)-1))

# install ycsb-cpp binding
cmake --install . --component ycsb_binding
```
* Please change `-Dsm` to your own [Computer Compacity](https://developer.nvidia.com/cuda-gpus) of GPU.

## YCSB benchmark

![YCSB Performance](imgs/ycsb.svg)

* If you already install MLKV+(PyTorch), you can directly use the benchmark toolkit. If you only want to install YCSB benchmark, please install by `MLKV_BENCHMARK_ONLY=true pip install -e ".[benchmark]"`
* Run Visualizer: `python -m benchmark.visualizer`
* Run single benchmark: `python ./benchmark/single_run.py`


## How to install GPUDirect Storage
To be added

## Known issues
* The G-Page Cache might raise IO errors, like: "Failed to get from SST files: IO error: GDS read failed: Incomplete GDS read: requested 262144 bytes (aligned), got 262144 bytes at offset 10223616, need at least 265268 bytes for requested range" in Get operation.
* The Multiget logic is not perfect.