from benchmark.ycsb.YCSBController import YCSBController
from benchmark.ycsb.ConfigLoader import get_workload_config, get_binding_config


def main():
    
    
    # workload_names = ["Multi_Get_1024", "Multi_Get_2048", "Multi_Get_4096", "Multi_Get_8192", "Multi_Get_16384", "Multi_Get_32768", "Multi_Get_65536", "Multi_Get_131072", "Multi_Get_262144", "Multi_Get_524288", "Multi_Get_1048576", "Multi_Get_2097152"]
    workload_names = ["Multi_Get_1048576"]
    throughputs = []
    
    
    binding_name = "mlkv_plus"
    for workload_name in workload_names:
        workload_config = get_workload_config(workload_name)
        binding_config = get_binding_config(binding_name)
        controller = YCSBController(
            num_records=workload_config["num_records"],
            operations=workload_config["operations"],
            workload_name=workload_config["name"],
            distribution=workload_config["distribution"],
            zipfian_theta=workload_config["zipfian_theta"],
            orderedinserts=workload_config["orderedinserts"],
            data_integrity=workload_config["data_integrity"],
            min_field_length=workload_config["min_field_length"],
            max_field_length=workload_config["max_field_length"],
            field_count=workload_config["field_count"],
            binding_type="cpp",
            binding_name=binding_name,
            output_file=f"{binding_name}_result.json",
            binding_config=binding_config,
            gpu_device=binding_config["gpu_ids"],
            is_cuda=True,
            generator_num_processes=50
        )
        result = controller.run(num_ops=workload_config["ops"])
        throughputs.append(result["throughput"])
        print(f"Throughput: {result['throughput']}")
    
    print(throughputs)


if __name__ == "__main__":
    main()