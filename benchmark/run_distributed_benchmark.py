# torchrun --nproc_per_node=4 benchmark/run_distributed_benchmark.py

from benchmark.ycsb.DistYCSBController import DistYCSBController
from benchmark.ycsb.ConfigLoader import get_workload_config, get_binding_config


def main():
    
    write_probabilities = ["20_Write_131072", "40_Write_131072", "60_Write_131072", "80_Write_131072", "100_Write_131072", "0_Write_131072"]
    results_throughput = []

    
    for batch_size in write_probabilities:
    
        binding_name = "DistMLKVPlusBinding"
        workload_config = get_workload_config(batch_size)
        binding_config = get_binding_config(binding_name)
        controller = DistYCSBController(
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
            binding_name=binding_name,
            output_file=f"{binding_name}_result.json",
            binding_config=binding_config,
            generator_num_processes=50
        )
        result = controller.run(num_ops=workload_config["ops"])
        results_throughput.append(result["throughput"])
    
    print(results_throughput)


if __name__ == "__main__":
    main()