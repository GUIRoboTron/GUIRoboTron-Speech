import os
import json


cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
role = cluster_spec["role"]
assert role == "worker", "{} vs worker".format(role)
node_rank = cluster_spec["index"]
nnodes = len(cluster_spec[role])
nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
master = cluster_spec[role][0]


master_port = "port"
master_addr = "localhost"

master_strs = master.split(":")

if len(master_strs) == 2:
    master_addr, master_ports = master_strs[0], master_strs[1]
    master_ports = master_ports.split(",")
    master_port = master_ports[0]
else:
    master_addr = master_strs[0]

if nnodes == 1:
    master_addr = "localhost"
    
if master_port == "port":
    master_port = "12459"



print("{} {} {} {} {}".format(nnodes, nproc_per_node, master_addr, master_port, node_rank))