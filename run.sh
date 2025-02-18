export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eno1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_P2P_LEVEL=NVL
# export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=104857600
torchrun --nproc_per_node=1 \
         --nnodes=4 \
         --node_rank=0 \
         --master_addr=172.28.13.151 \
         --master_port=1234 \
         main.py
