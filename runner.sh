#!/bin/bash
spack load openmpi@4.1.4%gcc@12.2.0

np=4
device=""
export OMP_NUM_THREADS=$(($(nproc --all) / 2 / $np))    # 2 threads for hyperthreading $(($(nproc --all) / 2 / $np))

# set device
if [ -z "$2" ]; then
    device="cpu"
else
    device="$2"
fi

# set batch size
if [ -z "$3" ]; then
    batch_size=1
else
    batch_size=$3
fi

echo "Running $1 with $np processes and $OMP_NUM_THREADS threads per process"
echo "Using device $device"

# print running command
# bind-to none is for mpi to access all cores
echo "mpirun -np $np --bind-to none -x OMP_NUM_THREADS  \
python $1 --model ./model/clip-vit-base-patch32 --dataset ./dataset/natural_list_2021 --device $device --batch-size $batch_size"
mpirun -np $np --bind-to none -x OMP_NUM_THREADS python "$1" --model ./model/clip-vit-base-patch32 \
 --dataset ./dataset/natural_list_2021 --device $device --batch-size $batch_size