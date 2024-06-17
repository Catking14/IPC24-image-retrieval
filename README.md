# IPC24 Optimizations on Text-Image Retrieval with CLIP ViT

This is the repository of implementations of text-image retrieval optimizations with CLIP vision transformers (ViT) in Introduction to Parallel Programming 2024 in National Tsing Hua University. The introduction slides can be viewed via [this link](https://docs.google.com/presentation/d/1p5cd9V_pF1e7GFXyg44guSlLcpQ1t1FVnzNHXE3mcUE/edit?usp=sharing).

## Quick Start

To run with the baseline code, just simply run

```
python baseline.py --model <model path> --dataset <dataset path>
```

We transfered our image database to a huggingface dataset for the ease of access, and it would be convenient to try with other existing dataset. Please makesure the image database you want to use can be accessed by `load_dataset()` of the `datasets` module from huggingface.

There are some optional flags for baseline code:

- `--device`: Select the device used for inference. Available options: `"cpu", "cuda"`.
- `--batch-size`: Set the batch size for inference.
- `--num-samples`: Set the number of images to iterate. **This is still an experimental feature and might have some bugs**.

For other files such `pipeline.py` for parallelized `CLIPProcessor` or `dynamic_schedule.py` for task dynamic scheduling version, use `python <filename>.py --help` for more information.

### runner.sh

`runner.sh` is used for `v_mpi4py.py` for pure model level data parallelism with MPI. Before you use this wrapper, make sure you have the MPI installed on your computer and `mpi4py` module installed for your python, and modify the second line in `runner.sh`. Current script use Spack to load a OpenMPI on our cluster, so make sure to modify that to load or use you MPI on your system correctly. Then, you can run the wrapper like

```
./runner.sh v_mpi4py.py <device> <batch size>
```

to run with data parallelism with MPI.
