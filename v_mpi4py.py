from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import itertools
from mpi4py import MPI

# https://github.com/openai/CLIP/issues/57
# for mixed-precision training, only for GPU
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def search(args):
    world_size = comm.Get_size()
    my_rank = comm.Get_rank()

    # load model
    print("Loading model...")
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(args.device)

    # load dataset
    database = load_dataset("imagefolder", data_dir=args.dataset, split="train+validation")

    # calucate batch workload of each thread
    total_batch_workload = math.ceil(len(database) / args.batch_size) 
    avg_batch_workload = total_batch_workload // world_size
    remain_batch_workload = total_batch_workload % world_size
    if(remain_batch_workload > 0): standard_size = (avg_batch_workload + 1) * args.batch_size
    else: standard_size = avg_batch_workload * args.batch_size

    my_batch_workload = avg_batch_workload + (my_rank < remain_batch_workload)
    my_batch_workload_start = my_rank * avg_batch_workload + min(my_rank, remain_batch_workload)

    # inference variables
    my_batch_start = my_batch_workload_start * args.batch_size
    print(f"Rank {my_rank}: {my_batch_workload_start}, {my_batch_workload}, {total_batch_workload}, {avg_batch_workload}, {remain_batch_workload}")

    if(my_rank == 0): prompt = "Bird"
    else: prompt = None
    prompt = comm.bcast(prompt, root=0)
    prompt = [prompt.lower()]
    print(f"Rank {my_rank}: {prompt}")

    my_all_sim_score = []
    inf_start_time = time.time()
    
    print(f"Rank {my_rank}: Model evaluation...")
    model.eval()
    print(f"Rank {my_rank}: Model start...")
    my_pbar = tqdm(range(my_batch_workload_start, my_batch_workload_start + my_batch_workload))
    for batch in my_pbar:
        my_pbar.set_description(f"Rank {my_rank}: Progressing batch {batch}")
        batch_buffer_img = []

        for offset in range(args.batch_size):
            if my_batch_start + offset < len(database):
                batch_buffer_img.append(database[my_batch_start + offset]["image"])
            else:
                break

        image_text_tensor = processor(text=prompt, images=batch_buffer_img, return_tensors="pt", padding=True)
            
        # organize data fields 
        # transfer to tensor and move to device
        image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(args.device)
        image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(args.device)
        image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(args.device)     

        # forward
        my_outputs = model(**image_text_tensor)

        # calculate result
        sim = my_outputs.logits_per_image.squeeze(1).tolist()
        
        for i in range(len(sim)):
            my_all_sim_score.append(tuple((sim[i], my_batch_start + i))) 

        my_batch_start += args.batch_size

    # summarize inference time
    print("\n====================== Execution Status ======================")
    print(f"For thread {my_rank} The inference time is {time.time() - inf_start_time} sec.")


    # padding 
    if(len(my_all_sim_score) < standard_size):
        for i in range(standard_size - len(my_all_sim_score)): my_all_sim_score.append(tuple((-1, -1)))
    comm.Barrier()

    # Gather the all_sim_score of each thread to root
    my_all_sim_score = comm.gather(my_all_sim_score, root=0)
    if(my_rank == 0):
        # sort all similarity scores
        all_sim_score = list(itertools.chain(*my_all_sim_score))
        all_sim_score = sorted(all_sim_score, key=lambda element: element[0], reverse=True)
        for pics in range(9):
            print(all_sim_score[pics][1])
            # plt.subplot(3, 3, pics + 1)
            # plt.imshow(database[all_sim_score[pics][1]]["image"])
        # plt.show()

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to train.")
    parser.add_argument("--dataset", type=str, default="", help="The directory path of the dataset to search.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for calculation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="The device used for model inference. Available choices: cpu, cuda.")

    args = parser.parse_args()

    all_start_time = time.time()
    search(args)

    print(f"    The total execution time is {time.time() - all_start_time} sec.")