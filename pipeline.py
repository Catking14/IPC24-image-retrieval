from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from mpi4py import MPI

import math
import time
import argparse
import os
import queue
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

world_size = 10

def load_and_preprocess(
        args, 
        tid, 
        database, 
        data_start_id,
        prompt, 
        all_batches) -> None:
    processor = CLIPProcessor.from_pretrained(args.model)
    print(f"len of dataset = {len(database)} from rank {tid}")

    # for i in range(1000000):
    #     a = 0
    #     for j in range(100000):
    #         a += 1

    # calculate number of local batches
    if args.num_samples:
        if args.num_samples > len(database) or args.num_samples < 0:
            raise ValueError(f"Invalid number of samples (received {args.num_samples}).")
        pbar = math.ceil(args.num_samples / args.batch_size)
    else:
        pbar = math.ceil(len(database) / args.batch_size)

    batch_start = 0

    # start processing data to batch
    print(f"Start processing in rank {tid}")
    print(f"pbar is {pbar}, batch size is {args.batch_size}")
    for batch in range(pbar):
        # pbar.set_description(f"Progressing batch {batch} on rank {tid}")
        batch_buffer_img = []
        id_list = []

        for offset in range(args.batch_size):
            if args.num_samples:
                if batch_start + offset < args.num_samples:
                    batch_buffer_img.append(database[batch_start + offset]["image"])
                    id_list.append(data_start_id + batch_start + offset)
            elif batch_start + offset < len(database):
                batch_buffer_img.append(database[batch_start + offset]["image"])
                id_list.append(data_start_id + batch_start + offset)
            else:
                break

        image_text_tensor = processor(text=prompt, images=batch_buffer_img, return_tensors="pt", padding=True)

        # organize data fields 
        # transfer to tensor and move to device
        image_text_tensor["input_ids"] = image_text_tensor["input_ids"]
        image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"]
        image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"]

        # local_batches.append((image_text_tensor, id_list))
        while True:    # 600 for approximated safe pipe upper bound, for preventing deadlock
            if all_batches.qsize() < 600:
                all_batches.put((image_text_tensor, id_list), timeout=0.2)
                break

        batch_start += args.batch_size

    # append local batches to all batches
    # all_batches.extend(local_batches)
    print(f"Finish processing in rank {tid}")


def search(
        args,
        total_batch, 
        all_batches) -> None:
    # load model
    print(f"Loading model...")
    model = CLIPModel.from_pretrained(args.model)
    model.to(args.device)

    all_sim_score = []
    inf_start_time = time.time()
    pbar = tqdm(range(total_batch))

    model.eval()
    with torch.no_grad():
        for i in pbar:
            # if i >= 695: print(all_batches.empty(), all_batches.full(), all_batches.qsize())
            # get batch from producer
            batch = all_batches.get()

            # move tensor to device
            batch[0]["input_ids"] = batch[0]["input_ids"].to(args.device)
            batch[0]["attention_mask"] = batch[0]["attention_mask"].to(args.device)
            batch[0]["pixel_values"] = batch[0]["pixel_values"].to(args.device)

            # forward
            outputs = model(**batch[0])

            # calculate result
            sim = outputs.logits_per_image.squeeze(1).tolist()
            
            for i in range(len(sim)):
                all_sim_score.append(tuple((sim[i], batch[1][i])))
            
            del batch

    # summarize inference time
    print("\n====================== Execution Status ======================")
    print(f"    The inference time is {time.time() - inf_start_time} sec.")

    # sort all similarity scores
    all_sim_score = sorted(all_sim_score, key=lambda element: element[0], reverse=True)

    return all_sim_score[:9]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to train.")
    parser.add_argument("--dataset", type=str, default="", help="The directory path of the dataset to search.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for calculation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="The device used for model inference. Available choices: cpu, cuda.")
    parser.add_argument("--num-samples", type=int, help="Number of examples need to be searched. Used for testing only.")

    args = parser.parse_args()

    print(f"total size = {world_size}, threads per process = {os.environ['OMP_NUM_THREADS']}")

    all_start_time = time.time()

    # database, rank_data_size = load_data(args.dataset)
    # database = load_dataset("catking-14/iNaturalist-2021-train-mini", split="train+validation")
    # load dataset
    print(f"Loading dataset...")
    database = load_dataset(args.dataset, split="train+validation", num_proc=world_size)

    # create processes for preprocessing
    rank_data_size = math.ceil(len(database) / world_size)
    worker = []

    # calculate number of total batches
    if args.num_samples:
        if args.num_samples > len(database) or args.num_samples < 0:
            raise ValueError(f"Invalid number of samples (received {args.num_samples}).")
        total_batch = math.ceil(args.num_samples / args.batch_size) * world_size
    else:
        total_batch = math.ceil(rank_data_size / args.batch_size) * (world_size - 1)
        total_batch += math.ceil((len(database) % rank_data_size) / args.batch_size) if len(database) % rank_data_size else math.ceil(rank_data_size / args.batch_size)

    # inference variables
    prompt = input("Type in what you want to search: ")
    prompt = [prompt.lower()]
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    all_batches = manager.Queue()
    top_nine = []

    with ctx.Pool(processes=world_size) as pool:
        proc_args = []
        for tid in range(world_size):
            if tid < world_size - 1:
                proc_args.append((args, tid, database.select([tid * rank_data_size + i for i in range(rank_data_size)]), tid * rank_data_size, prompt, all_batches))
            else:
                proc_args.append((args, tid, database.select([i for i in range(tid * rank_data_size, len(database))]), tid * rank_data_size, prompt, all_batches))

        # non-blocking start
        pool.starmap_async(load_and_preprocess, proc_args)

        # model inference
        top_nine = search(args, total_batch, all_batches)

    # plot results
    for pics in range(9):
        plt.subplot(3, 3, pics + 1)
        plt.imshow(database[top_nine[pics][1]]["image"])
    plt.show()
    
    # # gather data for each rank
    # for tid in range(world_size):
    #     if tid < world_size - 1:
    #         # database = database[tid * rank_data_size : (tid + 1) * rank_data_size]
    #         p = ctx.Process(target=load_and_preprocess, args=(args, tid, database.select([tid * rank_data_size + i for i in range(rank_data_size)]), tid * rank_data_size, prompt, all_batches))
    #         p.start()
    #         worker.append(p)
    #     else:
    #         # database = database[tid * rank_data_size :]
    #         p = ctx.Process(target=load_and_preprocess, args=(args, tid, database.select([i for i in range(tid * rank_data_size, len(database))]), tid * rank_data_size, prompt, all_batches))
    #         p.start()
    #         worker.append(p)

    # # join all processes to main process
    # for subproc in worker:
    #     subproc.join()
    #     subproc.close()


    print(f"    The total execution time is {time.time() - all_start_time} sec.")

    


