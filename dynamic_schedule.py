from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

import torch
from PIL import Image

import math
import time
import argparse
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

world_size = 10

# tasklock = mp.Lock()

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
        all_batches,
        finished_batches,
        gpu_results,
        device) -> None:
    # load model
    print(f"Loading model...")
    model = CLIPModel.from_pretrained(args.model)
    model.to(device)

    all_sim_score = []
    inf_start_time = time.time()
    
    model.eval()
    with torch.no_grad():
        while True:
            # aquire lock for atomic add
            with finished_batches.get_lock():
                if finished_batches.value >= total_batch:
                    break
                finished_batches.value += 1
                    
            # get batch from producer
            batch = all_batches.get()

            # move tensor to device
            batch[0]["input_ids"] = batch[0]["input_ids"].to(device)
            batch[0]["attention_mask"] = batch[0]["attention_mask"].to(device)
            batch[0]["pixel_values"] = batch[0]["pixel_values"].to(device)

            # forward
            outputs = model(**batch[0])

            # calculate result
            sim = outputs.logits_per_image.squeeze(1).tolist()
                
            for i in range(len(sim)):
                all_sim_score.append(tuple((sim[i], batch[1][i])))
                
            del batch


    # summarize inference time
    print("\n====================== Execution Status ======================")
    print(f"    The inference time is {time.time() - inf_start_time} sec for {device} consumer.")

    # sort all similarity scores
    all_sim_score = sorted(all_sim_score, key=lambda element: element[0], reverse=True)

    if device == "cpu":
        return all_sim_score[:9]
    else:
        gpu_results.extend(all_sim_score[:9])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to use.")
    parser.add_argument("--dataset", type=str, default="", help="The directory path of the dataset to search.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for calculation.")
    parser.add_argument("--cpu-workers", type=int, default=0, help="The number of consumers for CPU inference. This number doesn't include the main process. If this argument is 0, the main process will do the inference.")
    parser.add_argument("--gpu-workers", type=int, default=1, help="The number of consumers for GPU inference. If torch.cuda.is_available() is False, this argument will be ignored.")
    parser.add_argument("--num-samples", type=int, help="Number of examples need to be searched. Used for testing only.")

    args = parser.parse_args()

    try:
        print(f"total size = {world_size}, threads per process = {os.environ['OMP_NUM_THREADS']}")
    except KeyError:    # OMP_NUM_THREADS is not set
        print(f"total size = {world_size}, threads per process = auto.")

    all_start_time = time.time()

    # database, rank_data_size = load_data(args.dataset)
    # database = load_dataset("catking-14/iNaturalist-2021-train-mini", split="train+validation")
    # load dataset
    print(f"Loading dataset...")
    database = load_dataset(args.dataset, split="train+validation", num_proc=world_size)

    # create processes for preprocessing
    rank_data_size = len(database) // world_size

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
    gpu_top_nines = manager.list()
    cpu_top_nines = manager.list()
    finished_batches = ctx.Value("i", 0)
    # tasklock = manager.Lock()
    gpu_workers = []
    cpu_workers = []

    top_nine = []

    with ctx.Pool(processes=world_size) as pool:
        proc_args = []
        gpu_worker_args = []
        for tid in range(world_size):
            if tid < world_size - 1:
                proc_args.append((args, tid, database.select([tid * rank_data_size + i for i in range(rank_data_size)]), tid * rank_data_size, prompt, all_batches))
            else:
                proc_args.append((args, tid, database.select([i for i in range(tid * rank_data_size, len(database))]), tid * rank_data_size, prompt, all_batches))

        # gpu workers
        for tid in range(args.gpu_workers):
            p = ctx.Process(target=search, args=(args, total_batch, all_batches, finished_batches, gpu_top_nines, "cuda"))
            p.start()
            gpu_workers.append(p)

        # cpu workers
        for tid in range(args.cpu_workers):
            p = ctx.Process(target=search, args=(args, total_batch, all_batches, finished_batches, cpu_top_nines, "cpu"))
            p.start()
            cpu_workers.append(p)

        # non-blocking start
        pool.starmap_async(load_and_preprocess, proc_args)

        # model inference
        top_nine = search(args, total_batch, all_batches, finished_batches, None, "cpu")
        
        for i in range(args.gpu_workers):
            gpu_workers[i].join()

        for i in range(args.cpu_workers):
            cpu_workers[i].join()

        # total sort
        top_nine.extend(gpu_top_nines)
        top_nine.extend(cpu_top_nines)
        top_nine = sorted(top_nine, key=lambda element: element[0], reverse=True) 
        top_nine = top_nine[:9]

    # plot results
    for pics in range(9):
        plt.subplot(3, 3, pics + 1)
        plt.imshow(database[top_nine[pics][1]]["image"])
    plt.show()


    print(f"    The total execution time is {time.time() - all_start_time} sec.")

    


