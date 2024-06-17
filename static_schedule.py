from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

import torch
from PIL import Image

import math
import time
import argparse
import os
import gc
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

world_size = 10

# tasklock = mp.Lock()

def device_ratio_prof(
        args,
        profile_iterations,
        profile_pic_path) -> tuple:
    """
    Return the ratio of GPU/CPU workload distribution by simple profiling.
    """
    # no gpu to use
    if not torch.cuda.is_available() or args.gpus == 0:
        return (0, 1)
    
    device = "cpu"
    # load model
    model = CLIPModel.from_pretrained(args.model)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(args.model)

    # load image
    images = [Image.open(profile_pic_path)]
    prompt = ["a photo of a cat"]

    cpu_inf_time = 0
    gpu_inf_time = 0

    for i in range(profile_iterations + 1):
        input_img = processor(text=prompt, images=images, return_tensors="pt", padding=True)
        start = time.time()
        input_img["input_ids"] = input_img["input_ids"].to(device)
        input_img["attention_mask"] = input_img["attention_mask"].to(device)
        input_img["pixel_values"] = input_img["pixel_values"].to(device)
        outputs = model(**input_img)
        end = time.time()
        cpu_inf_time = cpu_inf_time + end - start if i > 0 else cpu_inf_time

        print(f"CPU Inference time in {i} th iteration: {end - start}")

    print(f"Average CPU Inference time: {cpu_inf_time / 10}")

    device = "cuda"
    model.to(device)

    for i in range(profile_iterations + 1):
        input_img_gpu = processor(text=prompt, images=images, return_tensors="pt", padding=True)
        start = time.time()
        input_img_gpu["input_ids"] = input_img_gpu["input_ids"].to(device)
        input_img_gpu["attention_mask"] = input_img_gpu["attention_mask"].to(device)
        input_img_gpu["pixel_values"] = input_img_gpu["pixel_values"].to(device)
        outputs = model(**input_img_gpu)
        end = time.time()
        gpu_inf_time = gpu_inf_time + end - start if i > 0 else gpu_inf_time

        print(f"GPU Inference time in {i} th iteration: {end - start}")

    print(f"Average GPU Inference time: {gpu_inf_time / 10}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return (cpu_inf_time / 10 * args.gpus, gpu_inf_time / 20)   # gpu metric is twice faster since all cpu cores will be used for gpu inference, thus lowering cpu performance


def load_and_preprocess(
        args, 
        tid, 
        database, 
        data_start_id,
        prompt, 
        all_batches,
        device) -> None:
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
        image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(device)
        image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(device)
        image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(device)

        # local_batches.append((image_text_tensor, id_list))
        while True:    # 600 for approximated safe pipe upper bound, for preventing deadlock
            # print(all_batches.qsize())
            if all_batches.qsize() < 600 // (args.gpus + 1):    # 1 for cpu
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
        gpu_result,
        device) -> None:
    # load model
    print(f"Loading model...")
    model = CLIPModel.from_pretrained(args.model)
    model.to(device)

    all_sim_score = []
    inf_start_time = time.time()
    
    if device == "cpu":
        pbar = tqdm(range(total_batch))

        model.eval()
        with torch.no_grad():
            for i in pbar:
                # print(all_batches.empty(), all_batches.full(), all_batches.qsize())
                # get batch from producer
                batch = all_batches.get()

                # forward
                outputs = model(**batch[0])

                # calculate result
                sim = outputs.logits_per_image.squeeze(1).tolist()
                
                for i in range(len(sim)):
                    all_sim_score.append(tuple((sim[i], batch[1][i])))
                
                del batch
    else:   # gpu
        model.eval()
        with torch.no_grad():
            print("GPU consumer is started.")
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
            
            print(f"GPU consumer is finished with {finished_batches.value} batches.")


    # summarize inference time
    print("\n====================== Execution Status ======================")
    print(f"    The inference time is {time.time() - inf_start_time} sec for {device} consumer.")

    # sort all similarity scores
    all_sim_score = sorted(all_sim_score, key=lambda element: element[0], reverse=True)

    if device == "cpu":
        return all_sim_score[:9]
    else:
        gpu_result.extend(all_sim_score[:9])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to use.")
    parser.add_argument("--dataset", type=str, default="", help="The directory path of the dataset to search.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for calculation.")
    parser.add_argument("--gpus", type=int, default=1, help="The number of consumers for GPU inference. If torch.cuda.is_available() is False, this argument will be ignored.")
    parser.add_argument("--num-samples", type=int, help="Number of examples need to be searched. Used for testing only.")
    parser.add_argument("--profile-pic", type=str, default="./dataset/local_test/1.jpg", help="The path of the image used for profiling.")

    args = parser.parse_args()

    print(f"total size = {world_size}, threads per process = {os.environ['OMP_NUM_THREADS']}")

    all_start_time = time.time()

    # database, rank_data_size = load_data(args.dataset)
    # database = load_dataset("catking-14/iNaturalist-2021-train-mini", split="train+validation")
    # load dataset
    print(f"Loading dataset...")
    database = load_dataset(args.dataset, split="train+validation", num_proc=world_size)

    # device capability profiling
    print("Start Profiling")
    gpu_amount, cpu_amount = device_ratio_prof(args, 10, args.profile_pic)
    cpu_amount *= 100   # prevent overflow
    gpu_amount *= 100   # prevent overflow, and multiplies number models able to port to GPU

    # allocate workload
    cpu_data_idx = [i for i in range(int(cpu_amount / (cpu_amount + gpu_amount) * len(database)))]
    gpu_data_idx = [i for i in range(int(cpu_amount / (cpu_amount + gpu_amount) * len(database)), len(database))]
    cpu_dataset = database.select(cpu_data_idx)
    gpu_dataset = database.select(gpu_data_idx)
    print(f"CPU amount: {len(cpu_dataset)}, GPU amount: {len(gpu_dataset)}")

    # create processes for preprocessing
    cpu_worker_size = int(cpu_amount / (cpu_amount + gpu_amount) * world_size)
    gpu_worker_size = world_size - cpu_worker_size
    cpu_rank_data_size = math.ceil(len(cpu_dataset) / cpu_worker_size)
    gpu_rank_data_size = math.ceil(len(gpu_dataset) / gpu_worker_size) if gpu_worker_size > 0 else 0

    # calculate number of total batches
    if args.num_samples:
        if args.num_samples > cpu_rank_data_size or args.num_samples > gpu_rank_data_size or args.num_samples < 0:
            raise ValueError(f"Invalid number of samples (received {args.num_samples}).")
        cpu_batch = math.ceil(args.num_samples / args.batch_size) * cpu_worker_size
        gpu_batch = math.ceil(args.num_samples / args.batch_size) * gpu_worker_size
    else:
        cpu_batch = math.ceil(cpu_rank_data_size / args.batch_size) * (cpu_worker_size - 1)
        cpu_batch += math.ceil((len(cpu_dataset) % cpu_rank_data_size) / args.batch_size) if len(cpu_dataset) % cpu_rank_data_size else math.ceil(cpu_rank_data_size / args.batch_size)
        if gpu_amount > 0:
            gpu_batch = math.ceil(gpu_rank_data_size / args.batch_size) * (gpu_worker_size - 1)
            gpu_batch += math.ceil((len(gpu_dataset) % gpu_rank_data_size) / args.batch_size) if len(gpu_dataset) % gpu_rank_data_size else math.ceil(gpu_rank_data_size / args.batch_size)
        else:
            gpu_batch = 0
    
    print(f"CPU batch size: {cpu_batch}, GPU batch size: {gpu_batch}")

    # inference variables
    prompt = input("Type in what you want to search: ")
    prompt = [prompt.lower()]
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    # tasklock = ctx.lock()
    cpu_tasks = manager.Queue()
    gpu_tasks = manager.Queue()
    gpu_top_nines = manager.list()
    finished_batches = ctx.Value("i", 0)
    gpu_workers = []

    top_nine = []

    with ctx.Pool(processes=world_size) as pool:
        proc_args = []
        gpu_worker_args = []
        for tid in range(world_size):
            if tid < cpu_worker_size - 1:
                proc_args.append((args, tid, cpu_dataset.select([tid * cpu_rank_data_size + i for i in range(cpu_rank_data_size)]), tid * cpu_rank_data_size, prompt, cpu_tasks, "cpu"))
            elif tid == cpu_worker_size - 1:
                proc_args.append((args, tid, cpu_dataset.select([i for i in range(tid * cpu_rank_data_size, len(cpu_dataset))]), tid * cpu_rank_data_size, prompt, cpu_tasks, "cpu"))
            elif tid < world_size - 1:
                proc_args.append((args, tid, gpu_dataset.select([(tid - cpu_worker_size) * gpu_rank_data_size + i for i in range(gpu_rank_data_size)]), (tid - cpu_worker_size) * gpu_rank_data_size + len(cpu_dataset), prompt, gpu_tasks, "cpu"))
            else:
                proc_args.append((args, tid, gpu_dataset.select([i for i in range((tid - cpu_worker_size) * gpu_rank_data_size, len(gpu_dataset))]), (tid - cpu_worker_size) * gpu_rank_data_size + len(cpu_dataset), prompt, gpu_tasks, "cpu"))

        # gpu workers
        for tid in range(args.gpus):
            p = ctx.Process(target=search, args=(args, gpu_batch, gpu_tasks, finished_batches, gpu_top_nines, "cuda"))
            p.start()
            gpu_workers.append(p)

        # non-blocking start
        pool.starmap_async(load_and_preprocess, proc_args)

        # model inference
        top_nine = search(args, cpu_batch, cpu_tasks, None, None, "cpu")
        
        for i in range(args.gpus):
            gpu_workers[i].join()

        # total sort
        top_nine.extend(gpu_top_nines)
        top_nine = sorted(top_nine, key=lambda element: element[0], reverse=True) 
        top_nine = top_nine[:9]

    # plot results
    for pics in range(9):
        plt.subplot(3, 3, pics + 1)
        plt.imshow(database[top_nine[pics][1]]["image"])
    plt.show()


    print(f"    The total execution time is {time.time() - all_start_time} sec.")

    


