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


def search(args):
    # load model
    print("Loading model...")
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(args.device)

    # load dataset
    print("Loading dataset...")
    database = load_dataset(args.dataset, split="train+validation", num_proc=8)
    # database = load_dataset("catking-14/iNaturalist-2021-train-mini", split="train+validation")
    if args.num_samples:
        if args.num_samples > len(database) or args.num_samples < 0:
            raise ValueError(f"Invalid number of samples (received {args.num_samples}).")
        pbar = tqdm(range(math.ceil(args.num_samples / args.batch_size)))
    else:
        pbar = tqdm(range(math.ceil(len(database) / args.batch_size)))

    # inference variables
    batch_start = 0
    prompt = input("Type in what you want to search: ")
    prompt = [prompt.lower()]
    all_sim_score = []
    inf_start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch in pbar:
            pbar.set_description(f"Progressing batch {batch}")
            batch_buffer_img = []

            for offset in range(args.batch_size):
                if args.num_samples:
                    if batch_start + offset < args.num_samples:
                        batch_buffer_img.append(database[batch_start + offset]["image"])
                elif batch_start + offset < len(database):
                    batch_buffer_img.append(database[batch_start + offset]["image"])
                else:
                    break

            image_text_tensor = processor(text=prompt, images=batch_buffer_img, return_tensors="pt", padding=True)
                
            # organize data fields 
            # transfer to tensor and move to device
            image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(args.device)
            image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(args.device)
            image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(args.device)     

            # forward
            outputs = model(**image_text_tensor)

            # calculate result
            sim = outputs.logits_per_image.squeeze(1).tolist()
            
            for i in range(len(sim)):
                all_sim_score.append(tuple((sim[i], batch_start + i)))

            batch_start += args.batch_size

    # summarize inference time
    print("\n====================== Execution Status ======================")
    print(f"    The inference time is {time.time() - inf_start_time} sec.")

    # sort all similarity scores
    sorted(all_sim_score, key=lambda element: element[0], reverse=True)

    # plot results
    for pics in range(9):
        plt.subplot(3, 3, pics + 1)
        plt.imshow(database[all_sim_score[pics][1]]["image"])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to train.")
    parser.add_argument("--dataset", type=str, default="", help="The directory path of the dataset to search.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for calculation.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="The device used for model inference. Available choices: cpu, cuda.")
    parser.add_argument("--num-samples", type=int, help="Number of examples need to be searched. Used for testing only.")

    args = parser.parse_args()

    all_start_time = time.time()
    search(args)

    print(f"    The total execution time is {time.time() - all_start_time} sec.")

    


