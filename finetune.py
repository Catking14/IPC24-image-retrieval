from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datasets import Dataset, load_dataset

import torch
from torch.utils.data import DataLoader
from torch import nn

import clip
import math
import argparse
from tqdm import tqdm

# https://github.com/openai/CLIP/issues/57
# for mixed-precision training, only for GPU
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def train(args):
    # load model
    print("Loading model...")
    device = "cuda"
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(device)

    # load dataset
    print("Loading dataset...")
    train_data = load_dataset("imagefolder", data_dir="./dataset/natural_list_2021", split="train", num_proc=8)
    val_data = load_dataset("imagefolder", data_dir="./dataset/natural_list_2021", split="validation", num_proc=8)
    print(train_data.column_names)
    print(train_data[0])
    # train_data = train_data.map(lambda e: processor(text=e["text"], images=e["image"], return_tensors="pt", padding=True) , remove_columns=["image", "text"])    # remove original raw data
    # val_data = val_data.map(lambda e: processor(text=e["text"], images=e["image"], return_tensors="pt", padding=True), remove_columns=["image", "text"])

    # train_dataloader = DataLoader(train_data, batch_size=args.train_bsize, collate_fn=lambda x: x)
    # val_dataloader = DataLoader(val_data, batch_size=args.val_bsize, collate_fn=lambda x: x)

    # set hyper parameters and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    scaler = torch.cuda.amp.GradScaler()    # mixed precision
    img_loss = nn.CrossEntropyLoss()
    text_loss = nn.CrossEntropyLoss()
    epochs = 10
    print("Start Training...")
    for epoch in range(epochs):
        bar = tqdm(range(math.ceil(len(train_data) / args.train_bsize)))
        vbar = tqdm(range(math.ceil(len(val_data) / args.val_bsize)))

        batch_start = 0

        model.train()
        for batch in bar:
            # reset optimizer gradient
            optimizer.zero_grad()
            
            batch_buffer_img = []
            batch_buffer_text = []
            batch_size = args.train_bsize

            for offset in range(args.train_bsize):
                if batch_start + offset < len(train_data):
                    batch_buffer_img.append(train_data[batch_start + offset]["image"])
                    batch_buffer_text.append(train_data[batch_start + offset]["text"])
                else:
                    batch_size = offset
                    break

            image_text_tensor = processor(text=batch_buffer_text, images=batch_buffer_img, return_tensors="pt", padding=True)
            
            # organize data fields 
            # transfer to tensor and move to device
            image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(device)
            image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(device)
            image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(device)         

            # autocast with mixed precision
            with torch.cuda.amp.autocast():
                # forward - use fp16
                outputs = model(**image_text_tensor)

                # calculate loss
                y = torch.arange(batch_size, device=device)
                loss = (img_loss(outputs.logits_per_image, y) + text_loss(outputs.logits_per_text, y)) / 2  # take mathematical mean

            # backward - use fp16
            # loss.backward()
            scaler.scale(loss).backward()

            # optimizer step() - use fp32
            # convert_models_to_fp32(model)
            # optimizer.step()
            scaler.step(optimizer)
            # clip.model.convert_weights(model)   # back to fp16
            # model.half()
            scaler.update()
            batch_start += args.train_bsize

        print(f"Epoch {epoch}/{epochs}, loss = {loss.item()}.")

        
        # validation variables
        total_loss = 0
        batch_start = 0
        it = len(val_data)

        model.eval()
        for batch in vbar:
            batch_buffer_img = []
            batch_buffer_text = []
            batch_size = args.val_bsize

            for offset in range(args.val_bsize):
                if batch_start + offset < it:
                    batch_buffer_img.append(val_data[batch_start + offset]["image"])
                    batch_buffer_text.append(val_data[batch_start + offset]["text"])
                else:
                    batch_size = offset
                    break

            image_text_tensor = processor(text=batch_buffer_text, images=batch_buffer_img, return_tensors="pt", padding=True)
            
            # organize data fields 
            # transfer to tensor and move to device
            image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(device)
            image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(device)
            image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(device)     

            # forward
            outputs = model(**image_text_tensor)

            # calculate loss
            y_val = torch.arange(batch_size, device=device)
            val_loss = (img_loss(outputs.logits_per_image, y_val) + text_loss(outputs.logits_per_text, y_val)) / 2  # take mathematical mean
            total_loss += val_loss.item()
            batch_start += args.val_bsize

        print(f"Validation loss for epoch {epoch} = {total_loss / it}.")


    # save model
    model.save_pretrained(args.save)
    processor.save_pretrained(args.save)

def validation(args):
    # load model
    print("Loading model...")
    device = "cuda"
    model = CLIPModel.from_pretrained(args.model)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.to(device)

    # load dataset
    print("Loading dataset...")
    val_data = load_dataset("imagefolder", data_dir="./dataset/natural_list_2021", split="validation", num_proc=8)
    vbar = tqdm(range(math.ceil(len(val_data) / args.val_bsize)))
    img_loss = nn.CrossEntropyLoss()
    text_loss = nn.CrossEntropyLoss()

    # validation variables
    total_loss = 0
    batch_start = 0
    it = len(val_data)

    model.eval()
    for batch in vbar:
        batch_buffer_img = []
        batch_buffer_text = []
        batch_size = args.val_bsize

        for offset in range(args.val_bsize):
            if batch_start + offset < it:
                batch_buffer_img.append(val_data[batch_start + offset]["image"])
                batch_buffer_text.append(val_data[batch_start + offset]["text"])
            else:
                batch_size = offset
                break

        image_text_tensor = processor(text=batch_buffer_text, images=batch_buffer_img, return_tensors="pt", padding=True)
            
        # organize data fields 
        # transfer to tensor and move to device
        image_text_tensor["input_ids"] = image_text_tensor["input_ids"].to(device)
        image_text_tensor["attention_mask"] = image_text_tensor["attention_mask"].to(device)
        image_text_tensor["pixel_values"] = image_text_tensor["pixel_values"].to(device)     

        # forward
        outputs = model(**image_text_tensor)

        # calculate loss
        y_val = torch.arange(batch_size, device=device)
        val_loss = (img_loss(outputs.logits_per_image, y_val) + text_loss(outputs.logits_per_text, y_val)) / 2  # take mathematical mean
        total_loss += val_loss.item()
        batch_start += args.val_bsize

    print(f"Validation loss = {total_loss / it}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The path of CLIP model to train.")
    parser.add_argument("--save", type=str, default="", help="The directory path to save the trained model.")
    parser.add_argument("--train-bsize", type=int, default=32, help="The batch size for training.")
    parser.add_argument("--val-bsize", type=int, default=8, help="The batch size for validation.")
    parser.add_argument("--val-only", action="store_true", help="Only calculate validation loss for evaluation.")

    args = parser.parse_args()

    if args.val_only:
        validation(args)
    else:
        if args.save == "":
            raise ValueError("Please type in valid save path.")
        
        train(args)


