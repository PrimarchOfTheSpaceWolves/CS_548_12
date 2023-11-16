import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.models import (list_models, 
                                get_model, 
                                get_weight, 
                                get_model_weights)
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision.io import read_image
from dataclasses import dataclass
from accelerate import Accelerator, notebook_launcher
from diffusers import DiffusionPipeline, DDPMPipeline
from diffusers import EulerDiscreteScheduler
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel
import tqdm
import PIL
import numpy as np
import os
from datasets import load_dataset
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    eval_batch_size = 16
    mixed_precision = "fp16"
    output_dir = "BasicGenTrain"
    gradient_accumulation_steps = 1
    start_epoch = 0
    total_epochs = 100
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    overwrite_output_dir = True
    seed = 0
    

def main():
    config = TrainingConfig()
    
    dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset",
        split="train"
    )
    
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, 
                           config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    def transform(examples):
        images = [preprocess(image.convert("RGB"))
                  for image in examples["image"]]
        return {"images": images}
    
    dataset.set_transform(transform)
    
    dataloader = DataLoader(dataset,
                            batch_size=config.train_batch_size,
                            shuffle=True)
    
    model = UNet2DModel(
        in_channels = 3,
        out_channels = 3,
        sample_size = config.image_size,
        layers_per_block = 2,
        block_out_channels = (128,128,256,256,512,512),
        down_block_types = [
            "DownBlock2D","DownBlock2D",
            "DownBlock2D","DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ],
        up_block_types = [
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"
        ]        
    )
    
    print(model)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate)
    
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader)*config.total_epochs))
    
    def evaluate(config, epoch, pipeline):
        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=torch.manual_seed(config.seed)
        ).images
        
        image_grid = make_image_grid(images, 4, 4)
        
        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(os.path.join(test_dir,
                                     "Image_%04d.png" % epoch))
    
    def train_loop(config, model, optimizer,
                   noise_scheduler, dataloader,
                   lr_scheduler):
        
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs")
        )
        
        (model,
         optimizer,
         dataloader,
         noise_scheduler,
         lr_scheduler) = accelerator.prepare(
                                        model,
                                        optimizer,
                                        dataloader,
                                        noise_scheduler,
                                        lr_scheduler
                                        )
         
        if accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("training")  
        
        global_step = 0
        
        for epoch in range(config.total_epochs):        
            progress_bar = tqdm.tqdm(total=len(dataloader),
                                disable=not accelerator.is_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for batch in dataloader:
                clean_images = batch["images"]
                noise = torch.randn(clean_images.shape).to(
                            clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(0,
                                noise_scheduler.config.num_train_timesteps,
                                (bs,),
                                device=clean_images.device).long()
                noisy_images = noise_scheduler.add_noise(
                    clean_images,
                    noise,
                    timesteps
                )
                
                with accelerator.accumulate(model):
                    noise_pred = model(noisy_images,
                                       timesteps,
                                       return_dict=False)[0]
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "step": global_step,
                    "lr": lr_scheduler.get_last_lr()[0]                     
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1
                
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler)
                
                if(epoch+1)%config.save_image_epochs == 0:
                    evaluate(config, epoch, pipeline)
                    
                if(epoch+1)%config.save_model_epochs == 0:
                    pipeline.save_pretrained(config.output_dir)
                
    args = (config, model, optimizer, noise_scheduler,
            dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1)  
        
if __name__ == "__main__":
    main()
    