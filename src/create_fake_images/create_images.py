import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd

output_dir = ["sd_2_1","sd_xl_1","sd_xl_base","sd_1"]

models = ["stabilityai/stable-diffusion-2-1","stabilityai/stable-diffusion-xl-refiner-1.0",
          "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5"]

main_folder_path = "/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/"
df = pd.read_csv(main_folder_path+'captions/lsun/bedroom/captions.csv')


# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
for index, model_id in enumerate(models):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    image_threshold = 500*(index+1)
    prompts = [df.iat[ind,2] for ind in range(image_threshold-500, image_threshold)]
    images = pipe(prompts).images

    for image in images:
        image.save(main_folder_path+f"fake/stable_diffusion/bedroom/{output_dir[index]}")

    with open(main_folder_path+'progress.txt','w') as file:
        file.write(f'{model_id} is printed')
