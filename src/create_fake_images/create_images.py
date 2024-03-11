import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

output_dir = ["sd_2_1","sd_xl_base","sd_1"]

models = ["stabilityai/stable-diffusion-2-1",
          "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5"]

main_folder_path = "/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/"
df = pd.read_csv(main_folder_path+'data/captions/lsun/bedroom/unique_captions_indices.csv')


# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead

pipe = StableDiffusionPipeline.from_pretrained(models[1], torch_dtype=torch.float16, )
# pipe.scheduler   # change the scheduler. #DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

for prompt in df.itertuples():
    image = pipe(prompt[3], num_inference_steps=50).images[0]
    image.save(main_folder_path+f"data/fake/stable_diffusion/bedroom/{output_dir[1]}/{prompt[2]}")
    print(prompt[1])

with open(main_folder_path+'progress.txt', 'w') as file:
    file.write(f'{models[1]} is printed.')
