import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image

# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
checkpoint = "HuggingFaceM4/idefics-9b"
processor = AutoProcessor.from_pretrained(checkpoint)

def make_prompts(images_path):
    url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
    img_example = processor.image_processor.fetch_images([url])[0]

    prompt_example = ['User: '+img_example + '\n Instruction: Describe this image. \n Answer: An image of two kittens.']
    # prompts = [prompt_example]

    prompts = [['User: '+img_example + '\n Instruction: Describe this image. \n Answer: An image of two kittens.'],
               ['User: '+Image.open(images_path) + '\n Instruction:  Describe this image in detail. Mention color of all the objects in the image.'
                '\n Answer: ']]

    # for image in images_path:
    #     prompts.append([
    #     'User: '+Image.open(image)+'\n Instruction:  Describe this image in detail. Mention color of all the objects in the image.'
    #                                '\n Answer: '
    #     ])
    return prompts


def create_captions():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # checkpoint = "HuggingFaceM4/idefics-9b"
    # model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
    # processor = AutoProcessor.from_pretrained(checkpoint)
    model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b").to(device)

    main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/'
    image_folder = main_folder + 'data/real/lsun/bedroom/'
    image_paths = [image_folder + "1.jpg"] #, image_folder + "2.jpg", image_folder + "3.jpg"]

    prompts = make_prompts(image_paths)
    # --batched mode

    inputs = processor(prompts, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_length=200)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

    return generated_text
    # inputs = processor(prompts, return_tensors="pt").to(device)

    # Generation args
    # bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    # generated_ids = model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=100)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # for i, t in enumerate(generated_text):
    #     print(f"{i}:\n{t}\n")
    # return generated_text

create_captions()