from torchvision.models import resnet50, ResNet50_Weights
import torch


def retrieve_images_fake(path,transform):
    images_raw_dict=[]  # later save embeddings instead of raw
    for prompt_folder in os.listdir(path):
        # print('folder_name ', prompt_folder)
        for image_file in os.listdir(path+'/'+prompt_folder):
            # print('image_name ', image_file)
            image_name = path+'/'+prompt_folder+'/'+image_file
            image_opened = Image.open(image_name)
            image = transform(image=image_opened)['image']
            image_dict = {'image':image,
                          'image_name':image_name, 'type':0}
            images_raw_dict.append(image_dict)
            image_opened=''
    return images_raw_dict

def retrieve_images_real(path,transform):
    images_raw_dict=[]  # later save embeddings instead of raw
    for prompt_folder in os.listdir(path):
        # print('folder_name ', prompt_folder)
        if not os.path.isdir(path+'/'+prompt_folder):
            continue
        for image_folder in os.listdir(path+'/'+prompt_folder):
            if not os.path.isdir(path+'/'+prompt_folder+'/'+image_folder):
                continue
            
            for image_file in os.listdir(path+'/'+prompt_folder+'/'+image_folder):
                if not image_file.__contains__("jpg"):
                    continue
                image_name = path+'/'+prompt_folder+'/'+image_folder+'/'+image_file
                image_opened = Image.open(image_name)
                image = transform(image=image_opened)['image']
                image_dict = {'image':image,
                              'image_name':image_name, 'type':1}
                images_raw_dict.append(image_dict)
                image_opened=''
    return images_raw_dict

def create_pre_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
    ])

def define_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(2048, config['model']['num-classes'])