from glob import glob
from os.path import join

import monai
import torch
from monai import transforms
from monai.data import DataLoader, Dataset, NiftiSaver
from monai.inferers.utils import sliding_window_inference
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureChannelFirstD, LoadImageD,
                              OrientationD, ScaleIntensityRange,
                              ScaleIntensityRangeD, ToTensor, ToTensorD)
from monai.transforms.intensity.dictionary import NormalizeIntensityD
from monai.utils.enums import CommonKeys

BATCH_SZ = 4
device = torch.device('cuda')
channels = (8, 16, 32, 64)
strides = (1, 2, 2)
num_res_units = 0
num_epochs = 10

model_path = glob(join('.', 'TrainedModels', '*.pt'))[0]
base_image_dir = join('.', 'HCP-Interpolated')
base_label_dir = join('.', 'HCP')
image_filepaths = sorted(glob(join(base_image_dir, '**', '*.nii')))
label_filepaths = sorted(glob(join(base_label_dir, '**', '*.nii')))
split_idx = 1
validate_image_filepaths = image_filepaths[:split_idx]
validate_label_filepaths = label_filepaths[:split_idx]
keys = ['image', 'label']
train_img_label_dict = []
for imgname, labelname in zip(validate_image_filepaths, validate_label_filepaths):
    print(imgname, labelname)
    train_img_label_dict.append({keys[0]: imgname, keys[1]: labelname})

transforms_train = Compose([
    LoadImageD(keys),
    EnsureChannelFirstD(keys),
    OrientationD(keys, axcodes='RAS'),
    NormalizeIntensityD(keys, subtrahend=143.7974, divisor=87989.7634),
    ToTensorD(keys)
])

valds = Dataset(train_img_label_dict, transforms_train)
valloader = DataLoader(valds, batch_size=BATCH_SZ, shuffle=True)

model = UNet(3, 1, 1, channels=channels, strides=strides, num_res_units=num_res_units, norm=Norm.BATCH).to(device)
print(f'loading model from path {model_path}')
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

model.eval()

result_path = join('.', 'results')
iqt_saver = segsaver = NiftiSaver(result_path)


def run():
    roi_size = [96, ]*3
    sw_batch_size = 4
    for i, data in enumerate(valloader):
        image = data[CommonKeys.IMAGE].to(device)
        with torch.no_grad():
            val_out = sliding_window_inference(image, roi_size, sw_batch_size, model, device=device)
            val_out = val_out.detach()
            val_out = torch.multiply(val_out, 87989.7634)
            val_out = torch.add(val_out, 143.7974)
        iqt_saver.save_batch(val_out, data['image_meta_dict'])


run()
