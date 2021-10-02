import os
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import monai
import torch
from ignite.engine import (create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from ignite.utils import setup_logger
from monai import engines, transforms
from monai.data import DataLoader, Dataset, dataloader
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import StatsHandler, from_engine
from monai.networks.layers.factories import Norm
from monai.networks.nets import UNet
from monai.transforms import (Compose, CropForegroundD, EnsureChannelFirstD,
                              EnsureTypeD, LoadImageD, OrientationD,
                              ScaleIntensityD)
from monai.transforms.croppad.dictionary import RandSpatialCropD
from monai.transforms.intensity.dictionary import (NormalizeIntensityD,
                                                   ScaleIntensityRangeD)
from monai.transforms.utility.dictionary import ToTensorD
from monai.utils.enums import CommonKeys
from monai.utils.misc import first, set_determinism
from torch.nn import MSELoss
from torch.optim import Adam

set_determinism(8127301)
BATCH_SZ = 4
lr = 1e-3
device = torch.device('cuda')
channels = (8, 16, 32, 64)
strides = (1, 2, 2)
num_res_units = 0
num_epochs = 20000
step = 1

base_image_dir = join('.', 'HCP-Interpolated')
base_label_dir = join('.', 'HCP')
image_filepaths = sorted(glob(join(base_image_dir, '**', '*.nii')))
label_filepaths = sorted(glob(join(base_label_dir, '**', '*.nii')))

split_idx = 1
train_image_filepaths = image_filepaths[split_idx:]
train_label_filepaths = label_filepaths[split_idx:]

validate_image_filepaths = image_filepaths[:split_idx]
validate_label_filepaths = label_filepaths[:split_idx]

keys = ['image', 'label']
train_img_label_dict = []
for imgname, labelname in zip(train_image_filepaths, train_label_filepaths):
    print(imgname, labelname)
    train_img_label_dict.append({keys[0]: imgname, keys[1]: labelname})

validate_img_label_dict = []
for imgname, labelname in zip(validate_image_filepaths, validate_label_filepaths):
    print(imgname, labelname)
    train_img_label_dict.append({keys[0]: imgname, keys[1]: labelname})

transforms_train = Compose([
    LoadImageD(keys),
    EnsureChannelFirstD(keys),
    OrientationD(keys, axcodes='RAS'),
    NormalizeIntensityD(keys, subtrahend=143.7974, divisor=87989.7634),
    CropForegroundD(keys, source_key=keys[0]),
    RandSpatialCropD(keys, roi_size=[96, ]*3, random_size=False),
    ToTensorD(keys)
])

trainds = Dataset(train_img_label_dict, transforms_train)
trainloader = DataLoader(trainds, batch_size=BATCH_SZ, shuffle=True)

valds = Dataset(validate_img_label_dict, transforms_train)
valloader = DataLoader(trainds, batch_size=BATCH_SZ, shuffle=True)
# for data in trainds:
#     print(data['image'].shape, data['label'].shape)
#     plt.subplot(1, 2, 1)
#     plt.imshow(data['image'][0, 40], cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(data['label'][0, 40], cmap='gray')
#     plt.show()
#     plt.tight_layout()

model = UNet(3, 1, 1, channels=channels, strides=strides, num_res_units=num_res_units, norm=Norm.BATCH).to(device)
loss_function = MSELoss()
optimizer = Adam(model.parameters(), lr)


# test the workflow

batch = first(trainloader)
image = batch[keys[0]].to(device)
label = batch[keys[1]].to(device)
predicted_label = model(image)
mse_error = loss_function(predicted_label, label)
print(f'MSE {mse_error.item():.3f}')

iter_loss_values = list()
val_loss_values = list()


def prep_batch(batch, device, non_blocking):
    return batch[CommonKeys.IMAGE].to(device), batch[CommonKeys.LABEL].to(device)


val_metrics = {
    "mse": Loss(loss_function, device=device)
}
trainer = create_supervised_trainer(model, optimizer, loss_function, device, prepare_batch=prep_batch)
trainer.logger = setup_logger('trainer')
evaluator = create_supervised_evaluator(model, val_metrics, device, True, prepare_batch=prep_batch)
evaluator.logger = setup_logger('evaluator')


@trainer.on(Events.ITERATION_COMPLETED)
def log_iteration_loss(engine: Engine):
    global step
    loss = engine.state.output
    iter_loss_values.append(loss)
    print(f'epoch {engine.state.epoch}/ {engine.state.max_epochs} step {step} Training MSE {loss:.3f}')
    step += 1


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation(engine: Engine):
    evaluator.run(valloader)
    val_loss_values.append(evaluator.state.metrics['mse'])
    print(f'epoch {engine.state.epoch} Validation MSE {val_loss_values[-1]:.3f}')


checkpoint_handler = ModelCheckpoint('./TrainedModels', filename_prefix='Unet_MSE_batchnorm', score_name='mse',
                                     n_saved=1,
                                     require_empty=False, score_function=lambda x: -iter_loss_values[-1])

trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={'model': model})
trainer.run(trainloader, num_epochs)
