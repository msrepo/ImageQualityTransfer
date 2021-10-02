import os
from glob import glob
from os.path import join

import numpy as np
import SimpleITK as sitk

base_dir = '/media/HDD1/mahesh/repo_clones/temp/IQT'
data_dir = join(base_dir, 'HCP')
out_dir = join(base_dir, 'HCP-preprocessed')
interpolated_dir = join(base_dir, 'HCP-Interpolated')

os.makedirs(out_dir, exist_ok=True)
os.makedirs(interpolated_dir, exist_ok=True)
root, dirs, files = next(os.walk(data_dir))
# print(root, dirs, files)


def downsample(img: sitk.Image, downsample_ratio, verbose=True):
    '''
    downsample by downsample_ratio along the axial direction
    '''

    img_spacing = img.GetSpacing()
    img_size = img.GetSize()

    downsample_ratio = 8
    new_spacing = list(img_spacing)
    new_spacing[2] *= downsample_ratio

    new_size = list(img_size)
    new_size[2] /= downsample_ratio
    new_size[2] = int(new_size[2])

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkGaussian)

    out_img: sitk.Image = resampler.Execute(img)

    if verbose:
        print(filepath)
        print(f'Original Size {img.GetSize()} Spacing {img.GetSpacing()} ')
        print(f'New size {out_img.GetSize()}, Spacing{out_img.GetSpacing()}')
    return out_img


means = []
stds = []
for dir in dirs:

    os.makedirs(join(root, dir), exist_ok=True)
    filepath = glob(join(root, dir, '*.nii'))
    filename = filepath[0].split('/')[-1]
    # print(filepath)
    img = sitk.ReadImage(filepath[0])

    out_img = img
    out_img = downsample(out_img, 8.0)
    interpolated_img: sitk.Image = sitk.Expand(out_img, [1, 1, 8], sitk.sitkNearestNeighbor)

    imagestatsf = sitk.StatisticsImageFilter()
    imagestatsf.Execute(img)
    print(f'Original min {imagestatsf.GetMinimum()} max {imagestatsf.GetMaximum()}'
          f'mean {imagestatsf.GetMean()} std {imagestatsf.GetVariance()}')
    imagestatsf.Execute(out_img)
    print(f'downscaled min {imagestatsf.GetMinimum()} max {imagestatsf.GetMaximum()}'
          f'mean {imagestatsf.GetMean()} std {imagestatsf.GetVariance()}')
    imagestatsf.Execute(interpolated_img)
    print(f'rescaled min {imagestatsf.GetMinimum()} max {imagestatsf.GetMaximum()}'
          f'mean {imagestatsf.GetMean()} std {imagestatsf.GetVariance()}')
    means.append(imagestatsf.GetMean())
    stds.append(imagestatsf.GetVariance())
    # write image
    os.makedirs(join(out_dir, dir), exist_ok=True)
    sitk.WriteImage(out_img, join(out_dir, dir, filename))

    os.makedirs(join(interpolated_dir, dir), exist_ok=True)
    sitk.WriteImage(interpolated_img, join(interpolated_dir, dir, filename))

print('means', means)
print('variances', stds)
print('Dataset mean', np.mean(means))
print('Dataset std', np.mean(stds))
