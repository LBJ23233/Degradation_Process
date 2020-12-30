import os
import glob
import scipy.io as sio
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import torch
import matlab.engine

from imresize import imresize
from util import calculate_psnr, calculate_ssim, read_image, save_image


### degradation code from Correction fliter
def downsample_using_h(I_in, Filter, scale, pad=None):
    I_in = torch.from_numpy(I_in.transpose((2, 0, 1))).float()
    Filter = torch.from_numpy(Filter).float()
    if len(Filter.shape) < 4:
        Filter = Filter.unsqueeze(0).unsqueeze(0).cuda()
        I_in = I_in.unsqueeze(0).cuda()
    filter_supp_x = Filter.shape[3]
    filter_supp_y = Filter.shape[2]
    h = (Filter) / torch.sum(Filter)
    if pad:
        pad_x = np.int((filter_supp_x - scale)/2)
        pad_y = np.int((filter_supp_y - scale)/2)
        I_padded = torch.nn.functional.pad(I_in, [pad_x, pad_x, pad_y, pad_y], mode=pad)
    else:
        pad_x = scale//2
        pad_y = scale//2
        I_padded = torch.nn.functional.pad(I_in, [pad_x, pad_x, pad_y, pad_y], mode='constant')
    batch, c, H, W = I_padded.shape
    I_out = torch.nn.functional.conv2d(I_padded.view(-1, 1, H, W), h, stride=scale)
    return I_out.view(batch, c, I_out.shape[2], I_out.shape[3]).squeeze().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)


# degradation process for kernel solver (By tang)
def downsample_averagex2(im, kernel):
    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im).astype(np.float)
    # for channel in range(np.ndim(im)):
    for channel in range(im.shape[2]):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    out_lr = out_im[::2, ::2, :] + out_im[1::2, ::2, :] + out_im[::2, 1::2, :] + out_im[1::2, 1::2, :]
    out_lr = out_lr / 4
    return out_lr.astype(np.uint8)


"""
Bicubic vs. Gaussian blur + bicubic vs. KernelGAN vs. CF vs. Kernel solver
"""
os.environ['QT_QPA_PLATFORM']='offscreen'
scale = 2
#### Path setting
# /home/yangyuqiang/tmp/KernelGAN/results/RealSR_V3_x2
# /home/yangyuqiang/tmp/Correction_filter/results/RealSRv3/Filters_mat
kernel_path = '/home/yangyuqiang/tmp/KernelGAN/results/RealSR_V3_x2'
cf_kernel_path = '/home/yangyuqiang/tmp/Correction_filter/results/RealSRv3/Filters_mat'
hr_path = '/home/yangyuqiang/datasets/RealSR/RealSR_V3_train_HR_x2sub'
gt_lr_path = '/home/yangyuqiang/datasets/RealSR/RealSR_V3_train_LR_x2sub'
syn_lr_path = '/home/yangyuqiang/datasets/RealSR/RealSR_V3_train_synthesis_x2sub_2/LRblur/x2'
bic_lr_path = '/home/yangyuqiang/datasets/RealSR/RealSR_V3_train_synthesis_x2sub_2/LR/x2'

save_path = './results/degradation_5'
save_kernel_path = './results/Tang_RealSR_V3/filter_mat'
save_kernel_img = './results/Tang_RealSR_V3/filter_img'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_kernel_path):
    os.makedirs(save_kernel_path)
if not os.path.exists(save_kernel_img):
    os.makedirs(save_kernel_img)

kernel_list = glob.glob(os.path.join(kernel_path, '*x{}.mat'.format(scale)))
# kernel_list = sorted(kernel_list)
# cf_kernel_list = glob.glob(os.path.join(cf_kernel_path, '*x{}.mat'.format(scale)))

num = len(kernel_list)
#### log
log_file = open(save_path + '/log_{}_tang.txt'.format(num), 'w')
log_file.write('-' * 10 + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\t-------RGB channel\n\n')
#### start matlab
eng = matlab.engine.start_matlab()

sig = None
cnt = 0
log_out = ''
b_avg_psnr, b_avg_ssim = 0.0, 0.0
g_avg_psnr, g_avg_ssim = 0.0, 0.0
c_avg_psnr, c_avg_ssim = 0.0, 0.0
k_avg_psnr, k_avg_ssim = 0.0, 0.0
t_avg_psnr, t_avg_ssim = 0.0, 0.0
tang_lr_patch_psnr = 0.0
for i in range(num):
    #### cf_kernel_name: 'Nikon_206_LR2_s009_x2_corr_kernel_x2.mat'
    #### kg_kernel_name: 'Nikon_206_LR2_s009_kernel_x2.mat'
    # path = cf_kernel_list[i]
    # cf_kernel = sio.loadmat(path)['Kernel']
    # kg_kernel_name = path.split('/')[-1].replace('x2_corr_kernel_x2', 'kernel_x2')
    # path = str(os.path.join(kernel_path, kg_kernel_name))
    # if os.path.exists(path): cnt += 1
    # else: continue
    cnt = i + 1
    path = kernel_list[i]
    kernel = sio.loadmat(path)['Kernel']
    #### create file name
    kernel_name = path.split('/')[-1]
    gt_lr_name = path.split('/')[-1][:-14] + '.png'  # remove _kernel_x2.mat
    hr_name = gt_lr_name.replace('LR2', 'HR')  #  'LR2' in RealSR's LR image, 'LR' in synthesis one
    bic_lr_name = hr_name.replace('HR', 'LR')
    syn_lr_name = glob.glob(os.path.join(syn_lr_path, '*' + bic_lr_name))[0]
    sig = syn_lr_name.split('/')[-1][:3]   # get the sigma of randomly Iso_Gauss
    #### load images
    gt_lr = read_image(os.path.join(gt_lr_path, gt_lr_name))
    hr = read_image(os.path.join(hr_path, hr_name))
    bic_lr = read_image(os.path.join(bic_lr_path, bic_lr_name))
    syn_lr = read_image(syn_lr_name)

    #### degrade HR with the kernel from  **** KernelGAN
    # ker_lr = imresize(hr, scale_factor=1/scale, kernel=kernel)

    #### degrade HR with the kernel from  **** Correction filter
    # cf_lr = imresize(hr, scale_factor=1/scale, kernel=cf_kernel)
    # cf_lr = downsample_using_h(hr, Filter=cf_kernel, scale=scale, pad='reflect')
    #### visualize
    # plt.imshow(cf_kernel)
    # plt.colorbar()
    # plt.savefig(os.path.join(save_kernel_img, kernel_name[:-4] + '_cf1.png'))
    # cf_kernel = cf_kernel[24:40, 24:40]
    # plt.imshow(cf_kernel)
    # plt.savefig(os.path.join(save_kernel_img, kernel_name[:-4] + '_cf2.png'))
    # plt.close()

    #### degrade HR with the kernel from  **** Tang_kernel_solver
    hr_in_path, lr_in_path = os.path.join(hr_path, hr_name), os.path.join(gt_lr_path, gt_lr_name)
    tang_kernel = eng.kernel(hr_in_path, lr_in_path)
    tang_kernel = np.array(tang_kernel)
    shape = tang_kernel.shape
    tang_lr = np.zeros_like(gt_lr)
    lr_size = gt_lr.shape[0:2]
    idx = -1
    if len(shape) > 2:  #### multi-patch degradation
        crop_size = (lr_size // np.sqrt(shape[0])).astype(np.int)  # 4/16 patches: LR: 128 / 64; HR: 256/128
        mse = 0.0
        for x in range(int(np.sqrt(shape[0]))):
            for y in range(int(np.sqrt(shape[0]))):
                idx += 1
                tang_kernel_i = tang_kernel[idx, :, :]
                #### crop LR patch and degrade corresponding HR patch
                tang_lr[x * crop_size[0]:x * crop_size[0] + crop_size[0],
                y * crop_size[1]:y * crop_size[1] + crop_size[1], :] = downsample_averagex2(
                    hr[x * crop_size[0] * scale:x * crop_size[0] * scale + crop_size[0] * scale,
                    y * crop_size[1] * scale:y * crop_size[1] * scale + crop_size[1] * scale, :], kernel=tang_kernel_i)
                #### calculate mse of patches
                img1 = tang_lr[x * crop_size[0]:x * crop_size[0] + crop_size[0],
                       y * crop_size[1]:y * crop_size[1] + crop_size[1], :].astype(np.float64)
                img2 = gt_lr[x * crop_size[0]:x * crop_size[0] + crop_size[0],
                       y * crop_size[1]:y * crop_size[1] + crop_size[1], :].astype(np.float64)
                mse += np.mean((img1 - img2) ** 2)
        tang_lr_patch_psnr += 20 * math.log10(255.0 / math.sqrt(mse/shape[0]))
    else:
        tang_lr = downsample_averagex2(hr, kernel=tang_kernel)
    #### visualize results
    # img_name = 'test3_16_' + gt_lr_name.replace('LR2', 'LRtang_{}'.format(idx))
    # save_image(os.path.join(save_path, img_name), tang_lr)
    log_out = '\n{}:\tTang({}): {:.4f}'.format(
        gt_lr_name, cnt, tang_lr_patch_psnr/cnt
    )
    log_file.write(log_out)
    print(log_out)
    # print(sum(sum(tang_kernel)))
    #### save results and kernel visualization
    # if not os.path.exists(os.path.join(save_kernel_path, kernel_name)):
    sio.savemat(os.path.join(save_kernel_path, kernel_name), {'Kernel': tang_kernel})
    plt.imshow(tang_kernel)
    plt.colorbar()
    plt.savefig(os.path.join(save_kernel_img, kernel_name[:-4]+'.png'))
    plt.close()
    # if cnt == 3000: break
    continue

    #### quantify the performance of different methods
    # only bicubic
    b_psnr = calculate_psnr(gt_lr, bic_lr)
    b_ssim = calculate_ssim(gt_lr, bic_lr)
    b_avg_psnr += b_psnr
    b_avg_ssim += b_ssim
    # Gaussian kernel(0.2-2) + bic
    g_psnr = calculate_psnr(gt_lr, syn_lr)
    g_ssim = calculate_ssim(gt_lr, syn_lr)
    g_avg_psnr += g_psnr
    g_avg_ssim += g_ssim
    # Correction filter
    c_psnr = calculate_psnr(gt_lr, cf_lr)
    c_ssim = calculate_ssim(gt_lr, cf_lr)
    c_avg_psnr += c_psnr
    c_avg_ssim += c_ssim
    # KernelGAN
    k_psnr = calculate_psnr(gt_lr, ker_lr)
    k_ssim = calculate_ssim(gt_lr, ker_lr)
    k_avg_psnr += k_psnr
    k_avg_ssim += k_ssim
    # Tang_kernel_solver
    t_psnr = calculate_psnr(gt_lr, tang_lr)
    t_ssim = calculate_ssim(gt_lr, tang_lr)
    t_avg_psnr += t_psnr
    t_avg_ssim += t_ssim

    #### save estimated LR and print log
    # if i >= 0:
    #     error = k_psnr - b_psnr
    #     error = str(error)[:5]
    #     img_name = error + gt_lr_name.replace('LR2', 'LRreal')
    #     save_image(os.path.join(save_path, img_name), gt_lr)
    #     img_name = error + gt_lr_name.replace('LR2', 'LRbic')
    #     save_image(os.path.join(save_path, img_name), bic_lr)
    #     img_name = error + gt_lr_name.replace('LR2', 'LRsyn{}'.format(sig))
    #     save_image(os.path.join(save_path, img_name), syn_lr)
    #     img_name = error + gt_lr_name.replace('LR2', 'LRcf')
    #     save_image(os.path.join(save_path, img_name), cf_lr)
    #     img_name = error + gt_lr_name.replace('LR2', 'LRker')
    #     save_image(os.path.join(save_path, img_name), ker_lr)
    #     img_name = error + gt_lr_name.replace('LR2', 'LRtang')
    #     save_image(os.path.join(save_path, img_name), tang_lr)

    log_out = '\n{}:\n\tBic: {:.4f}/{:.4f}\tGauss{}: {:.4f}/{:.4f}\tCF: {:.4f}/{:.4f}\tKG: {:.4f}/{:.4f}\tTang: {:.4f}/{:.4f}'.format(
        gt_lr_name, b_psnr, b_ssim, sig, g_psnr, g_ssim, c_psnr, c_ssim, k_psnr, k_ssim, t_psnr, t_ssim
    )
    log_file.write(log_out)
    print(log_out)
eng.quit()
log_out = '\nAverage {}:\n\tBicubic: {:.4f}/{:.4f}\tGauss(0.2-2.0): {:.4f}/{:.4f}\tCF: {:.4f}/{:.4f}\tKG: {:.4f}/{:.4f}\tTang: {:.4f}/{:.4f}'.format(
    cnt, b_avg_psnr / cnt, b_avg_ssim / cnt, g_avg_psnr / cnt, g_avg_ssim / cnt, c_avg_psnr / cnt, c_avg_ssim / cnt,
         k_avg_psnr / cnt, k_avg_ssim / cnt, t_avg_psnr / cnt, t_avg_ssim / cnt)
log_file.write(log_out)
print(log_out)





