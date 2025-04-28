import torch
import torch.nn.functional as F
import math
from skimage.metrics import structural_similarity as _SSIM

@torch.no_grad()
def mse_(first: torch.Tensor, delay: torch.Tensor):
    return F.mse_loss(first.detach(), delay.detach()).item()

@torch.no_grad()
def psnr_(first: torch.Tensor, delay: torch.Tensor):
    mse = mse_(first.detach(), delay.detach())
    if mse == 0:
        return 100
    else:
        return 20 * math.log10(1. / math.sqrt(mse))

@torch.no_grad()
def ssim_(first: torch.Tensor, delay: torch.Tensor):
    B = first.detach().shape[0]
    ssim = 0.
    for i in range(B):
        fz = first.detach().permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :]
        fx = delay.detach().permute(0, 2, 3, 1).cpu().numpy()[i, :, :, :]
        ssim_one_image = _SSIM(fz, fx, multichannel=True)
        ssim += ssim_one_image
    ssim /= B
    return ssim


from pytorch_fid import fid_score

def fid_score_(real_image_folder, generated_image_folder, device):

    fid_value = fid_score.calculate_fid_given_paths([real_image_folder, generated_image_folder],
                                                    batch_size=1,
                                                    device=device,
                                                    dims=2048,
                                                    )
    return fid_value
