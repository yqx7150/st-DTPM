import torch
import torch.nn.functional as F

def linear_bate_schedual(timeSteps):
    beta_start = 0.00001
    beta_end = 0.002
    return torch.linspace(beta_start, beta_end, timeSteps)

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

class Diffusion:
    def __init__(self, betas, loss_type, timeSteps):
        self.loss_type = loss_type
        self.timeSteps = timeSteps
        self.betas = betas
        self.alpha = 1 - betas
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.one_minus_alpha_cumprod = 1. - self.alpha_cumprod
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(self.one_minus_alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1. / self.alpha)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alpha_cumprod_prev) / self.one_minus_alpha_cumprod


    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            z = torch.randn_like(x_start)
        else:
            z = noise

        sqrt_alpha_cumprod = extract(self.sqrt_alpha_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * z
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y=None, delay_time=None):
        betas = extract(self.betas, t, x.shape)
        sqrt_recip_alpha = extract(self.sqrt_recip_alpha, t, x.shape)
        sqrt_one_minus_alpha_cumprod = extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)
        model_mean = sqrt_recip_alpha * (x - betas * model(x, t, delay_time, y) / sqrt_one_minus_alpha_cumprod)
        posterior_variance = extract(self.posterior_variance, t, x.shape)

        if t_index == 0:
            return model_mean
        else:
            z = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * z
        
    @torch.no_grad()
    def sample(self, model, shape, y=None, delay_time=None):
        device = next(model.parameters()).device
        B = shape[0]
        img = torch.randn(shape, device=device)
        imgs = []
        for i in reversed(range(0, self.timeSteps)):
            img = self.p_sample(model, img, torch.full((B, ), i , device=device, dtype=torch.long), i, y, delay_time)
            imgs.append(img)
        return imgs
    
    def p_losses(self, model, x_start, t, y=None, delay_time=None, noise=None):
        if noise is None:
            z = torch.randn_like(x_start)
        else:
            z = noise

        x_noise = self.q_sample(x_start, t, noise=z)
        predict_noise = model(x_noise, t, delay_time, y)

        if self.loss_type == "l1":
            loss = F.l1_loss(z, predict_noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(z, predict_noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(z, predict_noise)
        else:
            raise NotImplementedError()
        
        return loss

        
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


if __name__ == "__main__":
    import cv2
    import sys
    sys.path.append("..")
    from torch.utils.data import DataLoader
    from dataset.DcmDataset import DcmDataset, DcmTransforms

    DcmTransform = DcmTransforms(resolution=96)
    full_dataset = DcmDataset(datasetRootPath="../../Dual_Time_Dataset/PCP", transforms=DcmTransform)
    temp_dataloader = DataLoader(
        dataset=full_dataset,
        batch_size=1,
        shuffle=True,
    )

    timeSteps = 300
    betas = linear_bate_schedual(timeSteps)
    Diff = Diffusion(betas=betas, loss_type="l2", timeSteps=timeSteps)

    for batch in temp_dataloader:
        pet = batch["pet"]
        petDelay = batch["petDelay"]
        B = pet.shape[0]
        for i in range(timeSteps):
            pet_noise = Diff.q_sample(x_start=petDelay, t=torch.full((B,), fill_value=i, dtype=torch.long))
            cv2.imwrite(f"./temp/petDelay_noise{i}.png", DcmTransform.reverse_transform(pet_noise))
        break
    cv2.imwrite(f"./temp/pet.png", DcmTransform.reverse_transform(pet))

    cv2.imwrite(f"./temp/GuN.png", torch.randn(96, 96, 1).numpy()*255)

