import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.append("..")
import cv2
from model.diffusion import Unet
from torch.utils.data import DataLoader
from runner.diffusion import GaussianDiffusion, linear_beta_schedule
import torchvision.transforms as transforms
from dataset.DcmDataset import compute_delay_time
from dataset.DcmDataset import DcmTransformes
from dataset.DcmDataset import DcmDataset
from metrics import ssim_, psnr_
import platform


def procPetNPetDelay(pet_path, pet_delay_path):
    transform = DcmTransformes(resolution=96)
    pet = transform.preProc4pet(pet_path)
    petDelay = transform.preProc4petDelay(pet_delay_path)

    return pet.unsqueeze(0), petDelay.unsqueeze(0)


def predict(args):
    ddpm_ = GaussianDiffusion(betas=linear_beta_schedule(timesteps=1000), w=args.w, v=args.v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4, 8), self_condition=True, embDT=args.embDT).to(device)
    weight_dict = torch.load(args.weight_path)
    model.load_state_dict(weight_dict)

    pet, petDelay = procPetNPetDelay(args.pet_path, args.petDelay_path)
    pet = pet.to(device)
    petDelay = petDelay.to(device)

    # temp
    plt.figure(figsize=(1, 1))
    plt.axis("off")
    plt.imshow(petDelay.detach().squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.savefig("./petDelay.png", bbox_inches="tight", pad_inches = 0.0)
    # temp

    out = ddpm_.sample(model, image_size=(96, 96), channels=1, batch_size=1, condition=pet)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(out[-1].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.savefig(f"../assert/sample/gen_img.png")
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(petDelay.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.savefig(f"../assert/sample/petDelay.png")
    plt.close()
    '''
        for i in range(args.timeSteps):
        if i % 10 == 0 or i + 1 == args.timeSteps:
            img = out[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(img, cmap='gray')
            plt.savefig(f"../assert/sample/gen_img_{i}.png")
            plt.close()
    '''


def predict_dir(args):
    sysType = platform.system()
    if sysType == "Windows":
        gpus = [0]
    elif sysType == "Linux":
        gpus = [0, 1]

    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm_ = GaussianDiffusion(betas=linear_beta_schedule(timesteps=args.timeSteps), w=args.w, v=args.v)
    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4, 8), self_condition=True, embDT=args.embDT)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    weight_dict = torch.load(args.weight_path)
    model.load_state_dict(weight_dict)
    test_dataloader = DataLoader(
        DcmDataset(mode="test", args=args),
        batch_size = args.batch_size,
        shuffle=True,
    )
    for step, (pet, petDelay, delay_time) in enumerate(test_dataloader):
        pet = pet.cuda(non_blocking=True)
        petDelay = petDelay.cuda(non_blocking=True)
        delay_time = delay_time.cuda(non_blocking=True)
        delay_t = torch.ones(size=(1,), device=device) * delay_time
        out = ddpm_.sample(model, image_size=(96, 96), channels=1, batch_size=1, delay_time=delay_t, y=pet)
        out[-1] = torch.clip(out[-1], min=0., max=1.)
        plt.figure(figsize=(1, 1))
        fig, axes = plt.subplots(1, 2)
        axes[0].axis("off")
        axes[0].set_title("predict", pad=0)
        axes[0].imshow(out[-1].squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap="gray")

        axes[1].axis("off")
        axes[1].set_title("petDelay", pad=0)
        axes[1].imshow(petDelay.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap="gray")
        plt.savefig(f"../assert/sample/{step}.png", bbox_inches="tight")
        plt.close()
        break


def predict_4_fid_is(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = [0, 1,]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    PetDelayList = glob(f"{args.root_dir_path}/delay/*")
    PetList = glob(f"{args.root_dir_path}/first/*")
    ddpm_ = GaussianDiffusion(betas=linear_beta_schedule(timesteps=args.timeSteps), w=args.w, v=args.v)
    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4, 8), self_condition=True, embDT=args.embDT)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    weight_dict = torch.load(args.weight_path)
    model.load_state_dict(weight_dict)

    for i in range(len(PetDelayList)):
        pet_path = PetList[i]
        pet_delay_path = PetDelayList[i]
        delay_time = compute_delay_time(pet_path, pet_delay_path)
        delay_t = torch.ones(size=(1,)) * delay_time
        delay_t = delay_t.cuda(non_blocking=True)

        pet, petDelay = procPetNPetDelay(pet_path, pet_delay_path)
        pet = pet.cuda(non_blocking=True)
        petDelay = petDelay.cuda(non_blocking=True)
        out = ddpm_.sample(model, image_size=(96, 96), channels=1, batch_size=1, delay_time=delay_t, y=pet)[-1]
        out = torch.clip(out, min=0., max=1.)

        cv2.imwrite(f"{args.dir4FidNIs}/real/real_{i}.png", petDelay.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)
        cv2.imwrite(f"{args.dir4FidNIs}/gen/gen_{i}.png", out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)

        

def testing(args):
    sysType = platform.system()
    if sysType == "Windows":
        gpus = [0]
    elif sysType == "Linux":
        gpus = [0, 1]

    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm_ = GaussianDiffusion(betas=linear_beta_schedule(timesteps=args.timeSteps), w=args.w, v=args.v)
    model = Unet(dim=16, channels=1, dim_mults=(1, 2, 4, 8), self_condition=True, embDT=args.embDT)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    weight_dict = torch.load(args.weight_path)
    model.load_state_dict(weight_dict)
    test_dataloader = DataLoader(
        DcmDataset(mode="test", args=args),
        batch_size = args.batch_size,
        shuffle=True,
    )
    psnr_t = 0.
    ssim_t = 0.
    for step, (pet, petDelay, delay_time) in enumerate(test_dataloader):
        pet = pet.cuda(non_blocking=True)
        petDelay = petDelay.cuda(non_blocking=True)
        delay_time = delay_time.cuda(non_blocking=True)
        delay_t = torch.ones(size=(1,), device=device) * delay_time
        out = ddpm_.sample(model, image_size=(96, 96), channels=1, batch_size=1, delay_time=delay_t, y=pet)
        out = torch.clip(out[-1], min=0., max=1.)
        psnr_t += psnr_(out.cpu(), petDelay.cpu())
        ssim_t += ssim_(out.cpu(), petDelay.cpu())
    print(f"psnr: {psnr_t / step} | ssim: {ssim_t / step}")
    return {"psnr": psnr_t / step, "ssim": ssim_t / step}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_schedule", default=linear_beta_schedule, help="the sample function of beta")
    parser.add_argument("--loss_type", type=str, default="huber", help="loss fn type")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--timeSteps", type=int, default=300, help="forward & backward step")
    parser.add_argument("--weight_path", type=str, default="../../Dual_Time_Weights/PCP_DCM_1000.pth", help="the path of weight ")
    parser.add_argument("--pet_path", type=str,
                        default="../assert/Dicom/3/first/1.3.12.2.1107.5.1.4.59005.30000023110600580495100025326.dcm")
    parser.add_argument("--petDelay_path", type=str,
                        default="../assert/Dicom/3/delay/1.3.12.2.1107.5.1.4.59005.30000023110600580495100030464.dcm")
    parser.add_argument("--root_dir_path", type=str, default="../assert/Dicom/3")
    parser.add_argument("--test_root_path", type=str, default="../../Dual_Time_Dataset/test")
    parser.add_argument("--height", type=int, default=96, help="the height of input image")
    parser.add_argument("--dir4FidNIs", type=str, default="../assert/dir4FidNIs")
    parser.add_argument("--embDT", type=bool, default=True)
    parser.add_argument("--w", type=float, default=0.)
    parser.add_argument("--v", type=float, default=1.)
    args = parser.parse_args()
    print("batch_size:{}, timeSteps:{}, weight_path:{}, embDT:{}, w:{}. v:{}".format(
        args.batch_size, args.timeSteps, args.weight_path, args.embDT, args.w, args.v
    ))

    testing(args)
