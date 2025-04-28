import torch
import torch.nn as nn
import argparse
import math
import sys
import cv2
import logging
import math
import os
import numpy as np

sys.path.append("..")

from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import random_split

from model.UnetTrans import Unet, EmbDtMode, TransEmbDtMode
from diffusion_ import Diffusion, sigmoid_beta_schedule, linear_bate_schedual
from dataset.DcmDataset import DcmDataset, DcmTransforms, DcmDatasetFixed
from metrics import psnr_, ssim_, fid_score_, mse_

import platform
import logging

from test_data import DcmDatasetTest

def main(args):
    # build platform
    sysType = platform.system()
    if sysType == "Windows":
        gpus = [0]
    elif sysType == "Linux":

        if(args.deviceType == "both"):
            gpus = [0, 1]
        elif(args.deviceType == "device0"):
            gpus = [0]
        elif(args.deviceType == "device1"):
            gpus = [1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    DcmTransform = DcmTransforms(resolution=args.height)
    full_dataset = DcmDataset(datasetRootPath=args.dataset_root_path, transforms=DcmTransform)
    train_len = int(len(full_dataset) * 0.8)
    test_len = len(full_dataset) - train_len
    train_dataset, test_dataset = random_split(dataset=full_dataset, lengths=[train_len, test_len], generator=torch.Generator().manual_seed(7))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_dataset = DcmDatasetTest(datasetRootPath=args.dataset_root_path, transforms=DcmTransform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    # build diffusion
    betas = linear_bate_schedual(args.timeSteps)
    ddpm_ = Diffusion(betas=betas, loss_type=args.loss_type, timeSteps=args.timeSteps)

    # build model
    if args.embDTMode == 1:
        emb_dt_mode = EmbDtMode.EACH_BLOCK
    elif args.embDTMode == 2:
        emb_dt_mode = EmbDtMode.LINEAR_CAT
    elif args.embDTMode == 3:
        emb_dt_mode = EmbDtMode.ADD
    elif args.embDTMode == 4:
        emb_dt_mode = EmbDtMode.LINEAR_ADD

    if args.transEmbDTMode == 1:
        trans_emb_dt_mode = TransEmbDtMode.EACH_BLOCK
    elif args.transEmbDTMode == 2:
        trans_emb_dt_mode = TransEmbDtMode.LINEAR_CAT
    elif args.transEmbDTMode == 3:
        trans_emb_dt_mode = TransEmbDtMode.ADD
    elif args.transEmbDTMode == 4:
        trans_emb_dt_mode = TransEmbDtMode.LINEAR_ADD


    model = Unet(dim=args.model_dim, 
                dim_mults=args.dim_mults,
                channels=args.channels,
                embDTMode=emb_dt_mode,
                transEmbDTMode=trans_emb_dt_mode,
                self_condition=args.condition,
                embDT=args.embDT)
    # model = Unet(dim=args.model_dim, channels=args.channels, dim_mults=args.dim_mults, self_condition=args.condition, embDT=args.embDT)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    # build optimizer
    optimizer_ = optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.runType == "train":
        print("runType:{}| n_epochs:{}| weight_path:{}| batch_size:{}| timeSteps:{}| dim:{}| condition:{}| embDT:{}| device:{}".format(
            args.runType, args.n_epochs, args.weight_save_path, args.batch_size, args.timeSteps, args.model_dim, args.condition, args.embDT, args.deviceType))
        logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.logging_name, filemode="w")
        for epoch in range(args.epoch, args.n_epochs):
            loss_totle = 0.
            for step, batch in enumerate(train_dataloader):
                optimizer_.zero_grad()
                pet = batch["pet"].cuda(non_blocking=True)
                petDelay = batch["petDelay"].cuda(non_blocking=True)
                delay_time = batch["delay_time"].cuda(non_blocking=True)
                batch_size = pet.shape[0]
                t = torch.randint(0, args.timeSteps, (batch_size,), device=device).long()
                delay_t = torch.ones(size=(batch_size,), device=device) * delay_time

                loss = ddpm_.p_losses(model, petDelay, t, y=pet, delay_time=delay_t)

                loss.backward()
                optimizer_.step()

                loss_totle += loss.item()

                sys.stdout.write(
                    "\r[Epoch {}/{}] [Batch {}/{}] [loss {:.4f}]".format(
                        epoch,
                        args.n_epochs,
                        step,
                        len(train_dataloader),
                        loss.item(),
                    )
                )
            
            logging.info("Epoch: {} | loss: {:.4f} | psnr: {:.4f}".format(
                epoch,
                loss_totle / len(train_dataloader),
                20 * math.log10(1. / (math.sqrt(loss_totle / len(train_dataloader)))),
                
            ))
        torch.save(model.state_dict(), args.weight_save_path)

    elif args.runType == "test":
        all_image = []
        all_label = []
        if not os.path.exists(args.sample_name):
            os.makedirs(f"{args.sample_name}/pet")
            os.makedirs(f"{args.sample_name}/gen")
            os.makedirs(f"{args.sample_name}/real")
        logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.test_logging_name, filemode="w")
        print("runType:{}| weight_path:{}| batch_size:{}| timeSteps:{}| condition:{}| embDT:{}".format(
            args.runType, args.weight_path, args.batch_size, args.timeSteps, args.condition, args.embDT))
        # load weight
        weight_dict = torch.load(args.weight_path)
        model.load_state_dict(weight_dict)

        psnr_totle = 0.
        ssim_totle = 0.
        curr_num = 0
        for step, batch in enumerate(test_dataloader):
            pet = batch["pet"].cuda(non_blocking=True)
            petDelay = batch["petDelay"]
            delay_time = batch["delay_time"].cuda(non_blocking=True)
            SUVgap = batch["SUVgap"]
            B = delay_time.shape[0]
            delay_t = torch.ones(size=(B,), device=device) * delay_time
            out = ddpm_.sample(model, shape=pet.shape, y=pet, delay_time=delay_t)[-1].cpu()
            all_image.append(out.permute(0, 2, 3, 1).detach().cpu().numpy())
            all_label.append(petDelay.permute(0, 2, 3, 1).detach().cpu().numpy())
            cv2.imwrite(f"{args.sample_name}/pet/{step}_pet.png", DcmTransform.reverse_transform(pet.cpu()))
            cv2.imwrite(f"{args.sample_name}/gen/{step}_gen.png", DcmTransform.reverse_transform(out))
            cv2.imwrite(f"{args.sample_name}/real/{step}_real.png", DcmTransform.reverse_transform(petDelay))
            psnr_batch = psnr_(out, petDelay)
            ssim_batch = ssim_(out, petDelay)
            psnr_totle += psnr_batch
            ssim_totle += ssim_batch

            beforeSUVmax = float(SUVgap)
            afterSUVmax = float(torch.max(out.detach()).item() * float(SUVgap))
            if beforeSUVmax < 2.5 and afterSUVmax < 2.5:
                curr_num += 1
            elif beforeSUVmax >= 2.5 and afterSUVmax >= 2.5:
                curr_num += 1
            else:
                pass

            # add logging pre image
            imagesLoggingPath = args.imagesLossingPath
            file_bin = open(imagesLoggingPath, "a")
            file_bin.write(f"{step},{psnr_batch},{ssim_batch}\n")
            file_bin.close()
            sys.stdout.write(
                    "\r[Batch {}/{}] [psnr {:.4f}] [ssim {:.4f}]".format(
                        step,
                        len(test_dataloader),
                        psnr_batch,
                        ssim_batch,
                    )
                )
            logging.info("{:.4f},{:.4f},{:.4f},{:.4f}".format(psnr_batch, ssim_batch, beforeSUVmax, afterSUVmax))
        # fid = fid_score_( real_image_folder=f"{args.sample_name}/real",
        #                    generated_image_folder=f"{args.sample_name}/gen",
        #                    device=device)
        images = np.concatenate(all_image, axis=0)
        np.save(f"{args.sample_name}/gen.npy", images)
        labels = np.concatenate(all_label, axis=0)
        np.save(f"{args.sample_name}/gt.npy", labels)
        fid = 0.
        print(f"psnr: {psnr_totle / step} | ssim: {ssim_totle / step} | fid: {fid}")
        logging.info(f"psnr: {psnr_totle / step} | ssim: {ssim_totle / step} | fid: {fid}")
        print(f"SUVmax curr rate: {100 * curr_num / (step + 1):.4f}%")
        return {"psnr": psnr_totle / step, "ssim": ssim_totle / step, "fid": fid}

    elif args.runType == "test_time_emb":
        if not os.path.exists(args.sample_name):
            os.makedirs(f"{args.sample_name}/pet")
            os.makedirs(f"{args.sample_name}/gen")
            os.makedirs(f"{args.sample_name}/real")
        logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.test_logging_name, filemode="w")
        print("runType:{}| weight_path:{}| batch_size:{}| timeSteps:{}| condition:{}| embDT:{}".format(
            args.runType, args.weight_path, args.batch_size, args.timeSteps, args.condition, args.embDT))
        # load weight
        weight_dict = torch.load(args.weight_path)
        model.load_state_dict(weight_dict)

        psnr_totle = 0.
        ssim_totle = 0.
        dataProc = DcmTransforms(96)
        pet = dataProc.transform(args.pet_path).unsqueeze(0)
        gt = dataProc.transform(args.gt_path).unsqueeze(0)
        delay_time = torch.ones(1, ) * int(args.delayed_time)
        delay_time = delay_time.to(device)
        out = ddpm_.sample(model, shape=pet.shape, y=pet, delay_time=delay_time)[-1].cpu()
        psnr = psnr_(out, gt)
        ssim = ssim_(out, gt)
        mse = mse_(out, gt)
        logging.info(f"delay time: {delay_time.item()} | psnr: {psnr} | ssim: {ssim} | mse: {mse}")
        cv2.imwrite(f"{args.sample_name}/real/gt.png", DcmTransform.reverse_transform(gt.cpu()))
        cv2.imwrite(f"{args.sample_name}/pet/input_pet.png", DcmTransform.reverse_transform(pet.cpu()))
        cv2.imwrite(f"{args.sample_name}/gen/delayTime{args.delayed_time}_gen.png", DcmTransform.reverse_transform(out))

    elif args.runType == "testWholeBody":
        if not os.path.exists(args.sample_name):
            os.makedirs(f"{args.sample_name}/pet")
            os.makedirs(f"{args.sample_name}/gen")
            os.makedirs(f"{args.sample_name}/real")
        logging.basicConfig(format="%(message)s", level=logging.INFO, filename=args.test_logging_name, filemode="w")
        print("runType:{}| weight_path:{}| batch_size:{}| timeSteps:{}| condition:{}| embDT:{}".format(
            args.runType, args.weight_path, args.batch_size, args.timeSteps, args.condition, args.embDT))
        # load weight
        weight_dict = torch.load(args.weight_path)
        model.load_state_dict(weight_dict)

        wb_dataset = DcmDataset(datasetRootPath=args.wb_dataset_root_path, transforms=DcmTransform)
        wb_dataloader = DataLoader(
            dataset=wb_dataset,
            batch_size=1,
            shuffle=False,
        )

        all_image = []
        all_label = []
        all_input = []
        psnr_totle = 0.
        ssim_totle = 0.
        print(len(wb_dataloader))
        for step, batch in enumerate(wb_dataloader):
            pet = batch["pet"].cuda(non_blocking=True)
            petDelay = batch["petDelay"]
            delay_time = batch["delay_time"].cuda(non_blocking=True)
            # SUVgap = batch["SUVgap"]
            B = delay_time.shape[0]
            delay_t = torch.ones(size=(B,), device=device) * delay_time
            out = ddpm_.sample(model, shape=pet.shape, y=pet, delay_time=delay_t)[-1].cpu()

            all_image.append(out.permute(0, 2, 3, 1).detach().cpu().numpy())
            all_label.append(petDelay.permute(0, 2, 3, 1).detach().cpu().numpy())
            all_input.append(pet.permute(0, 2, 3, 1).detach().cpu().numpy())

            psnr_batch = psnr_(out, petDelay)
            ssim_batch = ssim_(out, petDelay)
            psnr_totle += psnr_batch
            ssim_totle += ssim_batch

            print(f"Layer [{step}/{len(wb_dataloader)}]...")

        images = np.concatenate(all_image, axis=0)
        np.save(f"{args.sample_name}/gen.npy", images)
        labels = np.concatenate(all_label, axis=0)
        np.save(f"{args.sample_name}/gt.npy", labels)
        inputs = np.concatenate(all_input, axis=0)
        np.save(f"{args.sample_name}/input.npy", inputs)

        print(f"psnr: {psnr_totle / step} | ssim: {ssim_totle / step}")




    elif args.runType == "predict":
        pass
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="train start from")
    parser.add_argument("--n_epochs", type=int, default=1000, help="train end to")
    parser.add_argument("--height", type=int, default=96, help="the height of input image")
    parser.add_argument("--width", type=int, default=96, help="the width of input image")
    parser.add_argument("--channels", type=int, default=1, help="the channels of input image")
    parser.add_argument("--dataset_name", type=str, default="PCP_DCM", help="the name of dataset")
    parser.add_argument("--dataset_root_path", type=str, default="../../Dual_Time_Dataset/PCP", help="the root path of dataset")
    parser.add_argument("--dim_mults", type=tuple, default=(1, 2, 4, 8), help="the dim mults rate of each stage")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: b1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: b2")
    parser.add_argument("--loss_type", type=str, default="l2", help="loss fn type")
    parser.add_argument("--model_dim", type=int, default=32, help="the dim of unet")

    parser.add_argument("--deviceType", type=str, default="device0", help="the device u use in linux")
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size of datasets")
    parser.add_argument("--timeSteps", type=int, default=300, help="forward & backward step")
    parser.add_argument("--embDTMode", type=int, default=1) # 1.Each_BLOCK 2.LINEAR_CAT 3.ADD 4.LINEAR_ADD
    parser.add_argument("--transEmbDTMode", type=int, default=1) # 1.Each_BLOCK 2.LINEAR_CAT 3.ADD 4.LINEAR_ADD
    parser.add_argument("--condition", type=bool, default=True, help="if use condition")
    parser.add_argument("--embDT", type=bool, default=True, help="if use time embedding")
    parser.add_argument("--runType", type=str, default="train", help="the runner type u use eg.train/test/predict")

    parser.add_argument("--weight_save_path", type=str, default=f"../../Dual_Time_Weights/PCP_DCM_1000.pth")
    parser.add_argument("--weight_path", type=str, default=f"../../Dual_Time_Weights/PCP_DCM_1000.pth")
    parser.add_argument("--logging_name", type=str, default=f"../assert/logging/TransUnet.log")
    parser.add_argument("--test_logging_name", type=str, default="../assert/TransUnet_test.log")
    parser.add_argument("--sample_name", type=str, default="../assert/wb_sample")

    parser.add_argument("--imagesLossingPath", type=str, default="../logging/imagesLogging.log")

    parser.add_argument("--delayed_time", type=int, default=120)
    parser.add_argument("--pet_path", type=str, default="")
    parser.add_argument("--gt_path", type=str, default="")

    parser.add_argument("--wb_dataset_root_path", type=str, default="../../Dual_Time_Dataset/wb_PCP")
    args = parser.parse_args()

    main(args=args)

