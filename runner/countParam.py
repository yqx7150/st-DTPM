from thop import profile
import torch
import sys
sys.path.append("..")
from model.UnetTrans import Unet, EmbDtMode, TransEmbDtMode
import argparse

def main(args):
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


    t = torch.randn(1,)
    dt = torch.randn(1,)
    x = torch.randn(1, 1, 96, 96)
    cond = torch.randn(1, 1, 96, 96)
    flops, params = profile(model=model, inputs=(x, t, dt, cond))
    print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embDTMode", type=int, default=1)
    parser.add_argument("--transEmbDTMode", type=int, default=4)
    parser.add_argument("--model_dim", type=int, default=32, help="the dim of unet")
    parser.add_argument("--dim_mults", type=tuple, default=(1, 2, 4, 8), help="the dim mults rate of each stage")
    parser.add_argument("--channels", type=int, default=1, help="the channels of input image")
    parser.add_argument("--condition", type=bool, default=True, help="if use condition")
    parser.add_argument("--embDT", type=bool, default=True, help="if use time embedding")
    args = parser.parse_args()

    main(args=args)