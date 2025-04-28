import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import enum

from inspect import isfunction
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class EmbDtMode(enum.Enum):
    EACH_BLOCK = enum.auto
    LINEAR_CAT = enum.auto
    ADD = enum.auto
    LINEAR_ADD = enum.auto

class TransEmbDtMode(enum.Enum):
    EACH_BLOCK = enum.auto
    LINEAR_CAT = enum.auto
    ADD = enum.auto
    LINEAR_ADD = enum.auto


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction else d

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(4 * dim, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class  WeightStandardizedConv2d(nn.Conv2d):

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", torch.var)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, embDTMode, time_emb_dim=None, embDT=False, groups=8):
        super().__init__()
        self.embDT = embDT
        self.embDTMode = embDTMode
        self.timeStep_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * dim_out)
            )
        ) if exists(time_emb_dim) else None
        self.delayTime_mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, 2 * dim_out)
            ) if embDT == True else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        if embDTMode == EmbDtMode.LINEAR_ADD:
            self.embDT_linear = nn.Linear(2 * dim_out, 2 * dim_out)
        elif embDTMode == EmbDtMode.LINEAR_CAT:
            self.embDT_linear = nn.Linear(4 * dim_out, 2 * dim_out)


    def forward(self, x, time_emb=None, delay_time_emb=None):
        scale_shift = None
        if exists(self.timeStep_mlp) and exists(time_emb):
            time_emb = self.timeStep_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        
        scale_shift_delay_time = None
        if exists(self.delayTime_mlp) and exists(delay_time_emb):
            delay_time_emb = self.delayTime_mlp(delay_time_emb)
            delay_time_emb = rearrange(delay_time_emb, "b c -> b c 1 1")
            scale_shift_delay_time = delay_time_emb.chunk(2, dim=1)

        if self.embDTMode == EmbDtMode.EACH_BLOCK:
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h, scale_shift=scale_shift_delay_time)
        
        elif self.embDTMode == EmbDtMode.ADD:
            scale_shift = scale_shift + scale_shift_delay_time
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h, scale_shift=scale_shift)
        
        elif self.embDTMode == EmbDtMode.LINEAR_ADD:
            scale_shift = torch.cat([scale_shift[0], scale_shift[1]], dim=1)
            scale_shift_delay_time = torch.cat([scale_shift_delay_time[0], scale_shift_delay_time[1]], dim=1)
            scale_shift = scale_shift + scale_shift_delay_time
            scale_shift = self.embDT_linear(scale_shift)
            scale_shift = torch.chunk(2, dim=1)
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h, scale_shift=scale_shift)
        
        elif self.embDTMode == EmbDtMode.LINEAR_CAT:
            scale_shift = torch.cat([scale_shift[0], scale_shift[1]], dim=1)
            scale_shift_delay_time = torch.cat([scale_shift_delay_time[0], scale_shift_delay_time[1]], dim=1)
            scale_shift = torch.cat([scale_shift, scale_shift_delay_time], dim=1)
            scale_shift = self.embDT_linear(scale_shift)
            scale_shift = torch.chunk(2, dim=1)
            h = self.block1(x, scale_shift=scale_shift)
            h = self.block2(h, scale_shift=scale_shift)
        
        else:
            print("No any time embedding!!!")
            h = self.block1(x, scale_shift=None)
            h = self.block2(h, scale_shift=None)

        h = h + self.res_conv(x)
        return h

class TBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.time_qkv = nn.Linear(dim, hidden_dim * 3)

        self.out_Attn = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, 3, padding=1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x, time_token=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        if exists(time_token):
            tqkv = self.time_qkv(time_token).chunk(3, dim=1)
            tq, tk, tv = map(
                lambda t: rearrange(t, "b (h c) -> b h c 1", h=self.heads), tqkv
            )
            q = torch.cat([q, tq], dim=-1)
            k = torch.cat([k, tk], dim=-1)
            v = torch.cat([v, tv], dim=-1)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k , v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        if exists(time_token):
            out = out[:, :, :, :-1]
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.out_Attn(out)

        x = out + x

        hidden = self.mlp(x)
        x = x + hidden

        return x

class TransBlock(nn.Module):
    def __init__(self, dim, embDtMode, heads=4, dim_head=32, time_emb_dim=None, embDT=False):
        super().__init__()
        self.embDtMode = embDtMode
        self.timeStepMlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim)
            ) if exists(time_emb_dim) else None
        )
        self.delayTimeMlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim)
            ) if embDT == True else None
        )

        self.block1 = TBlock(dim=dim, heads=heads, dim_head=dim_head)
        self.block2 = TBlock(dim=dim, heads=heads, dim_head=dim_head)

        if embDtMode == TransEmbDtMode.LINEAR_ADD:
            self.embDT_linear = nn.Linear(dim, dim)
        if embDtMode == TransEmbDtMode.LINEAR_CAT:
            self.embDT_linear = nn.Linear(2 * dim, dim)

    def forward(self, x, time_emb=None, delay_time_emb=None):
        time = None
        if exists(self.timeStepMlp) and exists(time_emb):
            time = self.timeStepMlp(time_emb)
        delay_time = None
        if exists(self.delayTimeMlp) and exists(delay_time_emb):
            delay_time = self.delayTimeMlp(delay_time_emb)

        if self.embDtMode == TransEmbDtMode.EACH_BLOCK:
            h = self.block1(x, time)
            h = self.block2(h, delay_time)

        elif self.embDtMode == TransEmbDtMode.ADD:
            time = time + delay_time
            h = self.block1(x, time)
            h = self.block2(h, time)

        elif self.embDtMode == TransEmbDtMode.LINEAR_ADD:
            time = time + delay_time
            time = self.embDT_linear(time)
            h = self.block1(x, time)
            h = self.block2(h, time)

        elif self.embDtMode == TransEmbDtMode.LINEAR_CAT:
            time = torch.cat([time, delay_time], dim=1)
            time = self.embDT_linear(time)
            h = self.block1(x, time)
            h = self.block2(h, time)

        return h

class Unet(nn.Module):
    def __init__(self, 
                dim, 
                dim_mults=(1, 2, 4, 8), 
                channels=1,
                embDTMode=EmbDtMode.EACH_BLOCK,
                transEmbDTMode=TransEmbDtMode.EACH_BLOCK,
                self_condition=True,
                embDT = True,
                ):
        super().__init__()

        self.channels = channels
        self.input_channsle = channels * (2 if self_condition else 1)
        self.self_condition = self_condition
        self.embDT = embDT
        self.embDTMode = embDTMode
        self.transEmbDTMode = transEmbDTMode

        self.init_conv = nn.Conv2d(self.input_channsle, dim, 1, padding=0)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out= list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        if embDT:
            self.delay_time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            self.delay_time_mlp = nn.Identity()


        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResBlock(dim=dim_in, 
                                dim_out=dim_in, 
                                embDTMode=self.embDTMode, 
                                time_emb_dim=time_dim, 
                                embDT=self.embDT),
                        TransBlock(dim=dim_in, 
                                    embDtMode=self.transEmbDTMode,
                                    time_emb_dim=time_dim,
                                    embDT=self.embDT),
                        Downsample(dim=dim_in, dim_out=dim_out)
                        if not is_last else
                        nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(dim=mid_dim,
                                    dim_out=mid_dim,
                                    embDTMode=self.embDTMode,
                                    time_emb_dim=time_dim,
                                    embDT=self.embDT)
        self.mid_transformer = TransBlock(dim=mid_dim,
                                        embDtMode=self.transEmbDTMode,
                                        time_emb_dim=time_dim,
                                        embDT=self.embDT)
        self.mid_block2 = ResBlock(dim=mid_dim,
                                    dim_out=mid_dim,
                                    embDTMode=self.embDTMode,
                                    time_emb_dim=time_dim,
                                    embDT=self.embDT)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResBlock(dim=dim_in + dim_out, 
                                dim_out=dim_out, 
                                embDTMode=self.embDTMode, 
                                time_emb_dim=time_dim, 
                                embDT=self.embDT),
                        TransBlock(dim=dim_out, 
                                    embDtMode=self.transEmbDTMode,
                                    time_emb_dim=time_dim,
                                    embDT=self.embDT),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )
        
        self.out_dim = self.channels
        self.final_res_block = ResBlock(dim=dim * 2, 
                                dim_out=dim, 
                                embDTMode=self.embDTMode, 
                                time_emb_dim=time_dim, 
                                embDT=self.embDT)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, delay_time, x_self_cond):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        delay_t = self.delay_time_mlp(delay_time)

        h = []

        for resBlock, transBlock, downsample in self.downs:
            x = resBlock(x, t, delay_t)
            x = transBlock(x, t, delay_t)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, delay_t)
        x = self.mid_transformer(x, t, delay_t)
        x = self.mid_block2(x, t, delay_t)

        for resBlock, transBlock, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resBlock(x, t, delay_t)
            x = transBlock(x, t, delay_t)

            x = upsample(x)

        x = torch.cat([x, r], dim=1)

        x = self.final_res_block(x, t, delay_t)
        return self.final_conv(x)


        
if __name__ == "__main__":
    '''
    # TB = TransBlock(128, TransEmbDtMode.LINEAR_CAT, time_emb_dim=128, embDT=True)
    x = torch.randn(32, 64, 32, 32)
    time = torch.randn(32, 128)
    delay_time = torch.randn(32, 128)
    # TB(x, time, delay_time)

    RB = ResBlock(64, 128, embDTMode=EmbDtMode.EACH_BLOCK, time_emb_dim=128)
    RB(x, time, delay_time)
    '''
    from thop import profile

    model = Unet(dim=32)
    x = torch.randn(1, 1, 96, 96)
    t = torch.randn(1, )
    delay_t = torch.randn(1, )
    y = torch.randn(1, 1, 96, 96)

    flops, params = profile(model=model, inputs=(x, t, delay_t, y))
    print('the flops is {}G,the params is {}M'.format(round(flops/(10**9),2), round(params/(10**6),2)))
