import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        freqs = x * self.weights.unsqueeze(0) * 2 * torch.pi
        x = torch.cat((x, freqs.sin(), freqs.cos()), dim = -1)
        return x

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Block1d(nn.Module):
    def __init__(self, input_dim, output_dim, groups=8, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(groups, output_dim)
        self.act = nn.Mish()

    def forward(self, x, scale_shift=None):
        x = self.conv(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock1d(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, time_emb_dim: int, groups: int = 8, kernel_size: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, output_dim * 2)
        )

        self.block1 = Block1d(input_dim, output_dim, groups=groups, kernel_size=kernel_size)
        self.block2 = Block1d(output_dim, output_dim, groups=groups, kernel_size=kernel_size)
        self.res_conv = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = None
        time_emb = self.mlp(time_emb)
        time_emb = time_emb.unsqueeze(2)
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention1d(nn.Module):
    def __init__(self, dim, heads=4, head_dim=32):
        super().__init__()
        self.scale = head_dim ** -0.5
        self.heads = heads
        hidden_dim = head_dim * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.out_proj = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, self.heads, -1, n), qkv)

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h i j, b h d j -> b h d i', attn, v)
        out = out.reshape(b, -1, n)
        out = self.out_proj(out)
        return out

class TemporalUNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        random_fourier_features: int,
        learned_sin_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = hidden_dim * 4
        self.learned_sin_dim = learned_sin_dim
        self.dim_mults = dim_mults
        self.kernel_size = kernel_size
        self.resnet_block_groups = resnet_block_groups
        self.random_fourier_features = random_fourier_features
        self.attn_heads = attn_heads
        self.attn_head_dim = attn_head_dim
        
        sin_pos_emb = SinusoidalPosEmb(self.learned_sin_dim, self.random_fourier_features)

        self.time_mlp = nn.Sequential(
            sin_pos_emb,
            nn.Linear(self.learned_sin_dim + 1, self.time_emb_dim),
            nn.Mish(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        self.context_dim = context_dim
        if context_dim is not None:
            self.context_mlp = nn.Sequential(
                nn.Linear(context_dim, self.time_emb_dim),
                nn.Mish(),
                nn.Linear(self.time_emb_dim, self.time_emb_dim)
            )

        self.init_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        
        dims = [hidden_dim, *[hidden_dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([nn.ModuleList([
                ResnetBlock1d(input_dim, output_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size),
                ResnetBlock1d(output_dim, output_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size),
                Downsample1d(output_dim),
            ]) for input_dim, output_dim in in_out])
        self.ups = nn.ModuleList([nn.ModuleList([
                Upsample1d(output_dim),  
                ResnetBlock1d(output_dim * 2, input_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size),             
                ResnetBlock1d(input_dim + output_dim, input_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size),
            ]) for input_dim, output_dim in reversed(in_out)])

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size)
        self.mid_attn = Attention1d(mid_dim, heads=attn_heads, head_dim=attn_head_dim)
        self.mid_block2 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size)

        self.final_res_block = ResnetBlock1d(hidden_dim * 2, hidden_dim, time_emb_dim=self.time_emb_dim, groups=resnet_block_groups, kernel_size=kernel_size)
        self.final_conv = nn.Conv1d(hidden_dim, input_dim, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        t = self.time_mlp(time)

        if self.context_dim is not None:
            if context is None:
                raise ValueError("Model initialized with context_dim but no context provided in forward")
            c = self.context_mlp(context)
            t = t + c

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        x = x.permute(0, 2, 1)
        return x
