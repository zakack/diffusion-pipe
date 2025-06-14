from statistics import NormalDist
from typing import Tuple
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/Wan2_1'))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import safetensors
from transformers import T5TokenizerFast, T5EncoderModel
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers import KDPM2DiscreteScheduler
from einops import rearrange

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from models.cosmos_predict2_modeling import MiniTrainDIT
from utils.common import load_state_dict, AUTOCAST_DTYPE
from utils.offloading import ModelOffloader
from wan.modules.vae import WanVAE_


SIGMA_DATA = 1.0
KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']


def get_per_sigma_loss_weights(sigma: torch.Tensor):
    """
    Args:
        sigma (tensor): noise level

    Returns:
        loss weights per sigma noise level
    """
    return (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA) ** 2


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    model.load_state_dict(
        load_state_dict(pretrained_path), assign=True)

    return model


class WanVAE:
    def __init__(self,
                 z_dim=16,
                 vae_pth=None,
                 dtype=torch.float,
                 device="cpu"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


class RectifiedFlowAB2Scheduler(KDPM2DiscreteScheduler):
    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        order: float = 7.0,
        t_scaling_factor: float = 1.0,
        use_double_precision: bool = True,
        **kpm2_kwargs,
    ):
        super().__init__(
            prediction_type="epsilon",  # placeholder, not used
            num_train_timesteps=1000,  # dummy, not used at inference
            **kpm2_kwargs,
        )
        self.gaussian_dist = NormalDist(mu=0.0, sigma=1.0)

    def sample_sigma(self, batch_size: int, timestep_quantile=None) -> torch.Tensor:
        if timestep_quantile is not None:
            cdf_vals = np.full((batch_size,), timestep_quantile)
        else:
            cdf_vals = np.random.uniform(size=(batch_size))
        samples_interval_gaussian = [self.gaussian_dist.inv_cdf(cdf_val) for cdf_val in cdf_vals]
        log_sigma = torch.tensor(samples_interval_gaussian, device="cuda")
        return torch.exp(log_sigma)

    def set_timesteps(self, num_inference_steps, device=None, num_train_timesteps: int | None = None):
        """Create Karras-like sigma schedule matching Rectified-Flow's paper."""

        device = device or torch.device("cpu")

        # Create (L + 1) sigma values following Karras et al. (Eq. 5)
        n_sigma = num_inference_steps + 1
        i = torch.arange(
            n_sigma, device=device, dtype=torch.float64 if self.config.use_double_precision else torch.float32
        )

        # Extract values from config to ensure consistency
        sigma_min = self.config.sigma_min
        sigma_max = self.config.sigma_max
        order = self.config.order

        ramp = (sigma_max ** (1 / order)) + i / (n_sigma - 1) * (sigma_min ** (1 / order) - sigma_max ** (1 / order))
        sigmas = ramp**order  # shape (n_sigma,)

        self.sigmas = sigmas.to(dtype=torch.float64 if self.config.use_double_precision else torch.float32)
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.long)
        self.num_inference_steps = num_inference_steps

        return self.timesteps


class RectifiedFlowScaling:
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1.0):
        assert abs(sigma_data - 1.0) < 1e-6, "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = sigma / (sigma + 1)
        c_skip = 1.0 - t
        c_out = -t
        c_in = 1.0 - t
        c_noise = t * self.t_scaling_factor
        return c_skip, c_out, c_in, c_noise

    def sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / sigma**2


def get_dit_config(state_dict, key_prefix=''):
    dit_config = {}
    dit_config["max_img_h"] = 240
    dit_config["max_img_w"] = 240
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[0]
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["crossattn_emb_channels"] = 1024
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = True
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["min_fps"] = 1
    dit_config["max_fps"] = 30

    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256
    if dit_config["model_channels"] == 2048:
        dit_config["num_blocks"] = 28
        dit_config["num_heads"] = 16
    elif dit_config["model_channels"] == 5120:
        dit_config["num_blocks"] = 36
        dit_config["num_heads"] = 40

    if dit_config["in_channels"] == 16:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif dit_config["in_channels"] == 17:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 3.0
        dit_config["rope_w_extrapolation_ratio"] = 3.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0

    dit_config["extra_h_extrapolation_ratio"] = 1.0
    dit_config["extra_w_extrapolation_ratio"] = 1.0
    dit_config["extra_t_extrapolation_ratio"] = 1.0
    dit_config["rope_enable_fps_modulation"] = False

    return dit_config


class CosmosPredict2Pipeline(BasePipeline):
    name = 'cosmos_predict2'
    framerate = 16
    checkpointable_layers = ['InitialLayer', 'TransformerLayer', 'FinalLayer']
    adapter_target_modules = ['Block']

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']

        # This isn't a nn.Module.
        self.vae = WanVAE(
            vae_pth=self.model_config['vae_path'],
            device='cpu',
            dtype=dtype,
        )
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        self.tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )
        t5_state_dict = load_state_dict(self.model_config['t5_path'])
        self.text_encoder = T5EncoderModel.from_pretrained(
            None,
            config='configs/t5_old/config.json',
            state_dict=t5_state_dict,
            torch_dtype='auto',
            local_files_only=True,
        )

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        rectified_flow_t_scaling_factor = 1.0
        self.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=0.002,
            sigma_max=80.0,
            order=7.0,
            t_scaling_factor=rectified_flow_t_scaling_factor,
        )
        self.scaling = RectifiedFlowScaling(SIGMA_DATA, rectified_flow_t_scaling_factor)

        state_dict = load_state_dict(self.model_config['transformer_path'])
        # Remove 'net.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                k = k[len('net.'):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        dit_config = get_dit_config(state_dict)
        with init_empty_weights():
            transformer = MiniTrainDIT(**dit_config)
        for name, p in transformer.named_parameters():
            dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
            set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])
        self.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        return [self.text_encoder]

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, diffusers_sd):
        raise NotImplementedError()

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
            round_height=8,
            round_width=8,
            round_frames=4,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # args are lists
            batch_encoding = self.tokenizer.batch_encode_plus(
                captions,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_length=True,
                return_offsets_mapping=False,
            )

            input_ids = batch_encoding.input_ids.to(text_encoder.device)
            attn_mask = batch_encoding.attention_mask.to(text_encoder.device)

            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

            encoded_text = outputs.last_hidden_state
            lengths = attn_mask.sum(dim=1).cpu()

            for batch_id in range(encoded_text.shape[0]):
                encoded_text[batch_id][lengths[batch_id] :] = 0

            return {'prompt_embeds': encoded_text}
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        x0_B_C_T_H_W = inputs['latents'].float()
        prompt_embeds = inputs['prompt_embeds']
        mask = inputs['mask']

        bs, channels, num_frames, h, w = x0_B_C_T_H_W.shape
        device = x0_B_C_T_H_W.device

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        # draw_training_sigma_and_epsilon in original code
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device=device)
        sigma_B = self.scheduler.sample_sigma(bs, timestep_quantile=timestep_quantile).to(device=device)
        sigma_B_T = rearrange(sigma_B, "b -> b 1")  # add a dimension for T, all frames share the same sigma

        # Get the mean and stand deviation of the marginal probability distribution.
        mean_B_C_T_H_W, std_B_T = x0_B_C_T_H_W, sigma_B_T
        # Generate noisy observations
        xt_B_C_T_H_W = mean_B_C_T_H_W + epsilon_B_C_T_H_W * rearrange(std_B_T, "b t -> b 1 t 1 1")

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        # get precondition for the network
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)

        x_B_C_T_H_W=(xt_B_C_T_H_W * c_in_B_1_T_1_1)
        timesteps_B_T=c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4])

        return (x_B_C_T_H_W, timesteps_B_T, prompt_embeds, xt_B_C_T_H_W, sigma_B_T), (x0_B_C_T_H_W, mask)

    def to_layers(self):
        transformer = self.transformer
        layers = [InitialLayer(transformer)]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer, self.scaling))
        return layers

    # Default loss_fn. MSE between output and target, with mask support.
    def get_loss_fn(self):
        def loss_fn(output, label):
            x0_pred_B_C_T_H_W, weights_per_sigma_B_T = output
            x0_B_C_T_H_W, mask = label
            with torch.autocast('cuda', enabled=False):
                x0_pred_B_C_T_H_W = x0_pred_B_C_T_H_W.to(torch.float32)
                x0_B_C_T_H_W = x0_B_C_T_H_W.to(x0_pred_B_C_T_H_W.device, torch.float32)
                pred_mse_B_C_T_H_W = F.mse_loss(x0_pred_B_C_T_H_W, x0_B_C_T_H_W, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    pred_mse_B_C_T_H_W *= mask
                edm_loss_B_C_T_H_W = pred_mse_B_C_T_H_W * rearrange(weights_per_sigma_B_T, "b t -> b 1 t 1 1")
                edm_loss_B_C_T_H_W = edm_loss_B_C_T_H_W.mean()
            return edm_loss_B_C_T_H_W
        return loss_fn

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()


class InitialLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        x_B_C_T_H_W, timesteps_B_T, crossattn_emb, xt_B_C_T_H_W, sigma_B_T = inputs

        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, xt_B_C_T_H_W, sigma_B_T)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, xt_B_C_T_H_W, sigma_B_T = inputs

        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, xt_B_C_T_H_W, sigma_B_T)


class FinalLayer(nn.Module):
    def __init__(self, model, scaling):
        super().__init__()
        self.final_layer = model.final_layer
        self.scaling = scaling
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def get_per_sigma_loss_weights(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.pipe.sigma_data**2) / (sigma * self.pipe.sigma_data) ** 2

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, xt_B_C_T_H_W, sigma_B_T = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)

        sigma_B_1_T_1_1 = rearrange(sigma_B_T, "b t -> b 1 t 1 1")
        c_skip_B_1_T_1_1, c_out_B_1_T_1_1, c_in_B_1_T_1_1, c_noise_B_1_T_1_1 = self.scaling(sigma=sigma_B_1_T_1_1)
        x0_pred_B_C_T_H_W = c_skip_B_1_T_1_1 * xt_B_C_T_H_W + c_out_B_1_T_1_1 * net_output_B_C_T_H_W
        weights_per_sigma_B_T = get_per_sigma_loss_weights(sigma=sigma_B_T)
        return x0_pred_B_C_T_H_W, weights_per_sigma_B_T
