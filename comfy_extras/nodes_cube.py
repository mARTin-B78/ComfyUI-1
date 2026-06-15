"""
Nodes for native Roblox Cube3D text-to-3D support.

Graph:
  CLIPLoader(clip-l) -> CLIPTextEncode -> CONDITIONING
  UNETLoader(shape_gpt) -> MODEL --\
  VAELoader(shape_tokenizer) -> VAE -> CubeCodebookPatch -> MODEL
  CFGGuider(MODEL, pos, neg, cfg) + SamplerCube + (trivial sigmas) + EmptyCubeLatent
      -> SamplerCustomAdvanced -> LATENT (token IDs)
  VAEDecodeCube(VAE, LATENT) -> MESH -> SaveGLB
"""

import numpy as np
import torch
from typing_extensions import override

import comfy.ldm.cube.vae
import comfy.model_management
import comfy.samplers
from comfy_api.latest import ComfyExtension, IO, Types
from comfy_extras.nodes_save_3d import pack_variable_mesh_batch


class EmptyCubeLatent(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyCubeLatent",
            category="latent/3d",
            inputs=[
                IO.Int.Input("num_tokens", default=1024, min=1, max=8192,
                             tooltip="Shape token sequence length. Must match the tokenizer "
                                     "(1024 for cube3d-v0.5, 512 for v0.1)."),
                IO.Int.Input("batch_size", default=1, min=1, max=64),
            ],
            outputs=[IO.Latent.Output()],
        )

    @classmethod
    def execute(cls, num_tokens, batch_size) -> IO.NodeOutput:
        # Channels-first 1D latent (B, 1, num_tokens), mirroring Hunyuan3Dv2's (B, C, L)
        # convention (latent_channels=1). The sampler only uses the sequence length.
        latent = torch.zeros([batch_size, 1, num_tokens], device=comfy.model_management.intermediate_device())
        return IO.NodeOutput({"samples": latent, "type": "cube_tokens"})


class CubeCodebookPatch(IO.ComfyNode):
    """Inject the projected VQ codebook into the GPT token-embedding table.

    Upstream copies shape_proj(tokenizer.codebook) into wte.weight[:num_codes] at load
    time; without it generation is garbage. Done here as a ModelPatcher object patch so
    it composes with normal model loading/offload."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CubeCodebookPatch",
            display_name="Cube Codebook Patch",
            category="advanced/model",
            inputs=[
                IO.Model.Input("model"),
                IO.Vae.Input("vae"),
            ],
            outputs=[IO.Model.Output()],
        )

    @classmethod
    def execute(cls, model, vae) -> IO.NodeOutput:
        gpt = model.get_model_object("diffusion_model")
        codebook = vae.first_stage_model.bottleneck.block.get_codebook()  # (num_codes, embed_dim) fp32
        w = gpt.shape_proj.weight
        proj = gpt.shape_proj(codebook.to(device=w.device, dtype=w.dtype))  # (num_codes, n_embd)

        old = model.get_model_object("diffusion_model.transformer.wte.weight")
        new = old.clone()
        new[:proj.shape[0]] = proj.to(device=new.device, dtype=new.dtype)

        m = model.clone()
        m.add_object_patch("diffusion_model.transformer.wte.weight",
                           torch.nn.Parameter(new, requires_grad=False))
        return IO.NodeOutput(m)


class SamplerCube(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SamplerCube",
            display_name="Sampler Cube (autoregressive)",
            category="sampling/custom_sampling/samplers",
            inputs=[
                IO.Float.Input("top_p", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="1.0 = deterministic greedy (upstream default). "
                                       "<1.0 enables nucleus sampling."),
            ],
            outputs=[IO.Sampler.Output()],
        )

    @classmethod
    def execute(cls, top_p) -> IO.NodeOutput:
        return IO.NodeOutput(comfy.samplers.ksampler("cube", {"top_p": top_p}))


class VAEDecodeCube(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VAEDecodeCube",
            display_name="VAE Decode Cube (3D)",
            category="latent/3d",
            inputs=[
                IO.Vae.Input("vae"),
                IO.Latent.Input("samples"),
                IO.Float.Input("resolution_base", default=8.0, min=4.0, max=10.0, step=0.5,
                               tooltip="Grid cells per axis = 2^resolution_base. 8.0 matches "
                                       "upstream default (257^3 grid)."),
                IO.Int.Input("chunk_size", default=100000, min=1000, max=2000000, advanced=True),
            ],
            outputs=[IO.Mesh.Output()],
        )

    @classmethod
    def execute(cls, vae, samples, resolution_base, chunk_size) -> IO.NodeOutput:
        comfy.model_management.load_models_gpu([vae.patcher])
        tok = vae.first_stage_model
        ids = samples["samples"]
        ids = ids.reshape(ids.shape[0], -1)[:, :tok.cfg_num_encoder_latents].long()
        ids = ids.clamp(0, tok.cfg_num_codes - 1).to(vae.device)

        latents = tok.decode_indices(ids)
        grid, grid_size, bbox_size, bbox_min = tok.extract_geometry(
            latents, resolution_base=resolution_base, chunk_size=chunk_size)

        verts_list, faces_list = [], []
        for i in range(grid.shape[0]):
            v, f = comfy.ldm.cube.vae.grid_logits_to_mesh(grid[i], grid_size, bbox_size, bbox_min)
            verts_list.append(torch.from_numpy(v))
            faces_list.append(torch.from_numpy(f.astype(np.int64)))

        mesh = pack_variable_mesh_batch(verts_list, faces_list)
        return IO.NodeOutput(mesh)


class CubeExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            EmptyCubeLatent,
            CubeCodebookPatch,
            SamplerCube,
            VAEDecodeCube,
        ]


async def comfy_entrypoint() -> CubeExtension:
    return CubeExtension()
