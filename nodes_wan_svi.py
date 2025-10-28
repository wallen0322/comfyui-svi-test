import math
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class WanSVIImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WanSVIImageToVideo",
            category="conditioning/video_models",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Image.Input("start_image", optional=True),
                io.Image.Input("motion_frames", optional=True),
                io.Int.Input("num_motion_frames", default=5, min=0, max=20, step=1),
                io.Latent.Input("prev_latent", optional=True),
                io.Int.Input("overlap_frames", default=9, min=0, max=40, step=1),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, batch_size, 
                start_image=None, clip_vision_output=None,
                motion_frames=None, num_motion_frames=5,
                prev_latent=None, overlap_frames=9) -> io.NodeOutput:
        
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                            device=comfy.model_management.intermediate_device())
        
        image = torch.ones((length, height, width, 3), device=comfy.model_management.intermediate_device()) * 0.5
        
        if prev_latent is not None:
            prev_samples = prev_latent["samples"]
            overlap_latent_frames = ((overlap_frames - 1) // 4) + 1 if overlap_frames > 0 else 0
            if overlap_latent_frames > 0 and prev_samples.shape[2] >= overlap_latent_frames:
                latent[:, :, :overlap_latent_frames] = prev_samples[:, :, -overlap_latent_frames:]
        
        if motion_frames is not None and num_motion_frames > 0:
            motion_frames = motion_frames[-num_motion_frames:]
            motion_frames = comfy.utils.common_upscale(motion_frames.movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
            image[:motion_frames.shape[0]] = motion_frames[:, :, :, :3]
        elif start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image[:start_image.shape[0]] = start_image[:, :, :, :3]

        concat_latent_image = vae.encode(image)
        mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), 
                         device=comfy.model_management.intermediate_device())
        
        if motion_frames is not None and num_motion_frames > 0:
            motion_latent_frames = ((motion_frames.shape[0] - 1) // 4) + 1
            mask[:, :, :motion_latent_frames] = 0.0
        elif start_image is not None:
            start_frames = start_image.shape[0]
            mask[:, :, :((start_frames - 1) // 4) + 1] = 0.0
        
        if prev_latent is not None and overlap_frames > 0:
            overlap_latent_frames = ((overlap_frames - 1) // 4) + 1
            mask[:, :, :overlap_latent_frames] = 0.0

        positive = node_helpers.conditioning_set_values(positive, {
            "concat_latent_image": concat_latent_image, 
            "concat_mask": mask
        })
        negative = node_helpers.conditioning_set_values(negative, {
            "concat_latent_image": concat_latent_image, 
            "concat_mask": mask
        })

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {"samples": latent}
        
        return io.NodeOutput(positive, negative, out_latent)


class SVIExtractLastFrames(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SVIExtractLastFrames",
            category="latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Int.Input("num_frames", default=9, min=0, max=81, step=1),
            ],
            outputs=[
                io.Latent.Output(display_name="last_frames"),
            ],
        )

    @classmethod
    def execute(cls, samples, num_frames) -> io.NodeOutput:
        if num_frames == 0:
            out = {"samples": torch.zeros_like(samples["samples"][:, :, :0])}
            return io.NodeOutput(out)
        latent_frames = ((num_frames - 1) // 4) + 1
        last_latent = samples["samples"][:, :, -latent_frames:].clone()
        out = {"samples": last_latent}
        return io.NodeOutput(out)


class SVIExtractLastImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SVIExtractLastImages",
            category="image",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("num_frames", default=9, min=0, max=81, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="last_images"),
            ],
        )

    @classmethod
    def execute(cls, images, num_frames) -> io.NodeOutput:
        if num_frames == 0:
            last_images = images[:0].clone()
            return io.NodeOutput(last_images)
        last_images = images[-num_frames:].clone()
        return io.NodeOutput(last_images)


class WanSVIExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WanSVIImageToVideo,
            SVIExtractLastFrames,
            SVIExtractLastImages,
        ]

async def comfy_entrypoint() -> WanSVIExtension:
    return WanSVIExtension()
