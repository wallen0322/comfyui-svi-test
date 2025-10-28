from .nodes_wan_svi import WanSVIImageToVideo, SVIExtractLastFrames, WanSVIExtension

NODE_CLASS_MAPPINGS = {
    "WanSVIImageToVideo": WanSVIImageToVideo,
    "SVIExtractLastFrames": SVIExtractLastFrames,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIImageToVideo": "WAN SVI Image to Video",
    "SVIExtractLastFrames": "SVI Extract Last Frames",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

async def comfy_entrypoint():
    return WanSVIExtension()
