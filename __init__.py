from .nodes_wan_svi import WanSVIImageToVideo, SVIExtractLastFrames, SVIExtractLastImages, SVIPaddingControl, WanSVIExtension

NODE_CLASS_MAPPINGS = {
    "WanSVIImageToVideo": WanSVIImageToVideo,
    "SVIExtractLastFrames": SVIExtractLastFrames,
    "SVIExtractLastImages": SVIExtractLastImages,
    "SVIPaddingControl": SVIPaddingControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanSVIImageToVideo": "WAN SVI Image to Video",
    "SVIExtractLastFrames": "SVI Extract Last Frames",
    "SVIExtractLastImages": "SVI Extract Last Images",
    "SVIPaddingControl": "SVI Padding Control",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

async def comfy_entrypoint():
    return WanSVIExtension()
