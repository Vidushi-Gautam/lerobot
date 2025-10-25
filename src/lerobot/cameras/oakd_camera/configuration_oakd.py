
from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

@CameraConfig.register_subclass("oakd")
@dataclass

class OakDCameraConfig(CameraConfig):
    serial_number_or_name = str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = True #may change later
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )
            
        if self.rotation not in (
                    Cv2Rotation.NO_ROTATION,
                    Cv2Rotation.ROTATE_90,
                    Cv2Rotation.ROTATE_180,
                    Cv2Rotation.ROTATE_270,
                ):
                    raise ValueError(
                        f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
                    )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
