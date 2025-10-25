
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray  

import depthai as dai

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_oakd import OAKCameraConfig

logger = logging.getLogger(__name__)

class OakDCamera(Camera):
    def __init__(self, config: OAKCameraConfig):
        super().__init__(config)
        self.config = config

        self.serial_number = config.serial_number
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.pipeline: dai.Pipeline | None = None
        self.device: dai.Device | None = None

        self.q_rgb = None
        self.q_depth = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.capture_width, self.capture_height = config.width, config.height

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"
    
    @property

    def is_connected(self) -> bool:
        return self.device is not None

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        self.pipeline = self._create_pipeline()
        self.device = (
            dai.Device(self.pipeline, self.serial_number)
            if self.serial_number
            else dai.Device(self.pipeline)
        )
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize = 4, blocking = False)
        if self.config.use_depth:
            self.q_depth = self.device.getOutputQueue("depth", maxSize = 4, blocking = False)

        if warmup:
            time.sleep(
                1 )
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)
        logger.info(f"{self} connected.")


    def _create_pipeline(self) -> dai.Pipeline:
        pipeline = dai.Pipeline()

        # Color camera
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setPreviewSize(self.capture_width, self.capture_height)
        cam_rgb.setFps(self.fps)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(
            dai.ColorCameraProperties.ColorOrder.RGB
            if self.color_mode == ColorMode.RGB
            else dai.ColorCameraProperties.ColorOrder.BGR
        )

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Stereo depth (optional)
        if self.config.use_depth:
            left = pipeline.create(dai.node.MonoCamera)
            right = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            left.out.link(stereo.left)
            right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        return pipeline

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Lists all connected OAK-D / DepthAI devices.

        Returns:
            List of dicts with:
            - id: MxID (serial number)
            - name: product name
            - usb_speed: USB connection type
            - state: connection availability
        """
        found_devices_info = []

        try:
            devices = dai.Device.getAllAvailableDevices()
        except Exception as e:
            logger.error(f"DepthAI device discovery failed: {e}")
            return found_devices_info

        for dev in devices:
            try:
                mxid = dev.getMxId()
                name = dev.getProductName() if hasattr(dev, "getProductName") else "Unknown"
                usb_speed = str(dev.getUsbSpeed()) if hasattr(dev, "getUsbSpeed") else "Unknown"
                found_devices_info.append(
                    {
                        "id": mxid,
                        "name": name,
                        "type": "OAK-D",
                        "usb_speed": usb_speed,
                        "state": "AVAILABLE" if dev.state == dai.XLinkDeviceState.X_LINK_UNBOOTED else "IN_USE",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to query device info: {e}")

        return found_devices_info
    
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """Reads one depth frame from the OAK-D (if enabled)."""
        if not self.is_connected or not self.config.use_depth:
            raise DeviceNotConnectedError(f"{self} depth stream not available.")

        in_depth = self.q_depth.tryGet() if self.q_depth else None
        if in_depth is None:
            raise TimeoutError(f"{self} timed out waiting for depth frame.")

        depth_frame = in_depth.getFrame()
        return depth_frame
    

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        """Reads one RGB frame from the OAK-D."""
        if not self.is_connected or self.q_rgb is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        in_rgb = self.q_rgb.tryGet()
        if in_rgb is None:
            raise TimeoutError(f"{self} timed out waiting for RGB frame.")

        frame = in_rgb.getCvFrame()
        return self._postprocess_image(frame, color_mode=color_mode)
    

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Applies color conversion and rotation."""
        if color_mode is None:
            color_mode = self.color_mode

        processed = image
        if color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]:
            processed = cv2.rotate(processed, self.rotation)

        return processed
    
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Returns the most recent frame read by a background thread."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timed out after {timeout_ms} ms")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self}: No frame available after event set.")
        return frame

    def _read_loop(self) -> None:
        """Background loop that continuously reads frames."""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized.")

        while not self.stop_event.is_set():
            try:
                frame = self.read(timeout_ms=500)
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except Exception as e:
                logger.warning(f"Error reading OAK frame: {e}")
                time.sleep(0.05)

    def _start_read_thread(self) -> None:
        """Starts a background thread to keep updating latest_frame."""
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Stops background thread safely."""
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self.thread, self.stop_event = None, None

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def disconnect(self) -> None:
        """Stops threads and closes the OAK-D device."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.device is not None:
            self.device.close()
            self.device = None

        logger.info(f"{self} disconnected.")