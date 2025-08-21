import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs
import logging_mp
import os
import signal
import subprocess
import glob
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.DEBUG)


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            logger_mp.error('[Image Server] pipe_profile.get_device() is None .')
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera():
    def __init__(self, device_id, img_shape, fps):
        """
        decive_id: /dev/video* or *
        img_shape: [height, width]
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        if not self._can_read_frame():
            logger_mp.error(f"[Image Server] Camera {self.id} Error: Failed to initialize the camera or read frames. Exiting...")
            self.release()

    def _can_read_frame(self):
        success, _ = self.cap.read()
        return success

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image


class ImageServer:
    def __init__(self, config, port = 5555, Unit_Test = False):
        """
        config example1:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            'wrist_camera_type': 'realsense', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # realsense camera's serial number
        }

        config example2:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'realsense',                                  # opencv or realsense
            'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': ["218622271739"],                       # realsense camera's serial number
            'wrist_camera_type': 'opencv', 
            'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            'wrist_camera_id_numbers': [0,1],                                 # '/dev/video0' and '/dev/video1' (opencv)
        }

        If you are not using the wrist camera, you can comment out its configuration, like this below:
        config:
        {
            'fps':30                                                          # frame per second
            'head_camera_type': 'opencv',                                     # opencv or realsense
            'head_camera_image_shape': [480, 1280],                           # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                    # '/dev/video0' (opencv)
            #'wrist_camera_type': 'realsense', 
            #'wrist_camera_image_shape': [480, 640],                           # Wrist camera resolution  [height, width]
            #'wrist_camera_id_numbers': ["218622271789", "241222076627"],      # serial number (realsense)
        }
        """
        logger_mp.info(config)
        self.fps = config.get('fps', 30)
        self.head_camera_type = config.get('head_camera_type', 'opencv')
        self.head_image_shape = config.get('head_camera_image_shape', [480, 640])      # (height, width)
        self.head_camera_id_numbers = config.get('head_camera_id_numbers', [0])

        # Legacy single wrist group (kept for backward compatibility)
        self.wrist_camera_type = config.get('wrist_camera_type', None)
        self.wrist_image_shape = config.get('wrist_camera_image_shape', [480, 640])    # (height, width)
        self.wrist_camera_id_numbers = config.get('wrist_camera_id_numbers', None)

        # New: left/right wrist groups
        self.left_wrist_camera_type = config.get('left_wrist_camera_type', None)
        self.left_wrist_image_shape = config.get('left_wrist_camera_image_shape', [480, 640])
        self.left_wrist_camera_id_numbers = config.get('left_wrist_camera_id_numbers', None)

        self.right_wrist_camera_type = config.get('right_wrist_camera_type', None)
        self.right_wrist_image_shape = config.get('right_wrist_camera_image_shape', [480, 640])
        self.right_wrist_camera_id_numbers = config.get('right_wrist_camera_id_numbers', None)

        self.port = port
        self.Unit_Test = Unit_Test


        # Helper to normalize opencv device id inputs
        def _normalize_opencv_ids(id_values):
            # Accept: int, str ("0"), str with commas ("0,1"), list/tuple of ints/strings, '/dev/videoX'
            def _coerce_one(x):
                if isinstance(x, int):
                    return x
                if isinstance(x, str):
                    xs = x.strip()
                    if xs.startswith('/dev/video'):
                        return xs
                    if xs.isdigit():
                        return int(xs)
                return x

            if id_values is None:
                return []
            if isinstance(id_values, (list, tuple)):
                # Handle accidental single element like ["0,1"]
                if len(id_values) == 1 and isinstance(id_values[0], str) and ',' in id_values[0]:
                    return [_coerce_one(p.strip()) for p in id_values[0].split(',') if p.strip() != '']
                return [_coerce_one(v) for v in id_values]
            if isinstance(id_values, str):
                # Split comma-separated string
                if ',' in id_values:
                    return [_coerce_one(p.strip()) for p in id_values.split(',') if p.strip() != '']
                return [_coerce_one(id_values)]
            if isinstance(id_values, int):
                return [id_values]
            # Fallback
            return [id_values]

        # Initialize head cameras
        self.head_cameras = []
        if self.head_camera_type == 'opencv':
            normalized_ids = _normalize_opencv_ids(self.head_camera_id_numbers)
            for device_id in normalized_ids:
                camera = OpenCVCamera(device_id=device_id, img_shape=self.head_image_shape, fps=self.fps)
                if camera.cap.isOpened():
                    self.head_cameras.append(camera)
                else:
                    logger_mp.error(f"[Image Server] Failed to open head camera {device_id}. Skipping.")
        elif self.head_camera_type == 'realsense':
            for serial_number in self.head_camera_id_numbers:
                try:
                    camera = RealSenseCamera(img_shape=self.head_image_shape, fps=self.fps, serial_number=serial_number)
                    self.head_cameras.append(camera)
                except Exception as e:
                    logger_mp.error(f"[Image Server] Failed to open head RealSense {serial_number}: {e}. Skipping.")
        else:
            logger_mp.warning(f"[Image Server] Unsupported head_camera_type: {self.head_camera_type}")

        # Initialize wrist cameras if provided (legacy single group)
        self.wrist_cameras = []
        if self.wrist_camera_type and self.wrist_camera_id_numbers:
            if self.wrist_camera_type == 'opencv':
                normalized_ids = _normalize_opencv_ids(self.wrist_camera_id_numbers)
                for device_id in normalized_ids:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.wrist_image_shape, fps=self.fps)
                    if camera.cap.isOpened():
                        self.wrist_cameras.append(camera)
                    else:
                        logger_mp.error(f"[Image Server] Failed to open wrist camera {device_id}. Skipping.")
            elif self.wrist_camera_type == 'realsense':
                for serial_number in self.wrist_camera_id_numbers:
                    try:
                        camera = RealSenseCamera(img_shape=self.wrist_image_shape, fps=self.fps, serial_number=serial_number)
                        self.wrist_cameras.append(camera)
                    except Exception as e:
                        logger_mp.error(f"[Image Server] Failed to open wrist RealSense {serial_number}: {e}. Skipping.")
            else:
                logger_mp.warning(f"[Image Server] Unsupported wrist_camera_type: {self.wrist_camera_type}")

        # Initialize left wrist cameras
        self.left_wrist_cameras = []
        if self.left_wrist_camera_type and self.left_wrist_camera_id_numbers:
            if self.left_wrist_camera_type == 'opencv':
                normalized_ids = _normalize_opencv_ids(self.left_wrist_camera_id_numbers)
                for device_id in normalized_ids:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.left_wrist_image_shape, fps=self.fps)
                    if camera.cap.isOpened():
                        self.left_wrist_cameras.append(camera)
                    else:
                        logger_mp.error(f"[Image Server] Failed to open left wrist camera {device_id}. Skipping.")
            elif self.left_wrist_camera_type == 'realsense':
                for serial_number in self.left_wrist_camera_id_numbers:
                    try:
                        camera = RealSenseCamera(img_shape=self.left_wrist_image_shape, fps=self.fps, serial_number=serial_number)
                        self.left_wrist_cameras.append(camera)
                    except Exception as e:
                        logger_mp.error(f"[Image Server] Failed to open left wrist RealSense {serial_number}: {e}. Skipping.")
            else:
                logger_mp.warning(f"[Image Server] Unsupported left_wrist_camera_type: {self.left_wrist_camera_type}")

        # Initialize right wrist cameras
        self.right_wrist_cameras = []
        if self.right_wrist_camera_type and self.right_wrist_camera_id_numbers:
            if self.right_wrist_camera_type == 'opencv':
                normalized_ids = _normalize_opencv_ids(self.right_wrist_camera_id_numbers)
                for device_id in normalized_ids:
                    camera = OpenCVCamera(device_id=device_id, img_shape=self.right_wrist_image_shape, fps=self.fps)
                    if camera.cap.isOpened():
                        self.right_wrist_cameras.append(camera)
                    else:
                        logger_mp.error(f"[Image Server] Failed to open right wrist camera {device_id}. Skipping.")
            elif self.right_wrist_camera_type == 'realsense':
                for serial_number in self.right_wrist_camera_id_numbers:
                    try:
                        camera = RealSenseCamera(img_shape=self.right_wrist_image_shape, fps=self.fps, serial_number=serial_number)
                        self.right_wrist_cameras.append(camera)
                    except Exception as e:
                        logger_mp.error(f"[Image Server] Failed to open right wrist RealSense {serial_number}: {e}. Skipping.")
            else:
                logger_mp.warning(f"[Image Server] Unsupported right_wrist_camera_type: {self.right_wrist_camera_type}")

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.head_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Head camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Head camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in head_cameras.")

        for cam in self.wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Wrist camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in wrist_cameras.")

        for cam in self.left_wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Left wrist camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Left wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in left_wrist_cameras.")

        for cam in self.right_wrist_cameras:
            if isinstance(cam, OpenCVCamera):
                logger_mp.info(f"[Image Server] Right wrist camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                logger_mp.info(f"[Image Server] Right wrist camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                logger_mp.warning("[Image Server] Unknown camera type in right_wrist_cameras.")

        logger_mp.info("[Image Server] Image server has started, waiting for client connections...")



    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            logger_mp.info(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        for cam in self.head_cameras:
            cam.release()
        for cam in self.wrist_cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        logger_mp.info("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                sections = []

                # Helper to read and concat frames for a camera group
                def _read_group_frames(cameras, cam_type):
                    if not cameras:
                        return None
                    group_frames = []
                    for cam in cameras:
                        if cam_type == 'opencv':
                            color_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.error("[Image Server] Camera frame read is error (opencv group).")
                                return None
                        elif cam_type == 'realsense':
                            color_image, depth_image = cam.get_frame()
                            if color_image is None:
                                logger_mp.error("[Image Server] Camera frame read is error (realsense group).")
                                return None
                        else:
                            return None
                        group_frames.append(color_image)
                    if not group_frames:
                        return None
                    return cv2.hconcat(group_frames)

                # Head group
                head_color = _read_group_frames(self.head_cameras, self.head_camera_type)
                if head_color is not None:
                    sections.append(head_color)

                # Legacy wrist group
                wrist_color = _read_group_frames(self.wrist_cameras, self.wrist_camera_type)
                if wrist_color is not None:
                    sections.append(wrist_color)

                # Left/right wrist groups
                left_wrist_color = _read_group_frames(self.left_wrist_cameras, self.left_wrist_camera_type)
                if left_wrist_color is not None:
                    sections.append(left_wrist_color)

                right_wrist_color = _read_group_frames(self.right_wrist_cameras, self.right_wrist_camera_type)
                if right_wrist_color is not None:
                    sections.append(right_wrist_color)

                if not sections:
                    time.sleep(0.01)
                    continue
                full_color = sections[0] if len(sections) == 1 else cv2.hconcat(sections)

                # Guard against empty images
                if full_color is None or (hasattr(full_color, 'size') and full_color.size == 0):
                    time.sleep(0.005)
                    continue

                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    logger_mp.error("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                self.socket.send(message)
                #print(f"send message out message.size = {message.count}")
                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            logger_mp.warning("[Image Server] Interrupted by user.")
        finally:
            self._close()


if __name__ == "__main__":
    def _preclean_camera_users():
        try:
            # Best-effort: terminate other image_server instances
            self_pid = os.getpid()
            try:
                res = subprocess.run(["pgrep", "-fa", "image_server.py"], capture_output=True, text=True)
                for line in res.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split(" ", 1)
                    try:
                        pid = int(parts[0])
                    except Exception:
                        continue
                    if pid == self_pid:
                        continue
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
            except Exception:
                pass

            # Kill processes using /dev/video* if fuser is available
            try:
                if subprocess.run(["which", "fuser"], capture_output=True).returncode == 0:
                    for dev in sorted(glob.glob("/dev/video*")):
                        subprocess.run(["fuser", "-k", "-TERM", dev], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.2)
            except Exception:
                pass
        except Exception:
            pass

    _preclean_camera_users()

    config = {
            'fps':30,                                                          # frame per second
            # head binocluar camera
            'head_camera_type': 'opencv',                                      # opencv or realsense
            'head_camera_image_shape': [480, 1280],                            # Head camera resolution  [height, width]
            'head_camera_id_numbers': [0],                                     # disable head for now
            # wrist RealSense D405
            'wrist_camera_type': 'realsense',
            'wrist_camera_image_shape': [480, 640],
            'wrist_camera_id_numbers': ["230422271136","230322271325"]         #left, right
    }

    server = ImageServer(config, port=6688, Unit_Test=False)
    server.send_process()