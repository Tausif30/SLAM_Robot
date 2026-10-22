#!/usr/bin/env python3
"""detector_node: finds the red and blue objects in the robot's camera picture and publishes the same picture with a label

Subscribes  /camera/image_raw        sensor_msgs/Image (rgb8)
Publishes   /detector/debug_image    sensor_msgs/Image (rgb8) 

Parameters
  Max_rate_hz            pictures analysed per second             ros2 param set /detector_node max_rate_hz 3.0
  Confidence_threshold   ignore detections below this, 0 to 1     ros2 param set /detector_node confidence_threshold 0.6
"""
import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

IMAGE_TOPIC = "/camera/image_raw"
DEBUG_TOPIC = "/detector/debug_image"
DEFAULT_MODEL = "~/slam_ws/runs/detector_v1/weights/best.pt"
# box colour per class, BGR (OpenCV order): red, blue, light blue. Any other class gets green.
COLOURS = {"red_box": (0, 0, 255), "blue_box": (255, 0, 0), "blue_cylinder": (255, 200, 0)}
OTHER_COLOUR = (0, 255, 0)


# Converts a sensor_msgs/Image to a BGR numpy array. Raises ValueError if the encoding is not rgb8 or bgr8.
def image_to_bgr(msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding '{msg.encoding}' (expected rgb8 or bgr8)")
    h, w = int(msg.height), int(msg.width)
    rows = np.frombuffer(msg.data, np.uint8).reshape(h, int(msg.step))[:, : w * 3]    # drop row padding
    img = rows.reshape(h, w, 3)
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)

# Copy of the picture with a box and a label around every object. boxes: (class id, confidence, x1, y1, x2, y2).
def draw_debug(bgr, boxes, names):
    img = bgr.copy()
    for cls_id, conf, x1, y1, x2, y2 in boxes:
        label = str(names[cls_id])
        colour = COLOURS.get(label, OTHER_COLOUR)
        left, top = int(x1), int(y1)
        cv2.rectangle(img, (left, top), (int(x2), int(y2)), colour, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_top = max(0, top - th - base - 4)               # the label sits on the box; inside it when the box is at the top edge
        cv2.rectangle(img, (left, y_top), (left + tw + 6, y_top + th + base + 4), colour, -1)
        luminance = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        ink = (0, 0, 0) if luminance > 140 else (255, 255, 255)                      # readable on any box colour
        cv2.putText(img, text, (left + 3, y_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ink, 1, cv2.LINE_AA)
    return img

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        self.declare_parameter("model_path", os.environ.get("SLAM_MODEL_PATH", DEFAULT_MODEL))
        self.declare_parameter("max_rate_hz", 8.0)
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cpu")

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = str(self.get_parameter("device").value)
        self.model = self._load_model(str(self.get_parameter("model_path").value))
        self.names = dict(self.model.names)
        self.get_logger().info(f"model classes: {list(self.names.values())}")

        self._last_run = -1e9
        self._bad_encoding_logged = False
        self._in_view = None
        self._frames = 0
        self._ms_total = 0.0
        self._stats_t = time.monotonic()
        self.pub = self.create_publisher(Image, DEBUG_TOPIC, 1)
        self.create_subscription(Image, IMAGE_TOPIC, self.on_image, qos_profile_sensor_data)

    def _load_model(self, path):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model file not found: {path} (set $SLAM_MODEL_PATH or the model_path parameter)")
        from ultralytics import YOLO
        model = YOLO(path)
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, device=self.device, verbose=False)   # warm-up
        self.get_logger().info(f"model: {path}")
        return model

    def on_image(self, msg):
        if self.pub.get_subscription_count() == 0:
            return                                                    # nobody is looking: do not spend CPU
        rate = max(float(self.get_parameter("max_rate_hz").value), 0.1)
        now = time.monotonic()
        if now - self._last_run < 1.0 / rate:
            return                                                    # too soon: drop this picture
        self._last_run = now
        try:
            bgr = image_to_bgr(msg)
        except ValueError as e:
            if not self._bad_encoding_logged:
                self.get_logger().error(str(e))
                self._bad_encoding_logged = True
            return

        conf = min(max(float(self.get_parameter("confidence_threshold").value), 0.0), 1.0)
        t0 = time.monotonic()
        result = self.model.predict(bgr, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self._ms_total += (time.monotonic() - t0) * 1000.0
        self._frames += 1

        boxes = [(int(b.cls), float(b.conf), *(float(v) for v in b.xyxy[0])) for b in result.boxes]
        h, w = bgr.shape[:2]
        out = Image()
        out.header = msg.header
        out.height, out.width, out.encoding = h, w, "rgb8"
        out.is_bigendian, out.step = 0, w * 3
        out.data = np.ascontiguousarray(draw_debug(bgr, boxes, self.names)[:, :, ::-1]).tobytes()
        self.pub.publish(out)
        self._report([self.names[b[0]] for b in boxes])

    def _report(self, labels):
        in_view = tuple(sorted(set(labels)))
        if in_view != self._in_view:                                  # only when it changes
            self._in_view = in_view
            self.get_logger().info("in view: " + (", ".join(in_view) if in_view else "nothing"))
        if time.monotonic() - self._stats_t >= 10.0 and self._frames:
            self.get_logger().info(f"{self._frames} pictures in the last 10 s, mean inference {self._ms_total / self._frames:.1f} ms")
            self._frames, self._ms_total, self._stats_t = 0, 0.0, time.monotonic()

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
