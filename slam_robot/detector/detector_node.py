#!/usr/bin/env python3
"""Detector_node: Finds the objects of interest in the camera image with the fine-tuned YOLO26 model.

Subscribes  image_topic          (sensor_msgs/Image, rgb8)             default /camera/image_raw
Publishes   detections_topic     (slam_robot_msgs/ObjectDetectionArray) default /detections
              one message per processed image, with the SAME header as the image (so TF lookups use the time
              the picture was taken). An empty list means "looked, saw nothing".
            debug_image_topic    (sensor_msgs/Image, rgb8)             default /detector/debug_image
              boxes drawn, target highlighted; only produced while somebody is subscribed.

All detections above confidence_threshold are published.

Command ros2 param set /detector_node target_label blue_box
"""
import os
import time
import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from slam_robot_msgs.msg import ObjectDetection, ObjectDetectionArray

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


def touches_border(x_min, y_min, x_max, y_max, width, height, margin):
    return x_min <= margin or y_min <= margin or x_max >= width - margin or y_max >= height - margin

# Boxes: iterable of (class id, confidence, x1, y1, x2, y2). Returns ObjectDetection messages, strongest first.
def to_detections(boxes, names, width, height, margin):
    out = []
    for cls_id, conf, x1, y1, x2, y2 in sorted(boxes, key=lambda b: -b[1]):
        xmin, xmax = (int(max(0, min(width, round(v)))) for v in (x1, x2))
        ymin, ymax = (int(max(0, min(height, round(v)))) for v in (y1, y2))
        out.append(ObjectDetection(label=str(names[int(cls_id)]), confidence=float(conf),
                                   x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax,
                                   truncated=bool(touches_border(xmin, ymin, xmax, ymax, width, height, margin))))
    return out


# Copy of the image with boxes: target = thick green, others = thin amber. '[cut]' marks truncated boxes.
def draw_debug(bgr, detections, target):
    img = bgr.copy()
    for d in detections:
        is_target = d.label == target
        colour = (0, 255, 0) if is_target else (0, 200, 255)                        # BGR
        cv2.rectangle(img, (d.x_min, d.y_min), (max(d.x_min, d.x_max - 1), max(d.y_min, d.y_max - 1)),
                      colour, 3 if is_target else 1)
        text = f"{d.label} {d.confidence:.2f}" + (" [cut]" if d.truncated else "")
        cv2.putText(img, text, (d.x_min, max(14, d.y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    return img


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        self.declare_parameter("model_path", "~/slam_ws/runs/detector_v1/weights/best.pt")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("debug_image_topic", "/detector/debug_image")
        self.declare_parameter("confidence_threshold", 0.4)
        self.declare_parameter("max_rate_hz", 5.0)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("edge_margin_px", 4)
        self.declare_parameter("target_label", "red_box")
        self.declare_parameter("publish_debug_image", True)

        p = lambda name: self.get_parameter(name).value
        self.conf = float(p("confidence_threshold"))
        self.min_interval = 1.0 / float(p("max_rate_hz"))
        self.imgsz = int(p("imgsz"))
        self.device = str(p("device"))
        self.margin = int(p("edge_margin_px"))
        self.target = str(p("target_label"))
        self.debug_on = bool(p("publish_debug_image"))

        self.model = self._load_model(str(p("model_path")))
        self.names = dict(self.model.names)
        self.get_logger().info(f"model classes: {list(self.names.values())}; target = '{self.target}'; "
                               f"confidence >= {self.conf}; at most {1.0 / self.min_interval:.1f} Hz")
        if self.target not in self.names.values():
            self.get_logger().warn(f"target_label '{self.target}' is not a class of this model")

        self._last_run = -1e9
        self._bad_encoding_logged = False
        self._target_seen = False
        self._frames = 0
        self._ms_total = 0.0
        self._stats_t = time.monotonic()
        self.add_on_set_parameters_callback(self._on_params)
        self.pub_det = self.create_publisher(ObjectDetectionArray, str(p("detections_topic")), 10)
        self.pub_dbg = self.create_publisher(Image, str(p("debug_image_topic")), 1)
        self.create_subscription(Image, str(p("image_topic")), self.on_image, qos_profile_sensor_data)

    # Model
    def _load_model(self, path):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"model file not found: {path} (set the model_path parameter)")
        from ultralytics import YOLO
        model = YOLO(path)
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=self.imgsz, device=self.device, verbose=False)   # warm-up
        return model

    # Runtime Parameter changes: validate and apply. Return SetParametersResult.
    def _on_params(self, params):
        for q in params:
            if q.name == "confidence_threshold" and not 0.0 <= q.value <= 1.0:
                return SetParametersResult(successful=False, reason="confidence_threshold must be between 0 and 1")
            if q.name == "max_rate_hz" and not q.value > 0:
                return SetParametersResult(successful=False, reason="max_rate_hz must be greater than 0")
            if q.name == "edge_margin_px" and q.value < 0:
                return SetParametersResult(successful=False, reason="edge_margin_px must not be negative")
            if q.name == "imgsz" and q.value < 32:
                return SetParametersResult(successful=False, reason="imgsz must be at least 32")
            if q.name == "target_label" and q.value not in self.names.values():
                return SetParametersResult(successful=False, reason=f"unknown label '{q.value}'; this model knows "
                                                                     f"{list(self.names.values())}")
        for q in params:
            if q.name == "confidence_threshold":
                self.conf = float(q.value)
            elif q.name == "max_rate_hz":
                self.min_interval = 1.0 / float(q.value)
            elif q.name == "edge_margin_px":
                self.margin = int(q.value)
            elif q.name == "imgsz":
                self.imgsz = int(q.value)
            elif q.name == "target_label":
                self.target = str(q.value)
                self._target_seen = False
            elif q.name == "publish_debug_image":
                self.debug_on = bool(q.value)
        return SetParametersResult(successful=True)

    # One camera image
    def on_image(self, msg):
        now = time.monotonic()
        if now - self._last_run < self.min_interval:
            return                                              # too soon: drop this image
        self._last_run = now
        try:
            bgr = image_to_bgr(msg)
        except ValueError as e:
            if not self._bad_encoding_logged:
                self.get_logger().error(str(e))
                self._bad_encoding_logged = True
            return

        t0 = time.monotonic()
        result = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self._ms_total += (time.monotonic() - t0) * 1000.0
        self._frames += 1

        h, w = bgr.shape[:2]
        boxes = [(int(b.cls), float(b.conf), *(float(v) for v in b.xyxy[0])) for b in result.boxes]
        dets = to_detections(boxes, self.names, w, h, self.margin)

        out = ObjectDetectionArray()
        out.header = msg.header
        out.image_width, out.image_height = w, h
        out.detections = dets
        self.pub_det.publish(out)

        if self.debug_on and self.pub_dbg.get_subscription_count() > 0:
            dbg = Image()
            dbg.header = msg.header
            dbg.height, dbg.width, dbg.encoding = h, w, "rgb8"
            dbg.is_bigendian, dbg.step = 0, w * 3
            dbg.data = np.ascontiguousarray(draw_debug(bgr, dets, self.target)[:, :, ::-1]).tobytes()
            self.pub_dbg.publish(dbg)

        self._report(dets)

    def _report(self, dets):
        target = [d for d in dets if d.label == self.target]
        if target and not self._target_seen:
            d = target[0]
            self.get_logger().info(f"'{self.target}' DETECTED  conf {d.confidence:.2f}  box ({d.x_min},{d.y_min})-({d.x_max},{d.y_max})"
                                   + ("  [cut off by the image edge]" if d.truncated else ""))
        elif not target and self._target_seen:
            self.get_logger().info(f"'{self.target}' no longer detected")
        self._target_seen = bool(target)
        if time.monotonic() - self._stats_t >= 10.0 and self._frames:
            self.get_logger().info(f"{self._frames} images in the last 10 s, mean inference {self._ms_total / self._frames:.1f} ms")
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
