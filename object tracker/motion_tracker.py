import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class MotionTrackerNode(Node):
    def __init__(self):
        super().__init__('motion_tracker_node')

        # Load MediaPipe EfficientDet model from the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'efficientdet_lite0.tflite')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f'Missing MediaPipe model file: {model_path}. '
                'Place efficientdet_lite0.tflite in the object tracker/ folder.'
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            max_results=5,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

        # Subscribe to Gazebo camera frames and publish detections to C++ bridge
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, 10
        )
        self.detection_pub = self.create_publisher(
            String, '/neurolens/detections', 10
        )
        self.get_logger().info('Motion tracker listening on /camera/image_raw')

    def on_image(self, msg: Image):
            # Convert ROS Image buffer to writable, contiguous RGB and BGR NumPy arrays
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))[:, :, :3]
        if msg.encoding in ('bgr8', 'bgra8'):
            image_rgb = np.ascontiguousarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            frame_bgr = np.ascontiguousarray(raw.copy())
        else:
            image_rgb = np.ascontiguousarray(raw.copy())
            frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w, _ = frame_bgr.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.detector.detect(mp_image)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_x = bbox.origin_x
            start_y = bbox.origin_y
            end_x = bbox.origin_x + bbox.width
            end_y = bbox.origin_y + bbox.height

            obj_center_x = start_x + (bbox.width // 2)
            if obj_center_x < w / 3:
                zone = 'Left Zone'
            elif obj_center_x < 2 * (w / 3):
                zone = 'Center Zone'
            else:
                zone = 'Right Zone'

            category = detection.categories[0]
            label_name = category.category_name
            score = round(category.score * 100, 1)

            # Publish detection packet for cpp_bridge: "label|x|y|w|h|confidence"
            packet = String()
            packet.data = f'{label_name}|{start_x}|{start_y}|{bbox.width}|{bbox.height}|{score}'
            self.detection_pub.publish(packet)

            cv2.rectangle(frame_bgr, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            display_text = f'{label_name} ({zone}) {score}%'
            y_text = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(
                frame_bgr, display_text, (start_x, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.imshow('NeuroLens - MediaPipe Object Detection', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MotionTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
