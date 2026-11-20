#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_ROS2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#endif

// Single object detection from motion_tracker.py
struct Detection {
    std::string label;
    int origin_x = 0;
    int origin_y = 0;
    int width = 0;
    int height = 0;
    double confidence = 0.0;
};

// 4-channel haptic vest PWM outputs (0-100%) and navigation state
struct HapticCommand {
    int front_left = 0;
    int front_right = 0;
    int lower_left = 0;
    int lower_right = 0;
    std::string zone_name = "Clear";
    std::string proximity = "None";
    std::string navigation_hint = "Path clear";
};

class NeuroLensBridge {
public:
    NeuroLensBridge(int frame_width = 640,
                    int frame_height = 640,
                    double min_confidence = 50.0,
                    int speech_cooldown_sec = 4)
        : frame_width_(frame_width),
          frame_height_(frame_height),
          min_confidence_(min_confidence),
          speech_cooldown_sec_(speech_cooldown_sec) {}

    // Parses "label|origin_x|origin_y|width|height|confidence"
    bool parseDetectionMessage(const std::string& raw_msg, Detection& out_detection) const {
        std::stringstream ss(raw_msg);
        std::string token;
        std::vector<std::string> parts;

        while (std::getline(ss, token, '|')) {
            parts.push_back(token);
        }

        if (parts.size() != 6) {
            return false;
        }

        try {
            out_detection.label = parts[0];
            out_detection.origin_x = std::stoi(parts[1]);
            out_detection.origin_y = std::stoi(parts[2]);
            out_detection.width = std::stoi(parts[3]);
            out_detection.height = std::stoi(parts[4]);
            out_detection.confidence = std::stod(parts[5]);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    // Maps bounding box center X to Left, Center, or Right zone
    std::string classifyHorizontalZone(const Detection& det) const {
        int obj_center_x = det.origin_x + (det.width / 2);

        if (obj_center_x < frame_width_ / 3) {
            return "Left Zone";
        } else if (obj_center_x < 2 * (frame_width_ / 3)) {
            return "Center Zone";
        }
        return "Right Zone";
    }

    // Estimates proximity tier and PWM intensity from bounding box height ratio
    std::string estimateProximity(const Detection& det, int& out_motor_intensity) const {
        double height_ratio = static_cast<double>(det.height) / static_cast<double>(frame_height_);

        if (height_ratio >= 0.50) {
            out_motor_intensity = 90;
            return "NEAR";
        } else if (height_ratio >= 0.25) {
            out_motor_intensity = 55;
            return "MEDIUM";
        }
        out_motor_intensity = 25;
        return "FAR";
    }

    // Computes 4-channel haptic vest PWM values for obstacle avoidance
    HapticCommand computeHapticCommand(const Detection& det) const {
        HapticCommand cmd;

        if (det.confidence < min_confidence_) {
            return cmd;
        }

        cmd.zone_name = classifyHorizontalZone(det);
        int intensity = 0;
        cmd.proximity = estimateProximity(det, intensity);

        if (cmd.zone_name == "Left Zone") {
            cmd.front_left = intensity;
            cmd.lower_left = intensity / 2;
            cmd.navigation_hint = "Obstacle on left, veer right";
        } else if (cmd.zone_name == "Center Zone") {
            cmd.front_left = intensity;
            cmd.front_right = intensity;
            if (cmd.proximity == "NEAR") {
                cmd.lower_left = intensity;
                cmd.lower_right = intensity;
            }
            cmd.navigation_hint = "Obstacle ahead, stop or step around";
        } else if (cmd.zone_name == "Right Zone") {
            cmd.front_right = intensity;
            cmd.lower_right = intensity / 2;
            cmd.navigation_hint = "Obstacle on right, veer left";
        }

        return cmd;
    }

    // Generates a spoken object identification callout with per-object cooldown
    std::string buildSpeechAnnouncement(const Detection& det,
                                        const HapticCommand& haptics,
                                        bool& out_should_speak) {
        out_should_speak = false;

        if (det.confidence < min_confidence_) {
            return "";
        }

        auto now = std::chrono::steady_clock::now();
        std::string key = det.label + "_" + haptics.zone_name;

        auto it = last_spoken_time_.find(key);
        if (it != last_spoken_time_.end()) {
            auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                now - it->second).count();
            if (elapsed_sec < speech_cooldown_sec_) {
                return "";
            }
        }

        last_spoken_time_[key] = now;
        out_should_speak = true;

        std::ostringstream speech;
        speech << det.label << " in " << haptics.zone_name
               << ", " << haptics.proximity << " range. "
               << haptics.navigation_hint << ".";
        return speech.str();
    }

private:
    int frame_width_;
    int frame_height_;
    double min_confidence_;
    int speech_cooldown_sec_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_spoken_time_;
};

#ifdef USE_ROS2
class NeuroLensRosNode : public rclcpp::Node {
public:
    NeuroLensRosNode() : Node("neurolens_bridge_node"), bridge_(640, 640, 50.0, 4) {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&NeuroLensRosNode::onCameraFrame, this, std::placeholders::_1));

        detection_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/neurolens/detections", 10,
            std::bind(&NeuroLensRosNode::onDetectionReceived, this, std::placeholders::_1));

        haptic_pub_ = this->create_publisher<std_msgs::msg::String>("/neurolens/haptics", 10);
        speech_pub_ = this->create_publisher<std_msgs::msg::String>("/neurolens/speech", 10);

        RCLCPP_INFO(this->get_logger(), "NeuroLens C++ bridge node started.");
    }

private:
    void onCameraFrame(const sensor_msgs::msg::Image::SharedPtr msg) {
        RCLCPP_INFO_ONCE(this->get_logger(),
            "Receiving /camera/image_raw (%ux%u, %s)",
            msg->width, msg->height, msg->encoding.c_str());
    }

    void onDetectionReceived(const std_msgs::msg::String::SharedPtr msg) {
        Detection det;
        if (!bridge_.parseDetectionMessage(msg->data, det)) {
            RCLCPP_WARN(this->get_logger(), "Invalid detection payload: %s", msg->data.c_str());
            return;
        }

        HapticCommand haptics = bridge_.computeHapticCommand(det);
        bool should_speak = false;
        std::string speech_text = bridge_.buildSpeechAnnouncement(det, haptics, should_speak);

        std_msgs::msg::String haptic_msg;
        haptic_msg.data = "FL:" + std::to_string(haptics.front_left) +
                          " FR:" + std::to_string(haptics.front_right) +
                          " LL:" + std::to_string(haptics.lower_left) +
                          " LR:" + std::to_string(haptics.lower_right);
        haptic_pub_->publish(haptic_msg);

        if (should_speak) {
            std_msgs::msg::String speech_msg;
            speech_msg.data = speech_text;
            speech_pub_->publish(speech_msg);
            RCLCPP_INFO(this->get_logger(), "SPEAK: %s", speech_text.c_str());
        }
    }

    NeuroLensBridge bridge_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr detection_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr haptic_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr speech_pub_;
};
#endif

int main(int argc, char* argv[]) {
#ifdef USE_ROS2
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NeuroLensRosNode>());
    rclcpp::shutdown();
    return 0;
#else
    (void)argc;
    (void)argv;

    NeuroLensBridge bridge(640, 640, 50.0, 4);
    std::string line;

    // Reads piped detection packets ("label|x|y|w|h|confidence") from stdin
    while (std::getline(std::cin, line)) {
        Detection det;
        if (!bridge.parseDetectionMessage(line, det)) {
            continue;
        }

        HapticCommand haptics = bridge.computeHapticCommand(det);
        bool should_speak = false;
        std::string speech_text = bridge.buildSpeechAnnouncement(det, haptics, should_speak);

        std::cout << "HAPTICS FL:" << haptics.front_left
                  << " FR:" << haptics.front_right
                  << " LL:" << haptics.lower_left
                  << " LR:" << haptics.lower_right << "\n";

        if (should_speak) {
            std::cout << "SPEECH " << speech_text << "\n";
        }
        std::cout.flush();
    }

    return 0;
#endif
}
