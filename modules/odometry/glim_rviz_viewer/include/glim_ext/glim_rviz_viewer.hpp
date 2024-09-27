#pragma once

#include <any>
#include <atomic>
#include <thread>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <glim/odometry/estimation_frame.hpp>
#include <glim/mapping/sub_map.hpp>
#include <glim/util/extension_module.hpp>
#include <glim/util/extension_module_ros2.hpp>

#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>

namespace spdlog {
class logger;
}

namespace glim {

class TrajectoryManager;

/**
 * @brief Rviz-based viewer
 */
class GlimRvizViewer : public ExtensionModuleROS2 {
public:
  GlimRvizViewer();
  ~GlimRvizViewer();

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions(rclcpp::Node& node) override;

private:
  void set_callbacks();
  void odometry_new_frame(const EstimationFrame::ConstPtr& new_frame);
  void globalmap_on_update_submaps(const std::vector<SubMap::Ptr>& submaps);
  void invoke(const std::function<void()>& task);
  void apply_transform_to_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, sensor_msgs::msg::PointCloud2::SharedPtr output_cloud);
  void utm2gnss_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
  
  void spin_once();

private:
  std::atomic_bool kill_switch;
  std::thread thread;
  geometry_msgs::msg::TransformStamped T_map_gnss;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  rclcpp::Time last_globalmap_pub_time;

  std::string imu_frame_id;
  std::string lidar_frame_id;
  std::string base_frame_id;
  std::string odom_frame_id;
  std::string map_frame_id;
  bool publish_imu2lidar;
  double tf_time_offset;

  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> points_pub;
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> aligned_points_pub;
  std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> map_pub;
  std::shared_ptr<rclcpp::Subscription<geometry_msgs::msg::TransformStamped>> utm2gnss_sub;

  std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Odometry>> odom_pub;
  std::shared_ptr<rclcpp::Publisher<geometry_msgs::msg::PoseStamped>> pose_pub;

  std::mutex trajectory_mutex;
  std::unique_ptr<TrajectoryManager> trajectory;

  std::vector<gtsam_points::PointCloud::ConstPtr> submaps;

  std::mutex invoke_queue_mutex;
  std::vector<std::function<void()>> invoke_queue;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};
}  // namespace glim