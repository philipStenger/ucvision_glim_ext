#include <deque>
#include <atomic>
#include <thread>
#include <numeric>
#include <Eigen/Core>

#define GLIM_ROS2

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

#ifdef GLIM_ROS2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <glim/util/extension_module_ros2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
using PoseWithCovarianceStampedConstPtr = geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <tf2_eigen/tf2_eigen.hpp>  
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Geometry>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/odometry.hpp>

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glim/util/logging.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>

namespace glim {

using gtsam::symbol_shorthand::X;

/**
 * @brief Naive implementation of GNSS constraints for the global optimization.
 * @note  This implementation is very naive and ignores the IMU-GNSS transformation and GNSS observation covariance.
 *        If you use a precise GNSS (e.g., RTK), consider asking for a closed-source extension module with better GNSS handling.
 */
class GNSSGlobal : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GNSSGlobal() : logger(create_module_logger("gnss_global")) {
    logger->info("initializing GNSS global constraints");
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_gnss_global");
    logger->info("gnss_global_config_path={}", config_path);

    glim::Config config(config_path);
    gnss_topic = config.param<std::string>("gnss", "gnss_topic", "/pose_with_cov");
    prior_inf_scale = config.param<Eigen::Vector3d>("gnss", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 0.0));
    min_baseline = config.param<double>("gnss", "min_baseline", 5.0);

    transformation_initialized = false;
    T_odom_enu.setIdentity();

    kill_switch = false;
    thread = std::thread([this] { backend_task(); });

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&GNSSGlobal::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&GNSSGlobal::on_smoother_update, this, _1, _2, _3));
  }


  // -----------------------------------------------------------------

  std::vector<GenericTopicSubscription::Ptr> create_subscriptions(rclcpp::Node& node) override {

    // Ensure the node is properly initialized
    if (!node.get_node_base_interface()) {
      logger->error("Node base interface is not initialized.");
      throw std::runtime_error("Node base interface is not initialized.");
    }

    utm2gnss_pub_ = node.create_publisher<geometry_msgs::msg::TransformStamped>("~/utm2gnss", 10);

    // std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);

    const auto sub = std::make_shared<TopicSubscription<nav_msgs::msg::Odometry>>(gnss_topic, [this](const nav_msgs::msg::Odometry::ConstSharedPtr msg) { gnss_callback(msg); });
    return {sub};
  }


  geometry_msgs::msg::TransformStamped eigenToTransform(const Eigen::Isometry3d& T) {
    geometry_msgs::msg::TransformStamped t;
    t.transform.translation.x = T.translation().x();
    t.transform.translation.y = T.translation().y();
    t.transform.translation.z = T.translation().z();
  
    Eigen::Quaterniond q(T.rotation());
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();
  
    return t;
  }

  void publish_utm2gnss_transform(const std::string& frame_id, const std::string& child_frame_id, const Eigen::Isometry3d& transform) {
    if (utm2gnss_pub_) {
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped = eigenToTransform(transform);
      transform_stamped.header.stamp = rclcpp::Clock().now();
      transform_stamped.header.frame_id = frame_id;
      transform_stamped.child_frame_id = child_frame_id;
      utm2gnss_pub_->publish(transform_stamped);
    }
  }

    void broadcastTransform(const std::string& frame_id, const std::string& child_frame_id, const Eigen::Isometry3d& transform)
  {
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped = eigenToTransform(transform);
      transform_stamped.header.stamp = rclcpp::Clock().now();
      transform_stamped.header.frame_id = frame_id; // local enu
      transform_stamped.child_frame_id = child_frame_id; // map


      logger->info("broadcasting transform from {} to {}", frame_id, child_frame_id);
      logger->info("Transform details: translation = ({}, {}, {}), rotation = ({}, {}, {}, {})",
              transform_stamped.transform.translation.x,
              transform_stamped.transform.translation.y,
              transform_stamped.transform.translation.z,
              transform_stamped.transform.rotation.x,
              transform_stamped.transform.rotation.y,
              transform_stamped.transform.rotation.z,
              transform_stamped.transform.rotation.w);
      try {
          static_broadcaster_->sendTransform(transform_stamped);
      } catch (const std::exception& e) {
          logger->error("Failed to send transform: {}", e.what());
      }
      
  }

  // -----------------------------------------------------------------

  ~GNSSGlobal() {
    kill_switch = true;
    thread.join();
  }

  void gnss_callback(const nav_msgs::msg::Odometry::ConstSharedPtr& gnss_msg) {
    Eigen::Vector4d gnss_data;
    const double stamp = to_sec(gnss_msg->header.stamp);
    const auto& pos = gnss_msg->pose.pose.position;
    gnss_data << stamp, pos.x, pos.y, pos.z;
    input_gnss_queue.push_back(gnss_data);
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) { input_submap_queue.push_back(submap); }

  void on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
    const auto factors = output_factors.get_all_and_clear();
    if (!factors.empty()) {
      logger->debug("insert {} GNSS prior factors", factors.size());
      new_factors.add(factors);
    }
  }

  void backend_task() {
    logger->info("starting GNSS global thread");
    std::deque<Eigen::Vector4d> utm_queue;
    std::deque<SubMap::ConstPtr> submap_queue;
    Eigen::Isometry3d T_placeholder = Eigen::Isometry3d::Identity();
    broadcastTransform("local_enu", "odom", T_placeholder);


    while (!kill_switch) {
      // Convert GeoPoint(lat/lon) to UTM
      const auto gnss_data = input_gnss_queue.get_all_and_clear();
      utm_queue.insert(utm_queue.end(), gnss_data.begin(), gnss_data.end());

      // Add new submaps
      const auto new_submaps = input_submap_queue.get_all_and_clear();
      if (new_submaps.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }
      logger->info("new_submaps={}", new_submaps.size());

      submap_queue.insert(submap_queue.end(), new_submaps.begin(), new_submaps.end());

      // Remove submaps that are created earlier than the oldest GNSS data
      while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp < utm_queue.front()[0]) {
        submap_queue.pop_front();
      }

      logger->info("AFTER REMOVAL={}", submaps.size());


      // Interpolate UTM coords and associate with submaps
      while (!utm_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp > utm_queue.front()[0] &&
             submap_queue.front()->frames.back()->stamp < utm_queue.back()[0]) {

        logger->info("INTERPOLATE LOOP", new_submaps.size());

        const auto& submap = submap_queue.front();
        const double stamp = submap->frames[submap->frames.size() / 2]->stamp;

        const auto right = std::lower_bound(utm_queue.begin(), utm_queue.end(), stamp, [](const Eigen::Vector4d& utm, const double t) { return utm[0] < t; });
        if (right == utm_queue.end() || (right + 1) == utm_queue.end()) {
          logger->warn("invalid condition in GNSS global module!!");
          break;
        }
        const auto left = right - 1;
        logger->info("submap={:.6f} utm_left={:.6f} utm_right={:.6f}", stamp, (*left)[0], (*right)[0]);

        const double tl = (*left)[0];
        const double tr = (*right)[0];
        const double p = (stamp - tl) / (tr - tl);
        const Eigen::Vector4d interpolated = (1.0 - p) * (*left) + p * (*right);

        submaps.push_back(submap);
        submap_coords.push_back(interpolated);

        submap_queue.pop_front();
        utm_queue.erase(utm_queue.begin(), left);
      }

      // Initialize T_odom_enu
      logger->info("ABOUT TO INIT {} {} {}", !transformation_initialized, !submaps.empty(), 1);

      if (!transformation_initialized && !submaps.empty() && (submaps.front()->T_world_origin.inverse() * submaps.back()->T_world_origin).translation().norm() > min_baseline) {
        
        logger->info("initializing UTM to GNSS transformation");  
        
        Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_gnss = Eigen::Vector3d::Zero();
        for (int i = 0; i < submaps.size(); i++) {
          mean_est += submaps[i]->T_world_origin.translation();
          mean_gnss += submap_coords[i].tail<3>();
        }
        mean_est /= submaps.size();
        mean_gnss /= submaps.size();

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int i = 0; i < submaps.size(); i++) {
          const Eigen::Vector3d centered_est = submaps[i]->T_world_origin.translation() - mean_est;
          const Eigen::Vector3d centered_gnss = submap_coords[i].tail<3>() - mean_gnss;
          cov += centered_gnss * centered_est.transpose();
        }
        cov /= submaps.size();

        const Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov.block<2, 2>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix2d U = svd.matrixU();
        const Eigen::Matrix2d V = svd.matrixV();
        const Eigen::Matrix2d D = svd.singularValues().asDiagonal();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

        const double det = U.determinant() * V.determinant();
        if (det < 0.0) {
          S(1, 1) = -1;
        }

        Eigen::Isometry3d T_enu_odom = Eigen::Isometry3d::Identity();
        T_enu_odom.linear().block<2, 2>(0, 0) = U * S * V.transpose();
        T_enu_odom.translation() = mean_gnss - T_enu_odom.linear() * mean_est;
      

        T_odom_enu = T_enu_odom.inverse();

        broadcastTransform("local_enu", "odom", T_enu_odom);
        // publish_utm2gnss_transform("world", "utm", T_odom_enu);

        for (int i = 0; i < submaps.size(); i++) {
          const Eigen::Vector3d gnss = T_odom_enu * submap_coords[i].tail<3>();
          logger->debug("submap={} gnss={}", convert_to_string(submaps[i]->T_world_origin.translation().eval()), convert_to_string(gnss));
        }

        logger->info("T_odom_enu={}", convert_to_string(T_odom_enu));
        transformation_initialized = true;
      }

      logger->info("PRE PRIOR FACTOR");

      // Add translation prior factor
      if (transformation_initialized) {
        publish_utm2gnss_transform("world", "utm", T_odom_enu);

        const Eigen::Vector3d xyz = T_odom_enu * submap_coords.back().tail<3>();
        logger->debug("submap={} gnss={}", convert_to_string(submaps.back()->T_world_origin.translation().eval()), convert_to_string(xyz));

        const auto& submap = submaps.back();
        // note: should use a more accurate information matrix
        const auto model = gtsam::noiseModel::Isotropic::Information(prior_inf_scale.asDiagonal());
        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::PoseTranslationPrior<gtsam::Pose3>(X(submap->id), xyz, model));
        output_factors.push_back(factor);
      }
    }
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr utm2gnss_pub_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;


  std::atomic_bool kill_switch;
  std::thread thread;

  ConcurrentVector<Eigen::Vector4d> input_gnss_queue;
  ConcurrentVector<SubMap::ConstPtr> input_submap_queue;
  ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> output_factors;

  std::vector<SubMap::ConstPtr> submaps;
  std::vector<Eigen::Vector4d> submap_coords;

  std::string gnss_topic;
  Eigen::Vector3d prior_inf_scale;
  double min_baseline;

  bool transformation_initialized;
  Eigen::Isometry3d T_odom_enu;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}