cmake_minimum_required(VERSION 3.3)
project(glim_rviz_viewer)

find_package(glim REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Iridescence REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(spdlog REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)


add_library(glim_rviz_viewer SHARED
  src/glim_ext/glim_rviz_viewer.cpp
)

target_include_directories(glim_rviz_viewer PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${glim_INCLUDE_DIRS}
  ${glim_ext_INCLUDE_DIRS}
  ${Iridescence_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${tf2_eigen_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${spdlog_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}

)

target_link_libraries(glim_rviz_viewer
  ${glim_LIBRARIES}
  ${glim_ext_LIBRARIES}
  ${Iridescence_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
  ${rclcpp_LIBRARIES}
  ${nav_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${spdlog_LIBRARIES}
  ${pcl_ros_LIBRARIES}
  ${pcl_conversions_LIBRARIES}

)

target_compile_options(glim_rviz_viewer PRIVATE
  -std=c++17
)