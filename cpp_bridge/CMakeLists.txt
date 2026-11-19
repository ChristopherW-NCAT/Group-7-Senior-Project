cmake_minimum_required(VERSION 3.16)
project(neurolens_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Standalone stdin/stdout bridge executable
add_executable(neurolens_bridge_cli src/neurolens_bridge_node.cpp)

# ROS 2 node executable (built when ament_cmake is available)
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(std_msgs REQUIRED)

  add_executable(neurolens_bridge_node src/neurolens_bridge_node.cpp)
  target_compile_definitions(neurolens_bridge_node PRIVATE USE_ROS2)
  ament_target_dependencies(neurolens_bridge_node
    rclcpp
    sensor_msgs
    std_msgs
  )

  install(TARGETS
    neurolens_bridge_cli
    neurolens_bridge_node
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_package()
endif()
