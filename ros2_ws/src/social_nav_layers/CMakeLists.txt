cmake_minimum_required(VERSION 3.8)
project(social_nav_layers)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_library(${PROJECT_NAME} SHARED src/prediction_layer.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME} PUBLIC
  rclcpp::rclcpp nav2_costmap_2d::nav2_costmap_2d_core nav2_costmap_2d::layers
  pluginlib::pluginlib ${geometry_msgs_TARGETS} tf2::tf2 tf2_geometry_msgs::tf2_geometry_msgs)

pluginlib_export_plugin_description_file(nav2_costmap_2d plugins.xml)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES plugins.xml DESTINATION share/${PROJECT_NAME})

ament_package()
