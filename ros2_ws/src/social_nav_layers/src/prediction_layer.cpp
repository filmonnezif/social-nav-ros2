// Nav2 costmap layer that paints where people are predicted to walk.
//
// Subscribes to a geometry_msgs/PoseArray of predicted positions (any frame
// tf can resolve). Each pose is one future point of one person; position.z
// carries a weight in [0, 1] (nearer future = higher weight). Every update the
// layer paints a disc of `radius` around each point with cost
// weight * max_cost, keeping the higher of that and the existing cost. Costs
// stay below INSCRIBED so the planner avoids predicted paths but can still
// cross them when there is no other way. The layer sits after the inflation
// layer, so predicted cells are not inflated a second time.

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace social_nav_layers
{

class PredictionLayer : public nav2_costmap_2d::Layer
{
public:
  void onInitialize() override
  {
    auto node = node_.lock();
    declareParameter("enabled", rclcpp::ParameterValue(true));
    declareParameter("topic", rclcpp::ParameterValue(std::string("/predicted_people")));
    declareParameter("radius", rclcpp::ParameterValue(0.45));
    declareParameter("max_cost", rclcpp::ParameterValue(200));
    declareParameter("max_age", rclcpp::ParameterValue(1.0));
    node->get_parameter(name_ + ".enabled", enabled_);
    node->get_parameter(name_ + ".topic", topic_);
    node->get_parameter(name_ + ".radius", radius_);
    node->get_parameter(name_ + ".max_cost", max_cost_);
    node->get_parameter(name_ + ".max_age", max_age_);
    max_cost_ = std::clamp(max_cost_, 1, static_cast<int>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) - 1);

    sub_ = node->create_subscription<geometry_msgs::msg::PoseArray>(
      topic_, rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::PoseArray::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = msg;
      });
    current_ = true;
    RCLCPP_INFO(logger_, "PredictionLayer on %s (radius %.2f m, max cost %d)",
      topic_.c_str(), radius_, max_cost_);
  }

  void reset() override {points_.clear();}
  bool isClearable() override {return true;}

  void updateBounds(
    double, double, double, double * min_x, double * min_y,
    double * max_x, double * max_y) override
  {
    if (!enabled_) {return;}
    // also cover last cycle's points, so their marks are reset this cycle
    for (const auto & p : points_) {
      touch(p.x, p.y, min_x, min_y, max_x, max_y);
    }
    points_.clear();

    geometry_msgs::msg::PoseArray::ConstSharedPtr msg;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      msg = latest_;
    }
    if (!msg) {return;}
    const auto now = clock_->now();
    if ((now - rclcpp::Time(msg->header.stamp)).seconds() > max_age_) {return;}

    const std::string frame = layered_costmap_->getGlobalFrameID();
    tf2::Transform tf;
    tf.setIdentity();
    if (msg->header.frame_id != frame) {
      try {
        auto t = tf_->lookupTransform(frame, msg->header.frame_id, tf2::TimePointZero);
        tf2::fromMsg(t.transform, tf);
      } catch (const tf2::TransformException & e) {
        RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "PredictionLayer: %s", e.what());
        return;
      }
    }
    for (const auto & pose : msg->poses) {
      tf2::Vector3 v = tf * tf2::Vector3(pose.position.x, pose.position.y, 0.0);
      Point p{v.x(), v.y(), std::clamp(pose.position.z, 0.0, 1.0)};
      points_.push_back(p);
      touch(p.x, p.y, min_x, min_y, max_x, max_y);
    }
  }

  void updateCosts(
    nav2_costmap_2d::Costmap2D & grid, int min_i, int min_j, int max_i, int max_j) override
  {
    if (!enabled_) {return;}
    const double res = grid.getResolution();
    const int r_cells = static_cast<int>(std::ceil(radius_ / res));
    for (const auto & p : points_) {
      unsigned int cx, cy;
      if (!grid.worldToMap(p.x, p.y, cx, cy)) {continue;}
      const auto cost = static_cast<unsigned char>(std::lround(p.w * max_cost_));
      for (int dy = -r_cells; dy <= r_cells; ++dy) {
        for (int dx = -r_cells; dx <= r_cells; ++dx) {
          if (dx * dx + dy * dy > r_cells * r_cells) {continue;}
          const int i = static_cast<int>(cx) + dx, j = static_cast<int>(cy) + dy;
          if (i < min_i || j < min_j || i >= max_i || j >= max_j) {continue;}
          const unsigned char old = grid.getCost(i, j);
          if (old == nav2_costmap_2d::NO_INFORMATION || old < cost) {grid.setCost(i, j, cost);}
        }
      }
    }
  }

private:
  struct Point {double x, y, w;};

  void touch(double x, double y, double * min_x, double * min_y, double * max_x, double * max_y)
  {
    *min_x = std::min(*min_x, x - radius_);
    *min_y = std::min(*min_y, y - radius_);
    *max_x = std::max(*max_x, x + radius_);
    *max_y = std::max(*max_y, y + radius_);
  }

  std::string topic_;
  double radius_{0.45};
  int max_cost_{200};
  double max_age_{1.0};
  std::vector<Point> points_;
  std::mutex mutex_;
  geometry_msgs::msg::PoseArray::ConstSharedPtr latest_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr sub_;
};

}  // namespace social_nav_layers

PLUGINLIB_EXPORT_CLASS(social_nav_layers::PredictionLayer, nav2_costmap_2d::Layer)
