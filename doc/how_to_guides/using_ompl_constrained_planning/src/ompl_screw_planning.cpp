#include <chrono>
#include <queue>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <std_msgs/msg/color_rgba.hpp>
#include <affordance_primitives/msg_types.hpp>
#include <affordance_primitives/screw_model/screw_execution.hpp>

static const auto LOGGER = rclcpp::get_logger("ompl_screw_planning");

int main(int argc, char** argv)
{
  using namespace std::chrono_literals;
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);
  auto node = rclcpp::Node::make_shared("ompl_screw_planning_node", node_options);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  auto spinner = std::thread([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface move_group_interface(node, "panda_arm");
  auto moveit_visual_tools =
      moveit_visual_tools::MoveItVisualTools{ node, "panda_link0", rviz_visual_tools::RVIZ_MARKER_TOPIC,
                                              move_group_interface.getRobotModel() };

  rclcpp::sleep_for(1s);

  // Create some helpful lambdas
  auto current_pose = move_group_interface.getCurrentPose();

  // Creates a pose at a given positional offset from the current pose
  auto get_relative_pose = [current_pose, &moveit_visual_tools](double x, double y, double z) {
    auto target_pose = current_pose;
    target_pose.pose.position.x += x;
    target_pose.pose.position.y += y;
    target_pose.pose.position.z += z;
    moveit_visual_tools.publishSphere(current_pose.pose, rviz_visual_tools::RED, 0.05);
    moveit_visual_tools.publishSphere(target_pose.pose, rviz_visual_tools::GREEN, 0.05);
    moveit_visual_tools.trigger();
    return target_pose;
  };

  // Resets the demo by cleaning up any constraints and markers
  auto reset_demo = [&move_group_interface, &moveit_visual_tools]() {
    move_group_interface.clearPathConstraints();
    move_group_interface.clearPoseTarget();
    moveit_visual_tools.deleteAllMarkers();
    moveit_visual_tools.trigger();
  };

  // Converts a screw message to a pose with the x-axis pointing the direction of the screw's axis
  // Returns geometry_msgs::msg::Pose
  auto get_screw_pose = [&moveit_visual_tools](const affordance_primitives::ScrewStamped& screw) {
    geometry_msgs::msg::Pose pose_to_display;  // Arrow will be along x-axis of this pose
    pose_to_display.position = screw.origin;

    // Set the x-axis of the pose to the axis of the screw
    Eigen::Vector3d x_axis(screw.axis.x, screw.axis.y, screw.axis.z);
    x_axis.normalize();

    // Y-axis is orthonormal to x-axis, found using Gram-Schimdt process
    Eigen::Vector3d y_axis(0.25, 0.25, 0.5);
    y_axis = (y_axis - (x_axis.dot(y_axis) / x_axis.dot(x_axis)) * x_axis).normalized();

    // Z-axis is the cross product of the other two
    Eigen::Vector3d z_axis = (x_axis.cross(y_axis)).normalized();

    // Stuff into a rotation matrix and convert to quaternion
    Eigen::Matrix3d rot_mat;
    rot_mat.col(0) = x_axis;
    rot_mat.col(1) = y_axis;
    rot_mat.col(2) = z_axis;
    Eigen::Quaterniond rot_quat(rot_mat);
    pose_to_display.orientation = tf2::toMsg(rot_quat);

    return pose_to_display;
  };

  // Runs one constraint test
  auto plan_one_constraint = [get_relative_pose, get_screw_pose, &move_group_interface,
                              &moveit_visual_tools](const affordance_primitives::ScrewStamped& screw_constraint) {
    auto current_pose = move_group_interface.getCurrentPose();
    auto screw_pose = get_screw_pose(screw_constraint);

    // We can also plan along a line. We can use the same pose as last time.
    Eigen::Vector3d relative_translation(0.3, 0, 0);
    Eigen::Isometry3d screw_pose_eig;
    tf2::fromMsg(screw_pose, screw_pose_eig);
    relative_translation = screw_pose_eig.linear() * relative_translation;

    auto target_pose = get_relative_pose(relative_translation.x(), relative_translation.y(), relative_translation.z());

    // Building on the previous constraint, we can make it a line, by also reducing the dimension of the box in the x-direction.
    moveit_msgs::msg::PositionConstraint line_constraint;
    line_constraint.header.frame_id = move_group_interface.getPoseReferenceFrame();
    line_constraint.link_name = move_group_interface.getEndEffectorLink();
    shape_msgs::msg::SolidPrimitive line;
    line.type = shape_msgs::msg::SolidPrimitive::BOX;
    line.dimensions = { 1.0, 0.0005, 0.0005 };
    line_constraint.constraint_region.primitives.emplace_back(line);

    geometry_msgs::msg::Pose line_pose;
    line_pose.position = current_pose.pose.position;
    line_pose.orientation = screw_pose.orientation;
    line_constraint.constraint_region.primitive_poses.emplace_back(line_pose);
    line_constraint.weight = 1.0;

    // Visualize the constraint
    moveit_visual_tools.publishLine(current_pose.pose.position, target_pose.pose.position,
                                    rviz_visual_tools::TRANSLUCENT_DARK);
    moveit_visual_tools.publishArrow(screw_pose);
    moveit_visual_tools.trigger();

    moveit_msgs::msg::Constraints line_constraints;
    line_constraints.position_constraints.emplace_back(line_constraint);
    line_constraints.name = "use_equality_constraints";
    move_group_interface.setPathConstraints(line_constraints);
    move_group_interface.setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto success = (move_group_interface.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    RCLCPP_INFO(LOGGER, "Plan with line constraint %s", success ? "" : "FAILED");
  };

  reset_demo();
  moveit_visual_tools.loadRemoteControl();

  // This will be our set of constrained planning attempts
  std::queue<affordance_primitives::ScrewStamped> screw_constraints;
  affordance_primitives::ScrewStamped constraint;
  constraint.header.frame_id = move_group_interface.getPlanningFrame();

  // Set some interesting constraints
  constraint.origin = current_pose.pose.position;
  constraint.axis.x = 1;
  screw_constraints.push(constraint);

  constraint.axis.x = -1;
  screw_constraints.push(constraint);

  constraint.axis.x = 0;
  constraint.axis.z = 1;
  screw_constraints.push(constraint);

  constraint.axis.x = -1;
  screw_constraints.push(constraint);

  while (screw_constraints.size() > 0 && rclcpp::ok())
  {
    moveit_visual_tools.prompt(
        "Press 'next' in the RvizVisualToolsGui window to continue to the linear constraint example");

    reset_demo();

    plan_one_constraint(screw_constraints.front());
    screw_constraints.pop();
  }

  constraint.origin.x += 0.25;
  screw_constraints.push(constraint);

  affordance_primitives::APScrewExecutor screw_exec(node);
  affordance_primitives::AffordancePrimitiveGoal ap_goal;

  geometry_msgs::msg::TransformStamped tf_ee_to_task;
  tf_ee_to_task.header.frame_id = move_group_interface.getEndEffectorLink();
  tf_ee_to_task.child_frame_id = move_group_interface.getPlanningFrame();
  tf_ee_to_task.transform.translation.x = -1 * current_pose.pose.position.x;
  tf_ee_to_task.transform.translation.y = -1 * current_pose.pose.position.y;
  tf_ee_to_task.transform.translation.z = -1 * current_pose.pose.position.z;

  ap_goal.moving_frame_source = ap_goal.PROVIDED;
  ap_goal.moving_to_task_frame = tf_ee_to_task;

  ap_goal.screw = screw_constraints.front();
  ap_goal.theta_dot = 0.2;
  ap_goal.screw_distance = 0.25 * M_PI;

  auto waypoints = screw_exec.getTrajectoryCommands(ap_goal, 10);
  EigenSTL::vector_Isometry3d test;
  for (auto& wp : waypoints->trajectory)
  {
    Eigen::Isometry3d this_wp;
    tf2::fromMsg(wp.pose, this_wp);
    test.push_back(this_wp);
  }
  auto screw_pose = get_screw_pose(screw_constraints.front());
  moveit_visual_tools.publishArrow(screw_pose);
  moveit_visual_tools.publishPath(test, rviz_visual_tools::RED, rviz_visual_tools::XSMALL, "Screw path");
  moveit_visual_tools.publishAxisPath(test, rviz_visual_tools::SMALL, "Screw path");
  moveit_visual_tools.trigger();

  // Done!
  moveit_visual_tools.prompt("Press 'Next' in the RvizVisualToolsGui window to clear the markers");
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.trigger();
  move_group_interface.clearPathConstraints();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
