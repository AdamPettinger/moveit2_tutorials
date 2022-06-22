#include <chrono>
#include <queue>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <std_msgs/msg/color_rgba.hpp>
#include <affordance_primitives/msg_types.hpp>
#include <affordance_primitives/screw_model/screw_axis.hpp>
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
  affordance_primitives::APScrewExecutor screw_exec(node);
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

    moveit_visual_tools.publishArrow(pose_to_display);
    moveit_visual_tools.trigger();
    return pose_to_display;
  };

  // Uses the current pose and a screw axis to find a target pose. All inputs are given in the planning frame
  auto get_target_pose = [&moveit_visual_tools, &screw_exec, &move_group_interface](
                             const affordance_primitives::ScrewStamped& screw,
                             const geometry_msgs::msg::PoseStamped& current_pose, const double theta) {
    geometry_msgs::msg::PoseStamped target_pose = current_pose;

    // Convert start pose to Eigen
    Eigen::Isometry3d tf_planning_to_current;
    tf2::fromMsg(current_pose.pose, tf_planning_to_current);

    // We need to convert the screw axis to be in the current frame instead of the planning. Make a tf message
    geometry_msgs::msg::TransformStamped tfmsg_planning_to_current;
    tfmsg_planning_to_current.header.frame_id = current_pose.header.frame_id;
    tfmsg_planning_to_current.child_frame_id = "start_pose";
    tfmsg_planning_to_current.transform.rotation = current_pose.pose.orientation;
    tfmsg_planning_to_current.transform.translation.x = current_pose.pose.position.x;
    tfmsg_planning_to_current.transform.translation.y = current_pose.pose.position.y;
    tfmsg_planning_to_current.transform.translation.z = current_pose.pose.position.z;

    // Convert the screw message
    const auto tfed_screw_message = affordance_primitives::transformScrew(screw, tfmsg_planning_to_current);

    // Create a screw axis from the message
    affordance_primitives::ScrewAxis screw_axis;
    screw_axis.setScrewAxis(tfed_screw_message);

    // Now calculate the target pose
    Eigen::Isometry3d tf_planning_to_target = tf_planning_to_current * screw_axis.getTF(theta);
    target_pose.pose = tf2::toMsg(tf_planning_to_target);

    // Visualize the start and end poses
    moveit_visual_tools.publishSphere(current_pose.pose, rviz_visual_tools::RED, 0.05);
    moveit_visual_tools.publishSphere(target_pose.pose, rviz_visual_tools::GREEN, 0.05);

    // Get path waypoints for visualization
    affordance_primitives::AffordancePrimitiveGoal ap_goal;

    // Set AP Goal with tf
    geometry_msgs::msg::TransformStamped tf_ee_to_task = tf2::eigenToTransform(tf_planning_to_current.inverse());
    tf_ee_to_task.header.frame_id = move_group_interface.getEndEffectorLink();
    tf_ee_to_task.child_frame_id = move_group_interface.getPlanningFrame();
    ap_goal.moving_frame_source = ap_goal.PROVIDED;
    ap_goal.moving_to_task_frame = tf_ee_to_task;

    ap_goal.theta_dot = 0.2;
    ap_goal.screw_distance = theta;
    ap_goal.screw = screw;

    // Get waypoints and visualize
    auto waypoints = screw_exec.getTrajectoryCommands(ap_goal, 10);
    if (waypoints.has_value())
    {
      EigenSTL::vector_Isometry3d waypoints_vec;
      for (auto& wp : waypoints->trajectory)
      {
        Eigen::Isometry3d this_wp;
        tf2::fromMsg(wp.pose, this_wp);
        waypoints_vec.push_back(this_wp);
      }
      // moveit_visual_tools.publishPath(waypoints_vec, rviz_visual_tools::RED, rviz_visual_tools::XSMALL, "Screw path");
      moveit_visual_tools.publishAxisPath(waypoints_vec, rviz_visual_tools::SMALL, "Screw path");
    }
    else
    {
      RCLCPP_ERROR(LOGGER, "\n\n\nWAYPOINTS NOT FOUND\n\n\n");
    }

    moveit_visual_tools.trigger();
    return target_pose;
  };

  // Converts the screw message to a constraints message
  auto get_constraint_msg = [&move_group_interface](const geometry_msgs::msg::Pose& screw_pose,
                                                    const geometry_msgs::msg::Pose& current_pose) {
    moveit_msgs::msg::Constraints constraints_msg;

    moveit_msgs::msg::PositionConstraint pos_constraint;
    pos_constraint.header.frame_id = move_group_interface.getPoseReferenceFrame();
    pos_constraint.link_name = move_group_interface.getEndEffectorLink();
    pos_constraint.weight = 1.0;

    // TODO: use the dimensions to set the pitch
    shape_msgs::msg::SolidPrimitive solid_primitive;
    solid_primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    solid_primitive.dimensions = { 1.0, 1.0, 1.0 };
    pos_constraint.constraint_region.primitives.emplace_back(solid_primitive);

    // Add screw then starting pose to message
    pos_constraint.constraint_region.primitive_poses.emplace_back(screw_pose);
    pos_constraint.constraint_region.primitive_poses.emplace_back(current_pose);

    // Return the whole message
    constraints_msg.position_constraints.emplace_back(pos_constraint);
    constraints_msg.name = "screw_constraint";
    return constraints_msg;
  };

  // Runs one constraint test
  auto plan_one_constraint = [get_target_pose, get_screw_pose, get_constraint_msg,
                              &move_group_interface](const affordance_primitives::ScrewStamped& screw_constraint) {
    const auto current_pose = move_group_interface.getCurrentPose();
    const auto screw_pose = get_screw_pose(screw_constraint);

    // Figure out the end pose (target)
    const double theta = 0.5 * M_PI;
    const auto target_pose = get_target_pose(screw_constraint, current_pose, theta);

    // Get the constraint message
    const auto constraints_msg = get_constraint_msg(screw_pose, current_pose.pose);

    // Plan
    move_group_interface.setPathConstraints(constraints_msg);
    move_group_interface.setPoseTarget(target_pose);
    move_group_interface.setPlanningTime(10.0);

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

  constraint.origin.x += 0.1;
  screw_constraints.push(constraint);

  while (screw_constraints.size() > 0 && rclcpp::ok())
  {
    moveit_visual_tools.prompt(
        "Press 'next' in the RvizVisualToolsGui window to continue to the linear constraint example");

    reset_demo();

    plan_one_constraint(screw_constraints.front());
    screw_constraints.pop();
  }

  // Done!
  moveit_visual_tools.prompt("Press 'Next' in the RvizVisualToolsGui window to clear the markers");
  moveit_visual_tools.deleteAllMarkers();
  moveit_visual_tools.trigger();
  move_group_interface.clearPathConstraints();

  rclcpp::shutdown();
  spinner.join();
  return 0;
}
