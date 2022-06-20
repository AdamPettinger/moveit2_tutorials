add_executable(ompl_constrained_planning
    src/ompl_constrained_planning_tutorial.cpp)
target_include_directories(ompl_constrained_planning
    PUBLIC include)
ament_target_dependencies(ompl_constrained_planning
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

add_executable(ompl_screw_planning
    src/ompl_screw_planning.cpp)
target_include_directories(ompl_screw_planning
    PUBLIC include)
ament_target_dependencies(ompl_screw_planning
    ${THIS_PACKAGE_INCLUDE_DEPENDS} Boost)

install(TARGETS ompl_constrained_planning ompl_screw_planning
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
