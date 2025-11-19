cmake_minimum_required(VERSION 3.5)

project(qjr7_9000_description)

find_package(ament_cmake REQUIRED)

install(
    DIRECTORY meshes
    DESTINATION share/${PROJECT_NAME}
)

install(
    FILES qjr7_9000_description.urdf
    DESTINATION share/${PROJECT_NAME}
)

ament_package()