cmake_minimum_required(VERSION 3.5)
project(ecn_mpc)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

file(GLOB scripts src/*)
install(PROGRAMS ${scripts} DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(${PROJECT_NAME})

ament_package()
