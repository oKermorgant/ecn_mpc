cmake_minimum_required(VERSION 3.5)
project(ecn_mpc)

find_package(ament_cmake_auto REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

file(GLOB scripts scripts/*)
install(PROGRAMS ${scripts} DESTINATION lib/${PROJECT_NAME})

file(GLOB scripts src/*.py)
install(PROGRAMS ${scripts} DESTINATION lib/${PROJECT_NAME})

ament_auto_add_executable(acados_test src/acados.cpp)

ament_python_install_package(${PROJECT_NAME})

ament_auto_package()
