set(CMAKE_MODULES_SOURCE_DIR "${PROJECT_DEPS_DIR}/cmake_modules-src")
if(NOT EXISTS "${CMAKE_MODULES_SOURCE_DIR}")
  message(FATAL_ERROR "Missing CMake Modules source directory: ${CMAKE_MODULES_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: CMake Modules at ${CMAKE_MODULES_SOURCE_DIR}")
set(CMAKE_MODULES_INCLUDE_DIR "${CMAKE_MODULES_SOURCE_DIR}")
