set(NVBENCH_SOURCE_DIR "${PROJECT_DEPS_DIR}/nvbench-src")
set(NVBENCH_BUILD_DIR "${PROJECT_DEPS_DIR}/nvbench-build")

if(NOT EXISTS "${NVBENCH_SOURCE_DIR}")
  message(FATAL_ERROR "Missing NVBench source directory: ${NVBENCH_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: NVBench at ${NVBENCH_SOURCE_DIR}")
if(NOT TARGET nvbench::main)
  add_subdirectory("${NVBENCH_SOURCE_DIR}" "${NVBENCH_BUILD_DIR}" EXCLUDE_FROM_ALL)
endif()

set(NVBENCH_INCLUDE_DIR "${NVBENCH_SOURCE_DIR}")
