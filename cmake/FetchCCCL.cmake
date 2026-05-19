set(CCCL_SOURCE_DIR "${PROJECT_DEPS_DIR}/cccl-src")
if(NOT EXISTS "${CCCL_SOURCE_DIR}")
  message(FATAL_ERROR "Missing CCCL source directory: ${CCCL_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: Thrust at ${CCCL_SOURCE_DIR}")
set(CCCL_INCLUDE_DIR "${CCCL_SOURCE_DIR}")
