set(MODERNGPU_SOURCE_DIR "${PROJECT_DEPS_DIR}/moderngpu-src")
if(NOT EXISTS "${MODERNGPU_SOURCE_DIR}")
  message(FATAL_ERROR "Missing ModernGPU source directory: ${MODERNGPU_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: ModernGPU at ${MODERNGPU_SOURCE_DIR}")
set(MODERNGPU_INCLUDE_DIR "${MODERNGPU_SOURCE_DIR}/src")
