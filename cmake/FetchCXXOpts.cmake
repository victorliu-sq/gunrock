set(CXXOPTS_SOURCE_DIR "${PROJECT_DEPS_DIR}/cxxopts-src")
if(NOT EXISTS "${CXXOPTS_SOURCE_DIR}")
  message(FATAL_ERROR "Missing CXXOPTS source directory: ${CXXOPTS_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: CXXOPTS at ${CXXOPTS_SOURCE_DIR}")
set(CXXOPTS_INCLUDE_DIR "${CXXOPTS_SOURCE_DIR}/include")
