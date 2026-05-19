set(GOOGLETEST_SOURCE_DIR "${PROJECT_DEPS_DIR}/googletest-src")
set(GOOGLETEST_BUILD_DIR "${PROJECT_DEPS_DIR}/googletest-build")

if(NOT EXISTS "${GOOGLETEST_SOURCE_DIR}")
  message(FATAL_ERROR "Missing GoogleTest source directory: ${GOOGLETEST_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: GoogleTests at ${GOOGLETEST_SOURCE_DIR}")
if(NOT TARGET gtest_main)
  add_subdirectory("${GOOGLETEST_SOURCE_DIR}" "${GOOGLETEST_BUILD_DIR}" EXCLUDE_FROM_ALL)
endif()

if(TARGET gtest_main AND NOT TARGET gtest::main)
  add_library(gtest::main ALIAS gtest_main)
endif()
