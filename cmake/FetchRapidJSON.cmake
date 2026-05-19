set(RAPIDJSON_SOURCE_DIR "${PROJECT_DEPS_DIR}/rapidjson-src")
if(NOT EXISTS "${RAPIDJSON_SOURCE_DIR}")
  message(FATAL_ERROR "Missing RapidJSON source directory: ${RAPIDJSON_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: RapidJSON at ${RAPIDJSON_SOURCE_DIR}")
set(RAPIDJSON_INCLUDE_DIR "${RAPIDJSON_SOURCE_DIR}/include")
