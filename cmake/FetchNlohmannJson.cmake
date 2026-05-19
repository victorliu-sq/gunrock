set(NHLOMANN_JSON_SOURCE_DIR "${PROJECT_DEPS_DIR}/json-src")
if(NOT EXISTS "${NHLOMANN_JSON_SOURCE_DIR}")
  message(FATAL_ERROR "Missing nlohmann_json source directory: ${NHLOMANN_JSON_SOURCE_DIR}. Run ./deps/_scripts/gunrock/install.sh first.")
endif()

message(STATUS "Using External Project: NLohmannJson at ${NHLOMANN_JSON_SOURCE_DIR}")
set(NHLOMANN_JSON_INCLUDE_DIR "${NHLOMANN_JSON_SOURCE_DIR}/include")
