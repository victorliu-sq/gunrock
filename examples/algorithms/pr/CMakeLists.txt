# begin /* Set the application name. */
set(APPLICATION_NAME pr)
# end /* Set the application name. */

# begin /* Add CUDA executables */
add_executable(${APPLICATION_NAME})

set(SOURCE_LIST 
    ${APPLICATION_NAME}.cu
)

target_sources(${APPLICATION_NAME} PRIVATE ${SOURCE_LIST})
target_link_libraries(${APPLICATION_NAME} PRIVATE essentials)
get_target_property(ESSENTIALS_ARCHITECTURES essentials CUDA_ARCHITECTURES)
set_target_properties(${APPLICATION_NAME} 
    PROPERTIES 
        CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES}
) # XXX: Find a better way to inherit essentials properties.

message(STATUS "Example Added: ${APPLICATION_NAME}")
# end /* Add CUDA executables */
# begin /* pr_cusparse: PageRank's pull as cuSPARSE's SpMV, the GPU reference (GACGE) */
add_executable(pr_cusparse pr_cusparse.cu)
target_link_libraries(pr_cusparse PRIVATE essentials cusparse)
set_target_properties(pr_cusparse PROPERTIES CUDA_ARCHITECTURES ${ESSENTIALS_ARCHITECTURES})
message(STATUS "Example Added: pr_cusparse")
# end /* pr_cusparse */
