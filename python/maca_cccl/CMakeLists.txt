cmake_minimum_required(VERSION 3.21...3.31 FATAL_ERROR)
project(maca_cccl_python LANGUAGES CXX)

# Find Python and Cython
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Cython REQUIRED)

# Set MACA paths
set(MACA_PATH "/opt/maca" CACHE PATH "Path to MACA installation")
if(NOT EXISTS ${MACA_PATH})
    set(MACA_PATH "/mnt/data/minxi/maca/maca-sdk-20250707-586/maca_install/opt/maca-20250707")
endif()

message(STATUS "Using MACA at: ${MACA_PATH}")

# Find MACA libraries
find_library(MACA_RUNTIME_LIB mcruntime PATHS ${MACA_PATH}/lib ${MACA_PATH}/lib64 REQUIRED)
find_library(MACA_RTC_LIB mcrtc PATHS ${MACA_PATH}/lib ${MACA_PATH}/lib64 REQUIRED)

# Set mcCub path
set(MCCUB_PATH "/mnt/data/minxi/1_15/mcCub" CACHE PATH "Path to mcCub")

# Detect Cython version
execute_process(
    COMMAND ${Python_EXECUTABLE} -m cython --version
    OUTPUT_VARIABLE CYTHON_VERSION_OUTPUT
    ERROR_VARIABLE CYTHON_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Cython version: ${CYTHON_VERSION_OUTPUT}")

# Define Cython source file
set(CYTHON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/maca/compute/_bindings_impl.pyx")
set(CYTHON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/maca/compute/_bindings_impl.cpp")

# Cythonize command
add_custom_command(
    OUTPUT ${CYTHON_OUTPUT}
    COMMAND ${Python_EXECUTABLE} -m cython
        --cplus
        -3
        -o ${CYTHON_OUTPUT}
        ${CYTHON_SOURCE}
    DEPENDS ${CYTHON_SOURCE}
    COMMENT "Cythonizing ${CYTHON_SOURCE}"
    VERBATIM
)

# Create Python extension module
Python_add_library(_bindings_impl MODULE ${CYTHON_OUTPUT})

# Include directories
target_include_directories(_bindings_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../parallel/include
    ${MACA_PATH}/include
    ${MCCUB_PATH}
    ${Python_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(_bindings_impl PRIVATE
    ${MACA_RUNTIME_LIB}
    ${MACA_RTC_LIB}
)

# Set C++ standard
target_compile_features(_bindings_impl PRIVATE cxx_std_17)

# Set RPATH for runtime library location
set_target_properties(_bindings_impl PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
)

# Install the extension module
install(TARGETS _bindings_impl
    LIBRARY DESTINATION maca/compute
)

# Install Python files
install(DIRECTORY maca/
    DESTINATION maca
    FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "__pycache__" EXCLUDE
)
