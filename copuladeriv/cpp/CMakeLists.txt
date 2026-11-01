cmake_minimum_required(VERSION 3.14)
project(CopulaDeriv CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Eigen3 OPTIONAL)

# Enable compiler warnings
if(NOT MSVC)
    add_compile_options("-Wall -Wextra -Wpedantic")
endif()

# Source files
file(GLOB SRC_FILES "src/*.cpp")
file(GLOB HEADER_FILES "include/copuladeriv/**/*.hpp")

add_executable(copuladeriv
    src/main.cpp
    src/pricing.cpp
    src/copula.cpp
    src/random.cpp
    src/payoffs.cpp
)

# Include directories
target_include_paths(copuladeriv PRIVATE
    include
    include/copuladeriv
)

# Link dependencies
if(EIGEN3_FOUND)
    target_link_libraries(copuladeriv PRIVATE Eigen3::Eigen)
endif()

# Test configuration (GoogleTest if available)
find_package(GTest OPTIONAL)
if(GTest_FOUND)
    enable_testing()
    add_test(NAME copuladeriv_basic
        WORKDIR .
        ./copuladeriv --help)
endif()

# Install target
install(TARGETS copuladeriv DESTINATION bin)