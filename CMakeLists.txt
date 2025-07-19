# Specify the minimum version of CMake required.
cmake_minimum_required(VERSION 3.10)

# Set the project name and language.
project(LinearRegression LANGUAGES CXX)

# Set the C++ standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable target and list all the .cpp source files.
# CMake will automatically find the corresponding headers.
add_executable(
    main
    main.cpp
    linear_regression.cpp
)

# Optional: Print a helpful message after the build finishes.
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build complete. Run with: ./build/main"
)