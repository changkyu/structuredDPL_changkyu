####################################################################
# Path Setting
SET(EXECUTABLE_OUTPUT_PATH ../bin)

####################################################################
# Compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

####################################################################
# Includes
include_directories(include)

####################################################################
# Sources
add_executable(sdpl ../src/main.cpp)
