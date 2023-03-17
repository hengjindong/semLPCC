cmake_minimum_required(VERSION 3.5)
project(pcl_pcc)
 
find_package(PCL 1.2 REQUIRED)
 
# 加入Boost setting
find_package(Boost COMPONENTS program_options REQUIRED )
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
 
add_executable(pcl_pcc pcl_pcc.cpp)
 
target_link_libraries (pcl_pcc ${PCL_LIBRARIES} ${Boost_LIBRARIES}) #此处也有修改
 
install(TARGETS pcl_pcc RUNTIME DESTINATION bin)

