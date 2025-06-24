cmake_minimum_required(VERSION 3.10)
project(ORB_SLAM3)

# ───────────────────────────────────────────────────────────
# Build-type & compiler flags
# ───────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS         "${CMAKE_C_FLAGS}   -Wall -O3")
set(CMAKE_CXX_FLAGS       "-std=c++14 -Wall -fopenmp -march=native -pthread")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ───────────────────────────────────────────────────────────
# Third-party packages
# ───────────────────────────────────────────────────────────
# OpenCV 3.4.13 (path may differ on your system)
set(OpenCV_DIR "~/Workspaces/PhD_ws/src/Prerequisites/OpenCV34/install/lib/cmake/opencv3")
find_package(OpenCV REQUIRED)

# LibTorch
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/Thirdparty/libtorch)
find_package(Torch REQUIRED)

# Other deps
find_package(Eigen3    3.1.0 REQUIRED)
find_package(Pangolin  REQUIRED)
find_package(CUDA      REQUIRED)
find_package(yaml-cpp  REQUIRED)
find_package(Boost     REQUIRED)
find_package(Gflags    REQUIRED)
find_package(Glog      REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    /home/anast/Requirements/onnxruntime-linux-x64-gpu-1.16.3/include
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ───────────────────────────────────────────────────────────
# Core library
# ───────────────────────────────────────────────────────────
add_library(${PROJECT_NAME} SHARED
    src/Random.cpp               include/Random.h
    src/Timestamp.cpp            include/Timestamp.h
    src/System.cc                include/System.h
    src/Tracking.cc              include/Tracking.h
    src/LocalMapping.cc          include/LocalMapping.h
    src/LoopClosing.cc           include/LoopClosing.h
    src/ORBextractor.cc          include/ORBextractor.h
    src/ORBmatcher.cc            include/ORBmatcher.h
    src/FrameDrawer.cc           include/FrameDrawer.h
    src/Converter.cc             include/Converter.h
    src/MapPoint.cc              include/MapPoint.h
    src/KeyFrame.cc              include/KeyFrame.h
    src/Atlas.cc                 include/Atlas.h
    src/Map.cc                   include/Map.h
    src/MapDrawer.cc             include/MapDrawer.h
    src/Optimizer.cc             include/Optimizer.h
    src/Frame.cc                 include/Frame.h
    src/KeyFrameDatabase.cc      include/KeyFrameDatabase.h
    src/Sim3Solver.cc            include/Sim3Solver.h
    src/Viewer.cc                include/Viewer.h
    src/ImuTypes.cc              include/ImuTypes.h
    src/G2oTypes.cc              include/G2oTypes.h
    src/CameraModels/Pinhole.cpp include/CameraModels/Pinhole.h
    src/CameraModels/KannalaBrandt8.cpp include/CameraModels/KannalaBrandt8.h
    src/OptimizableTypes.cpp     include/OptimizableTypes.h
    src/MLPnPsolver.cpp          include/MLPnPsolver.h
    src/GeometricTools.cc        include/GeometricTools.h
    src/TwoViewReconstruction.cc include/TwoViewReconstruction.h
    src/Config.cc                include/Config.h
    src/Settings.cc              include/Settings.h
    src/Extractors/SPextractor.cc         include/Extractors/SPextractor.h
    src/Extractors/superpoint_onnx.cc     include/Extractors/superpoint_onnx.h
    src/Matchers/SPmatcher.cc              include/Matchers/SPmatcher.h
    src/Matchers/lightglue_onnx.cpp        include/Matchers/lightglue_onnx.h
    src/Matchers/transform.cpp             include/Matchers/transform.h
    include/SerializationUtils.h
    testyolo.cpp
    src/YoloDetection.cpp         include/YoloDetection.h
)

add_subdirectory(Thirdparty/g2o)

# ───────────────────────────────────────────────────────────
# ONNX Runtime (adjust paths if needed)
# ───────────────────────────────────────────────────────────
set(ONNXRUNTIME_LIB_DIR "/home/anast/Requirements/onnxruntime-linux-x64-gpu-1.16.3/lib"     CACHE PATH "")
set(ONNXRUNTIME_INCLUDE_DIR "/home/anast/Requirements/onnxruntime-linux-x64-gpu-1.16.3/include" CACHE PATH "")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3/lib/libDBoW3.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${TORCH_LIBRARIES}
    -lboost_serialization
    -lcrypto
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

# ───────────────────────────────────────────────────────────
# Examples / test executables
# ───────────────────────────────────────────────────────────
# Monocular
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(testyolo               testyolo.cpp)
target_link_libraries(testyolo        ${PROJECT_NAME})

add_executable(mono_tum               Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum        ${PROJECT_NAME})

add_executable(mono_kitti             Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti      ${PROJECT_NAME})

add_executable(mono_euroc             Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc      ${PROJECT_NAME})

add_executable(mono_tum_vi            Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi     ${PROJECT_NAME})

# Monocular-Inertial
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc    Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi   Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

# Stereo-Inertial
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc  Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})
