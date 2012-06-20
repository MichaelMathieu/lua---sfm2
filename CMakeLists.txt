CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(src THpp.cpp sfm.cpp genericpp/common.cpp genericpp/LM.cpp genericpp/sfm2frames.cpp genericpp/egoMotion.cpp genericpp/calibration.cpp genericpp/epipoles.cpp)
SET(luasrc init.lua)

ADD_TORCH_PACKAGE(sfm2 "${src}" "${luasrc}" "Structure From Motion")
TARGET_LINK_LIBRARIES(sfm2 luaT TH ${OpenCV_LIBS})
