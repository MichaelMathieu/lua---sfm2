CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
#SET(CMAKE_MODULE_PATH ".." ${CMAKE_MODULE_PATH})
#FIND_PACKAGE(GSL REQUIRED)

SET(src THpp.cpp sfm.cpp genericpp/common.cpp genericpp/LM.cpp genericpp/sfm2frames.cpp genericpp/egoMotion.cpp genericpp/calibration.cpp genericpp/epipoles.cpp genericpp/random.cpp)
SET(luasrc init.lua)
#SET(USE_GSL)
IF(DEFINED USE_GSL)
SET(GSL_LIBS gsl m gslcblas)
ADD_DEFINITIONS(-DUSE_GSL)
ENDIF(DEFINED USE_GSL)

ADD_TORCH_PACKAGE(sfm2 "${src}" "${luasrc}" "Structure From Motion")
TARGET_LINK_LIBRARIES(sfm2 luaT TH ${OpenCV_LIBS} ${GSL_LIBS})

INSTALL(TARGETS sfm2
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")
