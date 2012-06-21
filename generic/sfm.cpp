#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/sfm.cpp"
#else

//======================================================================
// File: sfm
//
// Description: Structure From Motion
//
// Created: April 28th, 2012
//
// Author: Michael Mathieu // michael.mathieu@ens.fr
//======================================================================
extern "C" {
#include <luaT.h>
#include <TH.h>
}

//============================================================
// Register functions in LUA
//
static const luaL_reg libsfm2_(Main__) [] = 
{
  {"inverseMatrix", InverseMatrix<real>},
  {"get2DEgoMotion", Get2DEgoMotion<real>},
  {"getEgoMotion", GetEgoMotion<real>},
  {"removeEgoMotion", RemoveEgoMotion<real>},
  {"undistortImage", UndistortImage<real>},
  {"chessboardCalibrate", ChessboardCalibrate<real>},
  {"getEpipoles", GetEpipoles<real>},
  {"getEpipoleFromMatches", GetEpipoleFromMatches<real>},
  {NULL, NULL}  /* sentinel */
};

LUA_EXTERNC DLL_EXPORT int libsfm2_(Main_init) (lua_State *L) {
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, libsfm2_(Main__), "libsfm2");
  return 1;
}

#endif
