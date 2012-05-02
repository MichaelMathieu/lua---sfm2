#ifndef __THPP_HPP__
#define __THPP_HPP__
#define __INSIDE_THPP_HPP__

extern "C" {
#include <TH.h>
}

#undef THTensor
#undef THStorage

#define THTensorC TH_CONCAT_3(TH,Real,Tensor)
#define THStorageC TH_CONCAT_3(TH,Real,Storage)

#include "THTemplateGenerateFloatTypes.hpp"
#include "LuaTemplate.hpp"

extern lua_State* L_global;
#include<string>

inline void setLuaState(lua_State* L) {
  L_global = L;
}

inline void THerror(const std::string & str) {
  if (L_global == NULL)
    throw str;
  else
    luaL_error(L_global, str.c_str());
}

#undef __INSIDE_THPP_HPP__
#endif
