#ifndef __THPP_HPP__
#define __THPP_HPP__
#define __INSIDE_THPP_HPP__

extern "C" {
#include <TH.h>
#include<luaT.h>
}

#undef THTensor
#undef THStorage

#define THTensorC TH_CONCAT_3(TH,Real,Tensor)
#define THStorageC TH_CONCAT_3(TH,Real,Storage)

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

inline std::string THassertFormat(const char* msg, const char* file, int line) {
  char buffer[256];
  sprintf(buffer, "%s File %s, line %d\n", msg, file, line);
  return buffer;
}
#define THassert(a) (							\
  (!(a)) ?								\
  (THerror(THassertFormat("THassert: ", __FILE__, __LINE__))) :		\
  static_cast<void>(0))

#include "THTemplateGenerateFloatTypes.hpp"

template<typename T> void THcheckSizeFunction(const THTensor<T> & t,
					      const char* file, int line,
					      int s1, int s2=-1, int s3=-1, int s4=-1) {
  int s[4]; s[0] = s1; s[1] = s2; s[2] = s3; s[3] = s4;
  int nDims = 1;
  for (int i = 3; i >= 1; --i)
    if (s[i] != -1) {
      nDims = i+1;
      break;
    }
  if (t.nDimension() != nDims)
    THassertFormat("THcheckSize: ", file, line);
  for (int i = 0; i < nDims; ++i)
    if (t.size(i) != s[i])
      THassertFormat("THcheckSize: ", file, line);
}
#define THcheckSize(t,...) (THcheckSizeFunction((t),__FILE__,__LINE__,__VA_ARGS__))

#include "LuaTemplate.hpp"

#undef __INSIDE_THPP_HPP__
#endif
