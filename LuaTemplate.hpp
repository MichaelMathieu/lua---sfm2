#ifndef __LUA_TEMPLATE_HPP__
#define __LUA_TEMPLATE_HPP__

#ifndef __INSIDE_THPP_HPP__
#error LuaTemplate.hpp should not be included outside of THpp.hpp
#endif

extern "C" {
#include<luaT.h>
}
#include<THpp.hpp>
#include<vector>
#include<string>

template<typename T> inline T FromLuaStack(lua_State* L, int i) {
  THerror("Call of FromLuaStack on a non-implemented type");
}
template<> inline bool FromLuaStack<bool>(lua_State* L, int i) {
  return (bool)lua_toboolean(L, i);
}
template<> inline int FromLuaStack<int>(lua_State* L, int i) {
  return (int)lua_tointeger(L, i);
}
template<> inline long int FromLuaStack<long int>(lua_State* L, int i) {
  return (long int)lua_tointeger(L, i);
}
template<> inline float FromLuaStack<float>(lua_State* L, int i) {
  return (float)lua_tonumber(L, i);
}
template<> inline double FromLuaStack<double>(lua_State* L, int i) {
  return (double)lua_tonumber(L, i);
}
template<> inline std::string FromLuaStack<std::string>(lua_State* L, int i) {
  return std::string(lua_tostring(L, i));
}
template<> inline THTensor<float> FromLuaStack<THTensor<float> >(lua_State* L, int i) {
  return THTensor<float>((TH<float>::CTensor*)luaT_checkudata(L, i, luaT_checktypename2id(L, "torch.FloatTensor")));
}
template<> inline THTensor<double> FromLuaStack<THTensor<double> >(lua_State* L, int i) {
  return THTensor<double>((TH<double>::CTensor*)luaT_checkudata(L, i, luaT_checktypename2id(L, "torch.DoubleTensor")));
}

template<typename T> std::vector<T> TableFromLuaStack(lua_State* L, int i) {
  int n = luaL_getn(L, i);
  std::vector<T> ret;
  int newi = (i > 0) ? i : i-1;
  for (int j = 0; j < n; ++j) {
    lua_pushnumber(L, j+1);
    lua_gettable(L, newi);
    ret.push_back(FromLuaStack<T>(L, -1));
  }
  return ret;
}

#define MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(type)			\
  template<>								\
  inline std::vector< type > FromLuaStack<std::vector< type > >(lua_State* L, int i) { \
      return TableFromLuaStack< type >(L, i);				\
    }
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(int)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(long)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(float)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(double)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(std::string)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(THTensor<float>)
MAKE_FROM_LUA_STACK_TABLE_TEMPLATE(THTensor<double>)
#undef MAKE_FROM_LUA_STACK_TABLE_TEMPLATE


template<typename T> inline void PushOnLuaStack(lua_State* L, const T & topush) {
  THerror("Call of PushOnLuaStack on a non-implemented type");
}
template<> inline void PushOnLuaStack<int>(lua_State* L, const int & topush) {
  lua_pushinteger(L, topush);
}
template<> inline void PushOnLuaStack<long int>(lua_State* L, const long int & topush) {
  lua_pushinteger(L, topush);
}
template<> inline void PushOnLuaStack<float>(lua_State* L, const float & topush) {
  lua_pushnumber(L, topush);
}
template<> inline void PushOnLuaStack<double>(lua_State* L, const double & topush) {
  lua_pushnumber(L, topush);
}  

#endif
