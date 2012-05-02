extern "C" {
#include <TH.h>
#include <luaT.h>
}

#include "genericpp/sfm.cpp"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_string_(NAME) TH_CONCAT_STRING_3(torch., Real, NAME)
#define libsfm2_(NAME) TH_CONCAT_3(libsfm2_, Real, NAME)

static const void* torch_FloatTensor_id = NULL;
static const void* torch_DoubleTensor_id = NULL;

#include "generic/sfm.cpp"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libsfm2(lua_State *L)
{
  torch_FloatTensor_id = luaT_checktypename2id(L, "torch.FloatTensor");
  torch_DoubleTensor_id = luaT_checktypename2id(L, "torch.DoubleTensor");

  libsfm2_FloatMain_init(L);
  libsfm2_DoubleMain_init(L);

  //luaL_register(L, "libsfm2.double", libsfm2_DoubleMain__);
  //luaL_register(L, "libsfm2.float", libsfm2_FloatMain__);

  return 1;
}
