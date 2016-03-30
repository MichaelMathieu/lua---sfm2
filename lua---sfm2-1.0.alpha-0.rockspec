package = "lua---sfm2"
version = "1.0.alpha-0"

source = {
   url = "https://github.com/MichaelMathieu/lua---sfm2",
   tag = "master"
}

description = {
   summary = "Structure From Motion",
   homepage = "https://github.com/MichaelMathieu/lua---sfm2",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB) -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}