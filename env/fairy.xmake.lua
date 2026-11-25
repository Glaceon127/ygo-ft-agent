-- Build only the YGOPro environment against the same mobile core as training.
add_rules("mode.debug", "mode.release")
add_repositories("my-repo repo")
add_requires("pybind11 2.13.*", "fmt 10.2.*", "glog 0.6.0", "sqlite3 3.43.0+200",
             "concurrentqueue 1.0.4", "unordered_dense 4.4.*", "sqlitecpp 3.2.1")

target("ygopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense")
    set_languages("c++17")
    local core = os.getenv("FT_CORE_DIR") or "repo/ocgcore_mobile"
    local lua = os.getenv("FT_LUA_DIR") or "/usr/local"
    add_includedirs("ygoenv", core, path.join(lua, "include/lua"))
    add_linkdirs(path.join(core, "lib"), path.join(lua, "lib"))
    add_links("ocgcore_mobile", "lua")
    after_build(function (target)
        os.cp(target:targetfile(), "$(projectdir)/ygoenv/ygoenv/ygopro")
    end)
