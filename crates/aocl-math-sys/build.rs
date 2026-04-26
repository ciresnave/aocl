use aocl_build::{ensure_libclang_path, locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();
    let root = locate_aocl_root();

    Component {
        component_dir: "amd-libm",
        wrapper: "wrapper/math.h",
        module: "bindings",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["libalm-static"],
        win_dynamic: &["libalm"],
        unix_static: &["alm"],
        unix_dynamic: &["alm"],
        extra_includes: &[],
    }
    .build(&root);
}
