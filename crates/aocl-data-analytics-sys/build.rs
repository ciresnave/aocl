use aocl_build::{ensure_libclang_path, locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();
    let root = locate_aocl_root();

    Component {
        component_dir: "amd-da",
        wrapper: "wrapper/data_analytics.h",
        module: "bindings",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["aocl-da"],
        win_dynamic: &["aocl-da"],
        unix_static: &["aocl-da"],
        unix_dynamic: &["aocl-da"],
        extra_includes: &[],
    }
    .build(&root);
}
