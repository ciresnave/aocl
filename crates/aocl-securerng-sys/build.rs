use aocl_build::{ensure_libclang_path, locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();
    let root = locate_aocl_root();

    Component {
        component_dir: "amd-securerng",
        wrapper: "wrapper/securerng.h",
        module: "bindings",
        has_int_subdir: true,
        only_lp64: true,
        win_static: &["amdsecrng-static"],
        win_dynamic: &["amdsecrng"],
        unix_static: &["secrng"],
        unix_dynamic: &["secrng"],
        extra_includes: &[],
    }
    .build(&root);
}
