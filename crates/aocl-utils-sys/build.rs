use aocl_build::{ensure_libclang_path, locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-changed=wrapper");
    println!("cargo:rerun-if-changed=build.rs");

    ensure_libclang_path();
    let root = locate_aocl_root();

    Component {
        component_dir: "amd-utils",
        wrapper: "wrapper/utils.h",
        module: "bindings",
        has_int_subdir: false,
        only_lp64: false,
        win_static: &["libaoclutils_static", "au_cpuid_static"],
        win_dynamic: &["libaoclutils", "au_cpuid"],
        unix_static: &["aoclutils", "au_cpuid"],
        unix_dynamic: &["aoclutils", "au_cpuid"],
        extra_includes: &[],
    }
    .build(&root);
}
