use aocl_build::{locate_aocl_root, Component};

fn main() {
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    println!("cargo:rerun-if-changed=build.rs");

    let root = locate_aocl_root();

    // ScaLAPACK is a Fortran/MPI library; AOCL ships no public C headers,
    // so this crate is link-only at the moment. Hand-written Fortran
    // declarations and a safe wrapper will be added under aocl-scalapack
    // (and the corresponding extern blocks here) in a follow-up.
    Component {
        component_dir: "amd-scalapack",
        wrapper: "",
        module: "bindings",
        has_int_subdir: true,
        only_lp64: false,
        win_static: &["scalapack"],
        win_dynamic: &["scalapack"],
        unix_static: &["scalapack"],
        unix_dynamic: &["scalapack"],
        extra_includes: &[],
    }
    .build(&root);
}
