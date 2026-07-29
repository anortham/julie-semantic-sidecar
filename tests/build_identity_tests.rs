#[path = "../build_support.rs"]
mod build_support;

#[test]
fn source_path_remaps_do_not_change_the_native_build_identity() {
    let separator = '\x1f';
    let first = [
        "-C",
        "opt-level=3",
        "--remap-path-prefix=/first/target=/cargo-target",
        "--remap-path-prefix",
        "/first/workspace=/workspace",
    ]
    .join(&separator.to_string());
    let second = [
        "-C",
        "opt-level=3",
        "--remap-path-prefix=/second/target=/cargo-target",
        "--remap-path-prefix",
        "/second/workspace=/workspace",
    ]
    .join(&separator.to_string());

    assert_eq!(
        build_support::identity_rustflags(&first),
        build_support::identity_rustflags(&second)
    );
}

#[test]
fn codegen_flags_remain_part_of_the_native_build_identity() {
    assert_ne!(
        build_support::identity_rustflags("-C\x1fopt-level=2"),
        build_support::identity_rustflags("-C\x1fopt-level=3")
    );
}
