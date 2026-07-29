pub fn identity_rustflags(encoded: &str) -> String {
    let mut flags = encoded.split('\x1f');
    let mut identity_flags = Vec::new();

    while let Some(flag) = flags.next() {
        if flag.starts_with("--remap-path-prefix=") {
            continue;
        }
        if flag == "--remap-path-prefix" {
            flags.next();
            continue;
        }
        identity_flags.push(flag);
    }

    identity_flags
        .join("\x1f")
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
