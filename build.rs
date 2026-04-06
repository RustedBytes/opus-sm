use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=native/opus_sm_wrapper.c");

    let opus_root =
        find_audiopus_opus_root().expect("failed to locate audiopus_sys bundled opus source");
    let include_dir = opus_root.join("include");
    let src_dir = opus_root.join("src");
    let celt_dir = opus_root.join("celt");

    cc::Build::new()
        .file("native/opus_sm_wrapper.c")
        .include(&include_dir)
        .include(&src_dir)
        .include(&celt_dir)
        .flag_if_supported("-Wno-unused-parameter")
        .compile("opus_sm_wrapper");
}

fn find_audiopus_opus_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(cargo_home) = env::var("CARGO_HOME") {
        candidates.push(PathBuf::from(cargo_home));
    }
    if let Ok(home) = env::var("HOME") {
        candidates.push(PathBuf::from(home).join(".cargo"));
    }

    for cargo_home in candidates {
        let registry_src = cargo_home.join("registry").join("src");
        if let Some(found) = search_registry_src(&registry_src) {
            return Some(found);
        }
    }

    None
}

fn search_registry_src(registry_src: &Path) -> Option<PathBuf> {
    let registries = fs::read_dir(registry_src).ok()?;
    for registry in registries.flatten() {
        let registry_path = registry.path();
        if !registry_path.is_dir() {
            continue;
        }

        let packages = fs::read_dir(&registry_path).ok()?;
        for package in packages.flatten() {
            let package_path = package.path();
            if !package_path.is_dir() {
                continue;
            }

            let Some(name) = package_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !name.starts_with("audiopus_sys-") {
                continue;
            }

            let opus_root = package_path.join("opus");
            if opus_root.join("src").is_dir() && opus_root.join("include").is_dir() {
                return Some(opus_root);
            }
        }
    }

    None
}
