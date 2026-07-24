//! Deterministic package inventory creation and validation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path};
use std::time::UNIX_EPOCH;

use crate::{manifest, DEFAULT_MODEL_ID, VERSION};

pub const MANIFEST_FILE: &str = "package-manifest.json";
pub const SCHEMA_VERSION: u32 = 1;
const NATIVE_PATCH_IDENTITY_PREFIX: &str =
    "native_patch=llama-cpp-sys-2-0.1.151:vulkan-infinity-v3:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageTier {
    Portable,
    Vendor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvertisedBackend {
    Metal,
    Vulkan,
    Cuda,
}

impl AdvertisedBackend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageFileRole {
    Executable,
    CoreRuntime,
    CpuBackend,
    AcceleratorBackend,
    License,
    Readme,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageProfile {
    pub rust_target: String,
    pub tier: PackageTier,
    pub advertised_backend: AdvertisedBackend,
    pub native_build_identity: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageManifest {
    pub schema_version: u32,
    pub sidecar_version: String,
    pub rust_target: String,
    pub tier: PackageTier,
    pub advertised_backend: AdvertisedBackend,
    pub native_build_identity: String,
    pub model_policy: ModelPolicy,
    pub files: Vec<PackageFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPolicy {
    pub ids: Vec<String>,
    pub default_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageFile {
    pub path: String,
    pub sha256: String,
    pub size: u64,
    pub role: PackageFileRole,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackageError(String);

impl PackageError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for PackageError {}

pub fn write(root: &Path, profile: &PackageProfile) -> Result<PackageManifest, PackageError> {
    let package = create(root, profile)?;
    let mut encoded = serde_json::to_vec_pretty(&package)
        .map_err(|error| PackageError::new(format!("cannot encode package manifest: {error}")))?;
    encoded.push(b'\n');
    let mut temporary = tempfile::NamedTempFile::new_in(root)
        .map_err(|error| PackageError::new(format!("cannot create manifest temporary: {error}")))?;
    temporary
        .write_all(&encoded)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| PackageError::new(format!("cannot write package manifest: {error}")))?;
    temporary
        .persist(root.join(MANIFEST_FILE))
        .map_err(|error| {
            PackageError::new(format!(
                "cannot install package manifest atomically: {error}"
            ))
        })?;
    verify(root)
}

fn create(root: &Path, profile: &PackageProfile) -> Result<PackageManifest, PackageError> {
    validate_profile(profile)?;
    let mut files = payload_files(root)?
        .into_iter()
        .map(|path| describe_file(root, &path, profile))
        .collect::<Result<Vec<_>, _>>()?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    let package = PackageManifest {
        schema_version: SCHEMA_VERSION,
        sidecar_version: VERSION.to_string(),
        rust_target: profile.rust_target.clone(),
        tier: profile.tier,
        advertised_backend: profile.advertised_backend,
        native_build_identity: profile.native_build_identity.clone(),
        model_policy: model_policy(),
        files,
    };
    validate_shape(&package)?;
    Ok(package)
}

pub fn verify(root: &Path) -> Result<PackageManifest, PackageError> {
    let manifest_path = root.join(MANIFEST_FILE);
    let raw = std::fs::read(&manifest_path).map_err(|error| {
        PackageError::new(format!("cannot read {}: {error}", manifest_path.display()))
    })?;
    let package: PackageManifest = serde_json::from_slice(&raw)
        .map_err(|error| PackageError::new(format!("invalid package manifest JSON: {error}")))?;
    for file in &package.files {
        validate_relative_path(&file.path)?;
    }
    validate_manifest_metadata(&package)?;
    validate_shape(&package)?;

    let actual = payload_files(root)?
        .into_iter()
        .map(|path| relative_name(root, &path))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let declared = package
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<BTreeSet<_>>();
    if let Some(path) = actual.difference(&declared).next() {
        return Err(PackageError::new(format!("undeclared file {path}")));
    }
    if let Some(path) = declared.difference(&actual).next() {
        return Err(PackageError::new(format!(
            "declared file is missing: {path}"
        )));
    }

    let profile = PackageProfile {
        rust_target: package.rust_target.clone(),
        tier: package.tier,
        advertised_backend: package.advertised_backend,
        native_build_identity: package.native_build_identity.clone(),
    };
    for declared in &package.files {
        let path = root.join(&declared.path);
        let observed = describe_file(root, &path, &profile)?;
        if observed.role != declared.role {
            return Err(PackageError::new(format!(
                "role mismatch for {}",
                declared.path
            )));
        }
        if observed.size != declared.size {
            return Err(PackageError::new(format!(
                "size mismatch for {}",
                declared.path
            )));
        }
        if observed.sha256 != declared.sha256 {
            return Err(PackageError::new(format!(
                "checksum mismatch for {}",
                declared.path
            )));
        }
    }
    Ok(package)
}

pub fn verify_patched(root: &Path) -> Result<PackageManifest, PackageError> {
    let package = verify(root)?;
    let Some(patch_identity) = package
        .native_build_identity
        .split(';')
        .find(|value| value.starts_with("native_patch="))
    else {
        return Err(PackageError::new(
            "package native patch identity is missing",
        ));
    };
    let Some(digest) = patch_identity.strip_prefix(NATIVE_PATCH_IDENTITY_PREFIX) else {
        return Err(PackageError::new(
            "package native patch identity is invalid",
        ));
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PackageError::new(
            "package native patch identity is invalid",
        ));
    }
    Ok(package)
}

pub fn runtime_identity(root: &Path) -> Result<String, PackageError> {
    let manifest_path = root.join(MANIFEST_FILE);
    let raw = std::fs::read(&manifest_path).map_err(|error| {
        PackageError::new(format!("cannot read {}: {error}", manifest_path.display()))
    })?;
    let package: PackageManifest = serde_json::from_slice(&raw)
        .map_err(|error| PackageError::new(format!("invalid package manifest JSON: {error}")))?;
    for file in &package.files {
        validate_relative_path(&file.path)?;
    }
    validate_manifest_metadata(&package)?;
    validate_shape(&package)?;

    let mut hasher = Sha256::new();
    hasher.update(&raw);
    for file in package.files.iter().filter(|file| {
        matches!(
            file.role,
            PackageFileRole::CoreRuntime
                | PackageFileRole::CpuBackend
                | PackageFileRole::AcceleratorBackend
        )
    }) {
        let path = root.join(&file.path);
        let metadata = std::fs::metadata(&path).map_err(|error| {
            PackageError::new(format!("cannot inspect {}: {error}", path.display()))
        })?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
            .map(|value| value.as_nanos())
            .unwrap_or_default();
        hasher.update(file.path.len().to_le_bytes());
        hasher.update(file.path.as_bytes());
        hasher.update(metadata.len().to_le_bytes());
        hasher.update(modified.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_manifest_metadata(package: &PackageManifest) -> Result<(), PackageError> {
    if package.schema_version != SCHEMA_VERSION {
        return Err(PackageError::new("unsupported package manifest schema"));
    }
    if package.sidecar_version != VERSION {
        return Err(PackageError::new("package sidecar version mismatch"));
    }
    validate_profile(&PackageProfile {
        rust_target: package.rust_target.clone(),
        tier: package.tier,
        advertised_backend: package.advertised_backend,
        native_build_identity: package.native_build_identity.clone(),
    })?;
    if package.model_policy != model_policy() {
        return Err(PackageError::new("model policy mismatch"));
    }
    if !package
        .files
        .windows(2)
        .all(|pair| pair[0].path < pair[1].path)
    {
        return Err(PackageError::new(
            "package file list must be sorted without duplicates",
        ));
    }
    Ok(())
}

fn validate_profile(profile: &PackageProfile) -> Result<(), PackageError> {
    if profile.rust_target.is_empty()
        || !profile
            .rust_target
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(PackageError::new("invalid Rust target"));
    }
    if profile.native_build_identity.is_empty()
        || profile.native_build_identity.contains('/')
        || profile.native_build_identity.contains('\\')
        || profile.native_build_identity.contains(".worktrees")
    {
        return Err(PackageError::new(
            "native build identity contains a development path",
        ));
    }
    match (profile.advertised_backend, profile.tier) {
        (AdvertisedBackend::Metal, PackageTier::Portable)
            if matches!(
                profile.rust_target.as_str(),
                "aarch64-apple-darwin" | "x86_64-apple-darwin"
            ) => {}
        (AdvertisedBackend::Vulkan, PackageTier::Portable)
            if matches!(
                profile.rust_target.as_str(),
                "x86_64-unknown-linux-gnu" | "x86_64-pc-windows-msvc"
            ) => {}
        (AdvertisedBackend::Cuda, PackageTier::Vendor)
            if matches!(
                profile.rust_target.as_str(),
                "x86_64-unknown-linux-gnu" | "x86_64-pc-windows-msvc"
            ) => {}
        _ => return Err(PackageError::new("unsupported package profile")),
    }
    Ok(())
}

fn model_policy() -> ModelPolicy {
    let mut ids = manifest::manifest()
        .iter()
        .map(|pin| pin.id.to_string())
        .collect::<Vec<_>>();
    ids.sort();
    ModelPolicy {
        ids,
        default_id: DEFAULT_MODEL_ID.to_string(),
    }
}

fn payload_files(root: &Path) -> Result<Vec<std::path::PathBuf>, PackageError> {
    let entries = std::fs::read_dir(root)
        .map_err(|error| PackageError::new(format!("cannot read package root: {error}")))?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry
            .map_err(|error| PackageError::new(format!("cannot read package entry: {error}")))?;
        if entry.file_name() == MANIFEST_FILE {
            continue;
        }
        let kind = entry
            .file_type()
            .map_err(|error| PackageError::new(format!("cannot inspect package entry: {error}")))?;
        if !kind.is_file() {
            return Err(PackageError::new(format!(
                "{} must be a regular file in the flat package root",
                entry.path().display()
            )));
        }
        paths.push(entry.path());
    }
    Ok(paths)
}

fn describe_file(
    root: &Path,
    path: &Path,
    profile: &PackageProfile,
) -> Result<PackageFile, PackageError> {
    let relative = relative_name(root, path)?;
    validate_relative_path(&relative)?;
    if is_model_weight(&relative) {
        return Err(PackageError::new(format!(
            "model weight is forbidden in packages: {relative}"
        )));
    }
    let role = classify(&relative, profile)?;
    let metadata = std::fs::metadata(path)
        .map_err(|error| PackageError::new(format!("cannot inspect {relative}: {error}")))?;
    Ok(PackageFile {
        path: relative,
        sha256: file_sha256(path)?,
        size: metadata.len(),
        role,
    })
}

fn classify(name: &str, profile: &PackageProfile) -> Result<PackageFileRole, PackageError> {
    let executable = if profile.rust_target.contains("windows") {
        "julie-semantic-sidecar.exe"
    } else {
        "julie-semantic-sidecar"
    };
    if name == executable {
        return Ok(PackageFileRole::Executable);
    }
    if name == "LICENSE" {
        return Ok(PackageFileRole::License);
    }
    if name == "README.md" {
        return Ok(PackageFileRole::Readme);
    }
    if !is_native_library(name) {
        return Err(PackageError::new(format!(
            "unsupported package file {name}"
        )));
    }
    let lowercase = name.to_ascii_lowercase();
    if lowercase.contains("ggml-cpu") {
        return Ok(PackageFileRole::CpuBackend);
    }
    for backend in ["metal", "vulkan", "cuda", "hip", "sycl"] {
        if lowercase.contains(&format!("ggml-{backend}")) {
            if backend == profile.advertised_backend.as_str()
                && profile.advertised_backend != AdvertisedBackend::Metal
            {
                return Ok(PackageFileRole::AcceleratorBackend);
            }
            return Err(PackageError::new(format!(
                "accelerator module {name} disagrees with advertised backend {}",
                profile.advertised_backend.as_str()
            )));
        }
    }
    if lowercase.starts_with("libggml")
        || lowercase.starts_with("libllama")
        || lowercase.starts_with("ggml")
        || lowercase.starts_with("llama")
    {
        return Ok(PackageFileRole::CoreRuntime);
    }
    Err(PackageError::new(format!(
        "unsupported native library {name}"
    )))
}

fn validate_shape(package: &PackageManifest) -> Result<(), PackageError> {
    let count = |role| {
        package
            .files
            .iter()
            .filter(|file| file.role == role)
            .count()
    };
    for (role, name) in [
        (PackageFileRole::Executable, "executable"),
        (PackageFileRole::License, "license"),
        (PackageFileRole::Readme, "readme"),
    ] {
        if count(role) != 1 {
            return Err(PackageError::new(format!("missing required {name} role")));
        }
    }
    match package.advertised_backend {
        AdvertisedBackend::Metal => {
            if count(PackageFileRole::CoreRuntime)
                + count(PackageFileRole::CpuBackend)
                + count(PackageFileRole::AcceleratorBackend)
                != 0
            {
                return Err(PackageError::new(
                    "built-in Metal package must not contain native plugin files",
                ));
            }
        }
        AdvertisedBackend::Vulkan | AdvertisedBackend::Cuda => {
            for (role, name) in [
                (PackageFileRole::CoreRuntime, "core runtime"),
                (PackageFileRole::CpuBackend, "CPU backend"),
                (
                    PackageFileRole::AcceleratorBackend,
                    "advertised accelerator",
                ),
            ] {
                if count(role) == 0 {
                    return Err(PackageError::new(format!("missing required {name} role")));
                }
            }
            if count(PackageFileRole::AcceleratorBackend) != 1 {
                return Err(PackageError::new(
                    "package must contain exactly one advertised accelerator module",
                ));
            }
        }
    }
    Ok(())
}

fn relative_name(root: &Path, path: &Path) -> Result<String, PackageError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| PackageError::new("package path escapes root"))?;
    relative
        .to_str()
        .map(str::to_string)
        .ok_or_else(|| PackageError::new("package path is not valid UTF-8"))
}

fn validate_relative_path(path: &str) -> Result<(), PackageError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.contains('\\')
        || parsed.is_absolute()
        || parsed.components().count() != 1
        || !matches!(parsed.components().next(), Some(Component::Normal(_)))
    {
        return Err(PackageError::new(format!("invalid package path {path:?}")));
    }
    Ok(())
}

fn is_native_library(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    lowercase.ends_with(".dll") || lowercase.contains(".so") || lowercase.contains(".dylib")
}

fn is_model_weight(name: &str) -> bool {
    let lowercase = name.to_ascii_lowercase();
    [".gguf", ".onnx", ".safetensors", ".pt", ".pth"]
        .iter()
        .any(|extension| lowercase.ends_with(extension))
}

fn file_sha256(path: &Path) -> Result<String, PackageError> {
    let mut file = File::open(path)
        .map_err(|error| PackageError::new(format!("cannot read {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            PackageError::new(format!("cannot hash {}: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
