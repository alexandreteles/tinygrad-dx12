"""Install the exact preview SDK and DXC used by this backend."""

import argparse, hashlib, json, os, platform, shutil, struct, sys, tempfile, urllib.request, zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Package:
    """A NuGet dependency pinned by package version and SHA-256."""

    name: str
    version: str
    sha256: str


PACKAGES = (
    Package("microsoft.direct3d.d3d12", "1.721.3-preview", "0131bce1e4bace3fc08c03018c29a09ede2570b263721c6449b0ec75762ab22d"),
    Package("microsoft.direct3d.dxc", "1.10.2605.37-preview", "8cf5ec1a8abb832de43251bae1ffcec89b22240677c599c38d30a18dee2f4de6"),
)
SDK_VERSION = 721


def architecture() -> str:
    """Return the native 64-bit Python architecture; reject incompatible hosts."""
    if struct.calcsize("P") != 8:
        raise RuntimeError("DX12 requires 64-bit Python")
    if os.name == "nt":
        with Path(sys.executable).open("rb") as stream:
            stream.seek(60)
            stream.seek(struct.unpack("<I", stream.read(4))[0] + 4)
            machine_id = struct.unpack("<H", stream.read(2))[0]
        if machine_id in (0x8664, 0xAA64):
            return {0x8664: "x64", 0xAA64: "arm64"}[machine_id]
        raise RuntimeError(f"Unsupported Python PE architecture: 0x{machine_id:04x}")
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    raise RuntimeError(f"Unsupported Python architecture: {machine}")


def cache_root() -> Path:
    """Return the package cache, optionally overridden by TINYGRAD_DX12_CACHE."""
    return Path(os.environ.get("TINYGRAD_DX12_CACHE", str(Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "tinygrad_dx12")))


def toolchain_path() -> Path:
    """Return the immutable directory for the pinned packages and process architecture."""
    return cache_root() / f"agility-{PACKAGES[0].version}_dxc-{PACKAGES[1].version}_{architecture()}"


def check_pe(path: Path, arch: str) -> None:
    """Reject DLLs whose PE machine differs from the native Python architecture."""
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"Not a PE image: {path}")
        stream.seek(60)
        stream.seek(struct.unpack("<I", stream.read(4))[0])
        signature, machine = struct.unpack("<4sH", stream.read(6))
    if signature != b"PE\0\0" or machine != {"x64": 0x8664, "arm64": 0xAA64}[arch]:
        raise ValueError(f"Wrong PE architecture for {arch}: {path} (machine 0x{machine:04x})")


def extract_package(archive_path: Path, destination: Path, arch: str) -> None:
    """Extract headers, selected native binaries and licenses after validating every member path."""
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive symlink: {member.filename}")
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            binary = str(name).startswith(f"build/native/bin/{arch}/") and name.suffix.lower() in (".dll", ".exe")
            header = str(name).startswith("build/native/include/")
            license_file = "licen" in name.name.lower()
            if member.is_dir() or not (binary or header or license_file):
                continue
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if target.suffix.lower() == ".dll":
                check_pe(target, arch)


def install(archives: Path | None = None) -> Path:
    """Verify pinned archives and atomically install a new immutable toolchain directory.

    Existing installations are preserved so loaded DLLs are never overwritten. ``archives``
    selects an offline directory containing both original NuGet filenames.
    """
    destination, arch = toolchain_path(), architecture()
    if (destination / "manifest.json").is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        payload = staging / "toolchain"
        payload.mkdir()
        for package, short in zip(PACKAGES, ("agility", "dxc")):
            filename = f"{package.name}.{package.version}.nupkg"
            source = archives / filename if archives is not None else staging / filename
            if archives is None:
                urllib.request.urlretrieve(f"https://api.nuget.org/v3-flatcontainer/{package.name}/{package.version}/{filename}", source)
            with source.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != package.sha256:
                raise ValueError(f"SHA-256 mismatch for {filename}: {digest}")
            extract_package(source, payload / short, arch)
        required = (
            payload / "agility/build/native/bin" / arch / "D3D12Core.dll",
            payload / "dxc/build/native/bin" / arch / "dxcompiler.dll",
            payload / "dxc/build/native/bin" / arch / "dxil.dll",
            payload / "dxc/build/native/include/hlsl/dx/linalg.h",
        )
        if missing := [str(path) for path in required if not path.is_file()]:
            raise RuntimeError(f"Missing package assets: {missing}")
        (payload / "manifest.json").write_text(json.dumps({"architecture": arch, "packages": [p.__dict__ for p in PACKAGES]}, indent=2))
        try:
            payload.rename(destination)
        except FileExistsError:
            if not (destination / "manifest.json").is_file():
                raise
    return destination


def main() -> None:
    """Install the toolchain from NuGet or local archives using the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install",))
    parser.add_argument("--archives", type=Path, help="directory containing both pinned .nupkg files (offline installation)")
    args = parser.parse_args()
    print(install(args.archives))


if __name__ == "__main__":
    main()
