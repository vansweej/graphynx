{
  # ============================================================================
  # Reproducible Rust + CUDA development environment
  #
  # All build tools (Rust toolchain, CUDA toolkit, cargo utilities) are pinned
  # via flake.lock to exact nixpkgs and rust-overlay commits. This means the
  # environment can be reproduced exactly, even years from now, with:
  #
  #   nix develop
  #
  # The one unavoidable system dependency is the NVIDIA kernel driver interface
  # (libcuda.so.1 and friends). These libraries MUST match the kernel module
  # version installed on the host and cannot be packaged in Nix. Everything
  # else — headers, nvcc, nvrtc, the Rust toolchain — is fully Nix-managed.
  #
  # To update all pinned inputs:
  #
  #   nix flake update
  # ============================================================================

  description = "Reproducible Rust + CUDA development environment";

  inputs = {
    # nixos-unstable is used for up-to-date CUDA packages. The exact commit is
    # pinned in flake.lock — the "unstable" in the name refers to the NixOS
    # release channel, not to the reproducibility of this flake.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # rust-overlay provides nightly/stable/beta Rust toolchains via rustup
    # toolchain files. Follows nixpkgs to avoid duplicate glibc versions.
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ rust-overlay.overlays.default ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true; # required for CUDA packages
        };

        rustVersion = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        cudaPackages = pkgs.cudaPackages;

        # ── System NVIDIA driver libs ─────────────────────────────────────────
        # These three libraries are the userspace interface to the NVIDIA kernel
        # module. They must come from the host OS because:
        #
        #   1. They must exactly match the kernel module version.
        #   2. The kernel module is managed by the OS, not Nix.
        #
        # We symlink only these specific files into .nvidia-libs/ rather than
        # putting /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH or rpath
        # directly. That directory also contains the system glibc, which
        # conflicts with the newer Nix glibc used by the shell, causing
        # segfaults in unrelated tools (e.g. rust-analyzer, which).
        #
        # NOTE: This path is Debian/Ubuntu/Pop!_OS specific. On other distros
        # the NVIDIA driver libs may be in a different location, e.g.:
        #   Fedora/RHEL: /usr/lib64
        #   Arch:        /usr/lib
        systemNvidiaLibDir = "/usr/lib/x86_64-linux-gnu";
        systemNvidiaLibs = [
          "libcuda.so.1"                   # CUDA driver API
          "libnvidia-ptxjitcompiler.so.1"  # PTX JIT compiler
          "libnvidia-nvvm.so.4"            # NVVM IR compiler
        ];

      in
      {
        devShells.default = pkgs.mkShell {
          name = "rusty-cuda";

          buildInputs = with pkgs; [
            # Rust toolchain — version pinned via rust-toolchain.toml
            rustVersion

            # Cargo utilities
            cargo-deny
            cargo-edit
            cargo-tarpaulin
            cargo-watch
            cargo-outdated
            cargo-update

            # LSP — nvim rust-analyzer will use this binary from PATH
            rust-analyzer

            # CUDA toolkit components (all pinned via flake.lock)
            cudaPackages.cuda_cudart  # CUDA runtime headers + stub libs
            cudaPackages.cuda_nvrtc   # NVRTC runtime compilation library
            cudaPackages.cuda_nvcc    # CUDA compiler driver

            # C++ standard library needed by Rust's cc crate and CUDA code
            stdenv.cc.cc.lib

            # gcc-13 is used as the nvcc host compiler. gcc-14+ triggers a
            # stack overflow when processing CUDA headers because Nix injects
            # -fstack-protector-strong and -fstack-clash-protection via
            # NIX_HARDENING_ENABLE, which overwhelms the CUDA header template
            # instantiation depth.
            gcc13
          ];

          shellHook = ''
            export CUDA_PATH="${cudaPackages.cuda_cudart}"
            export NVRTC_PATH="${cudaPackages.cuda_nvrtc}"

            # Expose Nix CUDA and C++ runtime libs for dynamic linking at runtime.
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$CUDA_PATH/lib:$NVRTC_PATH/lib:$LD_LIBRARY_PATH"

            # LIBRARY_PATH is used by gcc-wrapper for link-time -L search paths.
            # Do NOT put stub directories here: gcc-wrapper records every
            # LIBRARY_PATH entry as an rpath in the output binary, which would
            # embed stub-only paths that must never be loaded at runtime.
            # Stubs are provided via cargo:rustc-link-search in build.rs instead.

            # NIX_LDFLAGS is set by mkShell and includes a -rpath $out/lib entry
            # where $out resolves to outputs/out inside the project directory.
            # That directory never exists and would be a dangling rpath entry in
            # every compiled binary. We remove only the two tokens "-rpath <out>"
            # while keeping all -L flags (which crtbeginS.o and friends need).
            export NIX_LDFLAGS="$(echo "$NIX_LDFLAGS" | sed 's|-rpath [^ ]*outputs/out[^ ]*||g')"

            # Create .nvidia-libs/ containing symlinks to the host NVIDIA driver
            # libs. This directory is added as an rpath in compiled binaries so
            # they find the real driver at runtime. See comment above for why
            # we symlink rather than using the system lib dir directly.
            mkdir -p .nvidia-libs
            for lib in ${pkgs.lib.concatStringsSep " " systemNvidiaLibs}; do
              src="${systemNvidiaLibDir}/$lib"
              if [ -f "$src" ]; then
                ln -sf "$src" ".nvidia-libs/$lib"
              fi
            done

            # Embed .nvidia-libs as rpath so compiled binaries resolve the
            # real NVIDIA driver libs at runtime without LD_PRELOAD or
            # LD_LIBRARY_PATH modifications. The Nix glibc rpath is also
            # embedded so that cargo-tarpaulin, which overrides RUSTFLAGS when
            # building test binaries, cannot accidentally cause the system
            # glibc (2.35 on Ubuntu/Pop!_OS) to be loaded instead of the Nix
            # glibc (2.42). The other half of the tarpaulin fix is in build.rs:
            # we must NOT emit cargo:rustc-link-search for /usr/lib/x86_64-linux-gnu
            # because tarpaulin converts those search paths into LD_LIBRARY_PATH
            # entries for test binaries, which would expose the system glibc.
            export RUSTFLAGS="-C link-arg=-Wl,-rpath,$PWD/.nvidia-libs -C link-arg=-Wl,-rpath,${pkgs.glibc}/lib $RUSTFLAGS"

            # Point nvcc at gcc-13 as its host compiler (see buildInputs above).
            export NVCC_HOST_COMPILER="${pkgs.gcc13}/bin/gcc"
            export PATH="${cudaPackages.cuda_nvcc}/bin:$PATH"
          '';
        };
      }
    );
}
