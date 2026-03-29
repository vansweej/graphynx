fn main() {
    // Provide the CUDA stub libs to the linker so it can resolve -lcuda and
    // -lnvrtc at link time without the real driver being present.
    //
    // CUDA_PATH and NVRTC_PATH are set by the Nix shell hook (flake.nix).
    //
    // The stubs directory will appear in the binary's rpath (Rust records any
    // -L native= path that contains .so files). This is harmless at runtime:
    // the real libcuda.so.1 is found first via .nvidia-libs/ (which appears
    // earlier in the rpath), and the stubs dir only contains libcuda.so (no
    // .1 suffix), so the dynamic linker ignores it.
    //
    // We do NOT add /usr/lib/x86_64-linux-gnu here. That would cause tarpaulin
    // to inject it into LD_LIBRARY_PATH for test binaries, loading the system
    // glibc (2.35) instead of the Nix glibc (2.42) and causing:
    //   symbol lookup error: /usr/lib/x86_64-linux-gnu/libc.so.6:
    //   undefined symbol: __nptl_change_stack_perm, version GLIBC_PRIVATE
    //
    // At runtime, the real libcuda.so.1 is resolved via the rpath pointing to
    // .nvidia-libs/, which is set in RUSTFLAGS from the shell hook.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={cuda_path}/lib/stubs");
    }
    if let Ok(nvrtc_path) = std::env::var("NVRTC_PATH") {
        println!("cargo:rustc-link-search=native={nvrtc_path}/lib/stubs");
    }
}
