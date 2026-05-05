# graphynx

graphynx is a **backend-agnostic dataflow graph execution engine** in Rust.
Users define computation as a directed graph of typed nodes. The engine schedules
and executes nodes in dependency order, dispatching work to whichever backend a
node targets — signal processing on CPU, raw CUDA kernels on GPU, or primitive
ML operations.

## Crate Workspace

```mermaid
graph LR
    GC["graph-core\nTypes · Ops · Graph IR"]
    B["backends\nBackend trait · BackendError"]
    BC["backends-cpu\nCpuBackend\nSignal ops (Window · FFT · BandExtract)"]
    BX["backends-cuda\nCudaBackend\nRaw PTX kernels"]
    RT["runtime\nExecutor · demo binary"]

    GC --> B
    B --> BC
    B --> BX
    GC --> RT
    B --> RT
    BC --> RT
    BX --> RT
```

## Backends

| Backend | Kind | Status |
|---|---|---|
| CPU (`backends-cpu`) | Signal ops (Window, FFT, BandExtract) | ✅ implemented |
| CUDA (`backends-cuda`) | Raw compute kernels (PTX) | 🔧 in progress |
| OpenCL | Compute | planned |
| Vulkan / wgpu | Compute | planned |
| ONNX Runtime | ML inference | planned |
| libtorch | ML ops + inference | planned |
| candle / burn | ML ops + inference | planned |

## Execution Pipeline

```mermaid
flowchart LR
    A["GraphBuilder\n(define nodes + edges)"]
    B["Graph\n(validated DAG)"]
    C["Executor::new()\n(schedule + allocate)"]
    D["executor.run()\n(dispatch in topo order)"]
    E["OutputHandle\n(read results)"]

    A -->|".build()"| B
    B --> C
    C -->|"write inputs"| D
    D --> E
```

## Getting Started

### Prerequisites

- [Nix](https://nixos.org/) with flakes enabled
- NVIDIA GPU + driver (for CUDA backend only)

### Build

```bash
# Enter the reproducible dev shell (sets CUDA_PATH, RUSTFLAGS, etc.)
nix develop

# Build all crates
nix develop --command cargo build
```

### Run the demo

```bash
nix develop --command cargo run --bin demo
```

### Test

```bash
nix develop --command cargo test
nix develop --command cargo tarpaulin   # coverage report
```

### Lint

```bash
nix develop --command cargo clippy
nix develop --command cargo fmt --check
nix develop --command cargo deny check
```

## Quick Example — Signal Processing Graph

```rust
use graph_core::{
    graph::GraphBuilder,
    ops::{Op, WindowParams, WindowKind, FftParams, FftDirection, FftOutput,
          BandExtractParams, BandDef},
    types::{DType, TensorType},
};
use backends::DeviceId;
use backends_cpu::CpuBackend;
use runtime::executor::Executor;

let n: usize = 1024;
let sr: f32 = 44_100.0;
let cpu = DeviceId::new("cpu:0");

let mut b = GraphBuilder::new();

// Input: raw audio frame (f32 × N)
let audio_src = b.add_source("audio", TensorType::vector(DType::F32, n).unwrap());

// Window node
let win_node = b.add_op_node(
    Op::Window(WindowParams { size: n, kind: WindowKind::Hann }),
    cpu.clone(),
    vec![TensorType::vector(DType::F32, n).unwrap()],
);

// FFT node
let fft_node = b.add_op_node(
    Op::Fft(FftParams { size: n, direction: FftDirection::Forward,
                        output: FftOutput::MagnitudeOneSided }),
    cpu.clone(),
    vec![TensorType::vector(DType::F32, n / 2 + 1).unwrap()],
);

// BandExtract node — 3 bands with EMA smoothing
let bands = vec![
    BandDef { low_hz: 20.0,  high_hz: 250.0  },
    BandDef { low_hz: 250.0, high_hz: 4_000.0 },
    BandDef { low_hz: 4_000.0, high_hz: 20_000.0 },
];
let band_node = b.add_op_node(
    Op::BandExtract(BandExtractParams::new(bands, sr, n / 2 + 1, 0.1).unwrap()),
    cpu.clone(),
    vec![TensorType::vector(DType::F32, 3).unwrap()],
);

// Wire the graph
b.add_edge(audio_src, win_node, 0);
b.add_edge(win_node,  fft_node, 0);
b.add_edge(fft_node,  band_node, 0);
b.add_sink("energies", band_node, 0);

let graph = b.build().unwrap();
let backend: Box<dyn backends::Backend> = Box::new(CpuBackend::new(cpu));
let mut exec = Executor::new(graph, vec![backend]).unwrap();

// Run one frame
let samples: Vec<f32> = vec![0.0_f32; n];
exec.input("audio").unwrap().write("audio", bytemuck::cast_slice(&samples)).unwrap();
exec.run().unwrap();
let energies: &[f32] = exec.output("energies").unwrap().read().unwrap();
println!("Band energies: {:?}", energies);
```

See [docs/signal-ops.md](docs/signal-ops.md) for the full signal processing guide.

## Project Structure

```
graphynx/                         # Cargo workspace root
├── core/                         # graph-core — pure backend-agnostic types
│   └── src/
│       ├── types/                # DType · Dim · Layout · TensorType · Shape
│       ├── ops/                  # Op enum · OpError · ML + Signal param structs
│       └── graph/                # Graph IR · GraphBuilder · Node · Edge · validator
├── backends/                     # backends — Backend trait · BackendError · DeviceId
├── backends-cpu/                 # backends-cpu — CpuBackend (signal ops)
│   └── src/signal/               # window.rs · fft.rs · band.rs
├── backends-cuda/                # backends-cuda — CudaBackend (raw PTX kernels)
│   ├── kernel.cu                 # CUDA C kernel source
│   ├── compile-kernel.sh         # compiles kernel.cu → kernel.ptx
│   └── build.rs                  # emits CUDA linker search paths
└── runtime/                      # runtime — Executor · demo binary
    └── src/executor/             # scheduler · buffer arena · handles · state
```

## Documentation

| Document | Description |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full long-term architecture plan |
| [docs/architecture.md](docs/architecture.md) | Current code structure overview |
| [docs/getting-started.md](docs/getting-started.md) | Build, test, and first steps |
| [docs/executor.md](docs/executor.md) | Executor developer guide |
| [docs/signal-ops.md](docs/signal-ops.md) | Signal processing ops guide (Window · FFT · BandExtract) |
| [docs/graph-ir.md](docs/graph-ir.md) | Graph IR and builder API |
| [docs/op-catalog.md](docs/op-catalog.md) | Op enum and parameter structs |
| [docs/backend-trait.md](docs/backend-trait.md) | Backend trait system |
| [docs/tensor-type.md](docs/tensor-type.md) | TensorType, Dim, Layout |
| [docs/shape.md](docs/shape.md) | Shape, broadcasting, strides |
| [docs/dtype.md](docs/dtype.md) | DType scalar element types |
| [docs/cuda-backend.md](docs/cuda-backend.md) | CUDA backend details |

## License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.
