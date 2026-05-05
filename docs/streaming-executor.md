# Streaming Executor — Threading Upgrade Path

This document describes the current synchronous execution model and the planned
path to a background-threaded `GraphRunner` that decouples audio capture from
the render loop.

## Current model (Phase 5)

```
render thread
│
├─ audio_source.next_frame()   ← non-blocking; returns None if ring is empty
│
├─ executor.input("audio").write(frame)
├─ executor.run()              ← synchronous; blocks render thread
└─ executor.output("energies").read()
```

The `Executor` runs the full graph on the render thread every frame.  For a
three-node signal graph (Window → FFT → BandExtract at 1024 samples) this
takes roughly **0.1–0.3 ms** on a modern CPU — well within a 16 ms frame
budget.  The synchronous model is therefore sufficient for Phase 5.

## When to upgrade

Consider the threaded `GraphRunner` if any of the following become true:

- The graph grows to 10+ nodes or includes heavy ML inference ops.
- The FFT size increases to 8192+ samples.
- The render thread budget drops below 4 ms (e.g. VR at 90 Hz).
- Audio latency becomes perceptible (graph execution > 5 ms).

## Planned threaded model (Phase 6)

```
audio thread (cpal)
│  push samples → RingBuffer
│
graph thread (GraphRunner)
│  loop:
│    wait until ring.available() >= frame_size
│    executor.input("audio").write(frame)
│    executor.run()
│    triple_buf.publish(energies)   ← atomic write, never blocks
│
render thread
│  energies = triple_buf.read()    ← atomic read, never blocks
│  animate(energies)
```

### Key types

```rust
/// Runs a graph on a dedicated background thread.
pub struct GraphRunner {
    handle: JoinHandle<()>,
    output: Arc<TripleBuffer<[f32; 3]>>,
}

impl GraphRunner {
    /// Spawn the graph thread.  The thread parks until audio is available.
    pub fn new(
        graph: Graph,
        backends: Vec<Box<dyn Backend>>,
        source: Box<dyn AudioSource>,
    ) -> Result<Self, ExecutorError>;

    /// Read the latest published output non-blocking.
    /// Returns the most recent value, or the previous one if no new output
    /// has been published since the last call.
    pub fn latest_energies(&self) -> [f32; 3];
}
```

### Triple buffer

An atomic triple buffer provides wait-free communication between the graph
thread (writer) and the render thread (reader):

```
slot[0]  ──  slot[1]  ──  slot[2]
  ↑                           ↑
writer                      reader
(graph thread)           (render thread)
```

One slot is always owned by the writer, one by the reader, and one is the
"clean" slot that can be atomically swapped.  The writer publishes by swapping
its slot with the clean slot; the reader acquires by swapping its slot with the
clean slot.  Neither side ever blocks.

### Interface compatibility

The `InputHandle` / `OutputHandle` API is unchanged.  The `Executor` struct is
reused inside `GraphRunner` — only the threading wrapper is new.

The `AudioSource` trait is also unchanged.  `CpalCapture` and `SynthSource`
both implement it and can be passed to `GraphRunner::new` without modification.

### Migration path for `voice_metaballs`

```rust
// Phase 5 (current)
if let Some(frame) = self.audio_source.next_frame() {
    self.executor.input("audio")?.write("audio", frame)?;
    self.executor.run()?;
    let e = self.executor.output("energies")?.read().unwrap();
    // ... animate ...
}

// Phase 6 (future)
let e = self.graph_runner.latest_energies();
// ... animate ...
```

The render thread no longer calls `executor.run()` at all — it just reads the
latest published value.

## Nix / dependency notes

Phase 6 adds no new external dependencies.  The triple buffer is implemented
in pure Rust using `AtomicUsize` (same approach as `RingBuffer`).  `cpal` and
`alsa-lib` are already present from Phase 5.

## See also

- [`runtime::audio`](../runtime/src/audio/mod.rs) — `AudioSource` trait,
  `SynthSource`, `CpalCapture`
- [`runtime::audio::ringbuf`](../runtime/src/audio/ringbuf.rs) — SPSC ring
  buffer used between cpal callback and frame consumer
- [`runtime::executor`](../runtime/src/executor/mod.rs) — synchronous executor
  (Phase 2–5)
