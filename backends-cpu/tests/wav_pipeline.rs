//! Integration test: full Window → FFT → BandExtract pipeline via Executor.
//!
//! Builds a three-node graph, wires it up with `GraphBuilder`, runs it through
//! the `Executor` with `CpuBackend`, and asserts that a 440 Hz sine wave
//! produces dominant energy in the mid-frequency band.
//!
//! Audio is generated synthetically in-process (no binary WAV fixture).
//! `hound` is used to round-trip the samples through a WAV encode/decode cycle
//! to exercise the full signal path.

use backends_cpu::CpuBackend;
use graph_core::graph::GraphBuilder;
use graph_core::ops::{
    BandDef, BandExtractParams, FftDirection, FftOutput, FftParams, Op, WindowKind, WindowParams,
};
use graph_core::types::{DType, Dim, Layout, TensorType};
use runtime::executor::Executor;

const SAMPLE_RATE: u32 = 8_000;
const FFT_SIZE: usize = 512;
const FREQ_HZ: f32 = 440.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn f32_vec(n: usize) -> TensorType {
    TensorType::new(DType::F32, vec![Dim::Fixed(n)], Layout::RowMajor).unwrap()
}

/// Generate a 440 Hz sine wave as `f32` samples.
fn sine_frame(freq: f32, sr: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
        .collect()
}

/// Encode samples to WAV bytes with `hound`, then decode back to `f32`.
/// This exercises the encode/decode path without needing a file on disk.
fn wav_roundtrip(samples: &[f32], sr: u32) -> Vec<f32> {
    use hound::{SampleFormat, WavSpec, WavWriter};

    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    // Encode to in-memory buffer.
    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut writer = WavWriter::new(cursor, spec).unwrap();
        for &s in samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    // Decode back.
    let cursor = std::io::Cursor::new(buf);
    let mut reader = hound::WavReader::new(cursor).unwrap();
    reader.samples::<f32>().map(|s| s.unwrap()).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn wav_pipeline_440hz_energy_in_mid_band() {
    // ── Build graph ───────────────────────────────────────────────────────
    let n_bins = FFT_SIZE / 2 + 1; // 257

    let window_params = WindowParams::new(WindowKind::Hann, FFT_SIZE).unwrap();
    let fft_params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    let bands = vec![
        BandDef::new(0.0, 250.0, "low").unwrap(),
        BandDef::new(250.0, 2000.0, "mid").unwrap(),
        BandDef::new(2000.0, 4000.0, "high").unwrap(),
    ];
    let band_params = BandExtractParams::new(bands, SAMPLE_RATE as f32, 0.0).unwrap();

    let frame_type = f32_vec(FFT_SIZE);
    let spectrum_type = f32_vec(n_bins);
    let energy_type = f32_vec(3);

    let graph = GraphBuilder::new()
        // Source: raw audio frame
        .source("audio", frame_type.clone())
        // Node 1: Window
        .add_node("window")
        .device("cpu")
        .op(Op::Window(window_params))
        .input_from_source("audio")
        .output(frame_type.clone())
        .done()
        // Node 2: FFT
        .add_node("fft")
        .device("cpu")
        .op(Op::Fft(fft_params))
        .input_from("window", 0)
        .output(spectrum_type.clone())
        .done()
        // Node 3: BandExtract
        .add_node("band")
        .device("cpu")
        .op(Op::BandExtract(band_params))
        .input_from("fft", 0)
        .output(energy_type.clone())
        .done()
        // Sink: band energies
        .sink("energies", energy_type.clone())
        .from("band", 0)
        .done()
        .build()
        .unwrap();

    // ── Build executor ────────────────────────────────────────────────────
    let backend: Box<dyn backends::Backend> = Box::new(CpuBackend::new("cpu"));
    let mut executor = Executor::new(graph, vec![backend]).unwrap();

    // ── Prepare input ─────────────────────────────────────────────────────
    let raw_samples = sine_frame(FREQ_HZ, SAMPLE_RATE, FFT_SIZE);
    let samples = wav_roundtrip(&raw_samples, SAMPLE_RATE);
    assert_eq!(samples.len(), FFT_SIZE);

    executor
        .input("audio")
        .unwrap()
        .write("audio", &samples)
        .unwrap();

    // ── Run ───────────────────────────────────────────────────────────────
    executor.run().unwrap();

    // ── Assert ────────────────────────────────────────────────────────────
    let energies: &[f32] = executor.output("energies").unwrap().read().unwrap();
    assert_eq!(energies.len(), 3);

    let (low, mid, high) = (energies[0], energies[1], energies[2]);
    assert!(
        mid > low && mid > high,
        "440 Hz should dominate the mid band: low={low}, mid={mid}, high={high}"
    );
}

#[test]
fn wav_pipeline_zero_input_gives_zero_energies() {
    let n_bins = FFT_SIZE / 2 + 1;

    let window_params = WindowParams::new(WindowKind::Hann, FFT_SIZE).unwrap();
    let fft_params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    let bands = vec![
        BandDef::new(0.0, 2000.0, "low").unwrap(),
        BandDef::new(2000.0, 4000.0, "high").unwrap(),
    ];
    let band_params = BandExtractParams::new(bands, SAMPLE_RATE as f32, 0.0).unwrap();

    let frame_type = f32_vec(FFT_SIZE);
    let spectrum_type = f32_vec(n_bins);
    let energy_type = f32_vec(2);

    let graph = GraphBuilder::new()
        .source("audio", frame_type.clone())
        .add_node("window")
        .device("cpu")
        .op(Op::Window(window_params))
        .input_from_source("audio")
        .output(frame_type.clone())
        .done()
        .add_node("fft")
        .device("cpu")
        .op(Op::Fft(fft_params))
        .input_from("window", 0)
        .output(spectrum_type.clone())
        .done()
        .add_node("band")
        .device("cpu")
        .op(Op::BandExtract(band_params))
        .input_from("fft", 0)
        .output(energy_type.clone())
        .done()
        .sink("energies", energy_type.clone())
        .from("band", 0)
        .done()
        .build()
        .unwrap();

    let backend: Box<dyn backends::Backend> = Box::new(CpuBackend::new("cpu"));
    let mut executor = Executor::new(graph, vec![backend]).unwrap();

    let zeros = vec![0.0f32; FFT_SIZE];
    executor
        .input("audio")
        .unwrap()
        .write("audio", &zeros)
        .unwrap();
    executor.run().unwrap();

    let energies: &[f32] = executor.output("energies").unwrap().read().unwrap();
    for &e in energies {
        assert!(
            e.abs() < 1e-6,
            "Zero input should give zero energy, got {e}"
        );
    }
}

#[test]
fn wav_pipeline_runs_multiple_ticks() {
    // Verify the executor can be called multiple times (stateless graph).
    let n_bins = FFT_SIZE / 2 + 1;

    let window_params = WindowParams::new(WindowKind::Hann, FFT_SIZE).unwrap();
    let fft_params = FftParams::new(FFT_SIZE, FftDirection::Forward, FftOutput::Magnitude).unwrap();
    let bands = vec![BandDef::new(0.0, 4000.0, "all").unwrap()];
    let band_params = BandExtractParams::new(bands, SAMPLE_RATE as f32, 0.0).unwrap();

    let frame_type = f32_vec(FFT_SIZE);
    let spectrum_type = f32_vec(n_bins);
    let energy_type = f32_vec(1);

    let graph = GraphBuilder::new()
        .source("audio", frame_type.clone())
        .add_node("window")
        .device("cpu")
        .op(Op::Window(window_params))
        .input_from_source("audio")
        .output(frame_type.clone())
        .done()
        .add_node("fft")
        .device("cpu")
        .op(Op::Fft(fft_params))
        .input_from("window", 0)
        .output(spectrum_type.clone())
        .done()
        .add_node("band")
        .device("cpu")
        .op(Op::BandExtract(band_params))
        .input_from("fft", 0)
        .output(energy_type.clone())
        .done()
        .sink("energies", energy_type.clone())
        .from("band", 0)
        .done()
        .build()
        .unwrap();

    let backend: Box<dyn backends::Backend> = Box::new(CpuBackend::new("cpu"));
    let mut executor = Executor::new(graph, vec![backend]).unwrap();

    for tick in 0..3 {
        let samples = sine_frame(FREQ_HZ * (1 + tick) as f32, SAMPLE_RATE, FFT_SIZE);
        executor
            .input("audio")
            .unwrap()
            .write("audio", &samples)
            .unwrap();
        executor.run().unwrap();
        let energies: &[f32] = executor.output("energies").unwrap().read().unwrap();
        assert_eq!(energies.len(), 1);
        assert!(energies[0] > 0.0, "Tick {tick}: expected positive energy");
    }
}
