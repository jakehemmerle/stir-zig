# Ralph Ziggum: STIR Rust-to-Zig Translation Experiment

> **WARNING: This code is completely vibe-coded.** It could be completely wrong and should not be used for anything other than failure analysis, and I cannot even confirm that the benchmarks are accurate. This was an experiment in AI-assisted code translation, not a production implementation.

## Overview

This experiment used **Ralph Wiggum** (we should call it Ralph Ziggum) - an iterative AI-assisted code translation technique - to translate the [STIR](https://github.com/stir-protocol/stir) Rust implementation into Zig. The goal was to explore whether a Ralph Wiggum-translated Zig implementation could achieve correctness that was equivalent (or close to) the Rust version and/or better performance.

**Tools used:**
- Claude Code (CLI)
- Claude Opus 4.5

**Versions tested:**
- Zig: 0.16.0-dev
- Rust: 1.92.0

## Results

**Benchmark: Degree 2^18, Folding Factor 16, 128-bit security**

| Metric | Rust | Zig | Ratio |
|--------|------|-----|-------|
| Prover Time | 640.65 ms | 1,385.52 ms | **Rust 2.2x faster** |
| Verifier Time | 1.44 ms | 8.22 ms | **Rust 5.7x faster** |
| Proof Size | 74,707 bytes | 56,640 bytes | Different (possible bug) |
| Prover Hashes | 229,373 | 229,373 | Equal |

### Conclusion: The Zig implementation is significantly slower and likely incorrect.

The proof size difference suggests the implementations may not be equivalent. The matching hash counts for the prover are encouraging but not conclusive.

**Do not use this code for anything other than studying what went wrong.**

## What Was Built

The Zig implementation includes:
- **Field arithmetic** (`src/field.zig`) - 192-bit prime field with Montgomery representation
- **FFT** (`src/fft.zig`) - Cooley-Tukey radix-2 FFT
- **Merkle trees** (`src/merkle.zig`) - Blake3-based Merkle tree
- **Polynomial operations** (`src/polynomial.zig`) - Evaluation, interpolation, folding
- **Prover/Verifier** (`src/prover.zig`, `src/verifier.zig`) - STIR protocol
- **Fiat-Shamir transcript** (`src/transcript.zig`) - Blake3-based

## Alignment Efforts

To enable comparison, we attempted to align the implementations:

1. **Hash function**: Changed Rust from SHA3-256 to Blake3 (matching Zig)
2. **Field arithmetic**: Implemented Montgomery representation in Zig (attempting to match Rust's ark-ff)
3. **Benchmark parameters**: Aligned degree (2^18), folding factor (16), security level (128-bit)

## Possible Reasons for Performance Gap

These are hypotheses, not verified causes:

1. **ASM-Optimized Field Arithmetic**: Rust's `ark-ff` uses x86-64 assembly. Zig uses pure Zig.
2. **FFT Implementation**: Rust's `ark-poly` may have optimizations the Zig version lacks.
3. **Batch Inversion**: Rust may use Montgomery's trick; Zig performs individual inversions.

## Running the Benchmarks

```bash
# Zig benchmark
zig build bench

# Rust benchmark (with Blake3)
cargo run --release --bin stir -- -d 18 -k 16 --reps 10
```

## Files

```
src/
├── field.zig          # Montgomery field arithmetic
├── fft.zig            # FFT
├── polynomial.zig     # Polynomial operations
├── merkle.zig         # Blake3 Merkle tree
├── prover.zig         # STIR prover
├── verifier.zig       # STIR verifier
├── transcript.zig     # Fiat-Shamir transcript
├── domain.zig         # Evaluation domains
├── folding.zig        # Polynomial folding
├── parameters.zig     # Protocol parameters
├── quotient.zig       # Quotient polynomial computation
├── interpolation.zig  # Lagrange interpolation
├── types.zig          # Shared type definitions
├── utils.zig          # Utility functions
└── benchmark.zig      # Benchmark harness
```

---

*This experiment was conducted using Claude Code with Opus 4.5, January 2025.*
