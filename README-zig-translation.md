# Ralph Ziggum: STIR Rust-to-Zig Translation Experiment

> **WARNING: This code is heavily vibe-coded.** It could be completely wrong and should not be used for anything other than failure analysis. This was an experiment in AI-assisted code translation, not a production implementation.

## Overview

This experiment used **Ralph Wiggum** (dubbed "Ralph Ziggum" for this Zig-focused effort) - an iterative AI-assisted code translation technique - to translate the [STIR](https://github.com/stir-protocol/stir) Rust implementation into Zig. The goal was to explore whether a Zig implementation could achieve better performance than the Rust version.

**Tools used:**
- Claude Code (CLI)
- Claude Opus 4.5
- Ralph Wiggum technique for iterative translation and refinement

## Motivation

STIR (Shift To Improve Rate) is a low-degree testing protocol used in proof systems. The Rust implementation relies heavily on the `ark-ff` and `ark-poly` libraries with ASM-optimized field arithmetic. We hypothesized that a hand-tuned Zig implementation might reduce overhead and improve performance.

## What Was Built

The Zig implementation includes:
- **Field arithmetic** (`src/field.zig`) - 192-bit prime field with Montgomery representation
- **FFT** (`src/fft.zig`) - Cooley-Tukey radix-2 FFT
- **Merkle trees** (`src/merkle.zig`) - Blake3-based Merkle tree
- **Polynomial operations** (`src/polynomial.zig`) - Evaluation, interpolation, folding
- **Prover/Verifier** (`src/prover.zig`, `src/verifier.zig`) - Full STIR protocol
- **Fiat-Shamir transcript** (`src/transcript.zig`) - Blake3-based

## Alignment Efforts

To enable fair comparison, we aligned the implementations:

1. **Hash function**: Changed Rust from SHA3-256 to Blake3 (matching Zig)
2. **Field arithmetic**: Implemented Montgomery representation in Zig (matching Rust's ark-ff)
3. **Benchmark parameters**: Aligned degree (2^18), folding factor (16), security level (128-bit)

## Results

**Benchmark: Degree 2^18, Folding Factor 16, 128-bit security**

| Metric | Rust | Zig | Ratio |
|--------|------|-----|-------|
| Prover Time | 640.65 ms | 1,385.52 ms | **Rust 2.2x faster** |
| Verifier Time | 1.44 ms | 8.22 ms | **Rust 5.7x faster** |
| Proof Size | 74,707 bytes | 56,640 bytes | Zig 24% smaller |
| Prover Hashes | 229,373 | 229,373 | Equal |

### Conclusion: The Zig implementation is significantly slower and likely incorrect.

**Do not use this code for anything other than studying what went wrong.**

## Why the Zig Version is Slower

Despite implementing Montgomery arithmetic and aligning the protocol parameters, several gaps remain:

### 1. ASM-Optimized Field Arithmetic
The Rust `ark-ff` library uses hand-written x86-64 assembly for Montgomery multiplication. The Zig implementation uses a pure Zig CIOS (Coarsely Integrated Operand Scanning) algorithm that cannot match the performance of specialized assembly.

### 2. FFT Implementation
Rust's `ark-poly` uses a cache-optimized FFT with precomputed twiddle factors. The Zig FFT is a straightforward Cooley-Tukey implementation without these optimizations.

### 3. Batch Inversion
Rust uses Montgomery's trick for batch field inversions (1 inversion + n-2 multiplications for n elements). Zig performs individual inversions, which is particularly costly for the verifier.

### 4. Library Maturity
The `ark-*` ecosystem has years of optimization. A from-scratch Zig implementation cannot match this without equivalent effort.

## Lessons Learned

1. **Language choice matters less than library quality**: Rust's performance advantage comes from the highly-optimized `ark-*` libraries, not from language-level features.

2. **Montgomery arithmetic is necessary but not sufficient**: While adding Montgomery representation to Zig was essential for a fair comparison, it doesn't close the gap without ASM optimization.

3. **Translation is easier than optimization**: Ralph Wiggum/Claude Code effectively translated the algorithmic structure, but achieving competitive performance requires deep optimization work beyond translation.

4. **Some wins exist**: The Zig proof size is ~24% smaller, suggesting potential serialization efficiency. The hash counts match for the prover, indicating algorithmic correctness.

## Files

```
src/
├── field.zig          # Field192Mont - Montgomery field arithmetic
├── fft.zig            # Cooley-Tukey FFT
├── polynomial.zig     # Polynomial operations
├── merkle.zig         # Blake3 Merkle tree
├── prover.zig         # STIR prover
├── verifier.zig       # STIR verifier
├── transcript.zig     # Fiat-Shamir transcript
├── domain.zig         # Evaluation domains
├── folding.zig        # BS08 polynomial folding
├── parameters.zig     # Protocol parameters
├── quotient.zig       # Quotient polynomial computation
├── interpolation.zig  # Lagrange interpolation
├── types.zig          # Shared type definitions
├── utils.zig          # Utility functions
└── benchmark.zig      # Benchmark harness
```

## Running the Benchmarks

```bash
# Zig benchmark
zig build bench

# Rust benchmark (with Blake3)
cargo run --release --bin stir -- -d 18 -k 16 --reps 10
```

## Future Work

To make the Zig implementation competitive, one would need to:
- Port `ark-ff` x86-64 assembly to Zig
- Implement batch inversion
- Add cache-optimized FFT
- Profile and optimize hot paths

This would require significant engineering effort and may not be justified given the mature Rust ecosystem.

---

*This experiment was conducted using Claude Code with Opus 4.5, January 2025.*
