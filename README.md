# STIR: Reed-Solomon Proximity Testing with Fewer Queries

This repository contains:
1. **Rust implementation** - The original academic prototype using [arkworks](https://arkworks.rs)
2. **Zig implementation** - An experimental translation of STIR to Zig

Based on [STIR: Reed-Solomon Proximity Testing with Fewer Queries](https://eprint.iacr.org/2024/390) by Gal Arnon, Alessandro Chiesa, Giacomo Fenzi, and Eylon Yogev. See also the [blog post](https://gfenzi.io/papers/stir).

## Zig Implementation

The Zig implementation is a **direct translation** of the Rust STIR protocol.

**WARNING:** This Zig code is:
- Untested beyond basic smoke tests
- "Vibe-coded" - translated without rigorous verification
- NOT suitable for production use
- Intended for experimentation and benchmarking only

The Zig implementation uses:
- **Field192** (same 192-bit field as Rust) - configurable in `field.zig`
- **Blake3** for both Merkle trees and Fiat-Shamir transcript
- Standard library implementations (no external dependencies)

## Benchmark Comparison

This is now an **apples-to-apples comparison** using the same Field192.

### Test Parameters
- Degree: 2^12 (4096)
- Stopping degree: 2^6 (64)
- Folding factor: 16
- Security level: 128 bits
- 1 STIR round

### Prover Profiling (Round 0 Breakdown)

| Component | Zig (Field192) | Rust (Field192) | Notes |
|-----------|----------------|-----------------|-------|
| Fold | 1.2 ms | 0.2 ms | |
| FFT | 112.5 ms | 1.0 ms | Zig: commit 242.5ms |
| Merkle | 0.5 ms | 0.8 ms | Blake3 vs SHA3 |
| **PoW (22-bit)** | **24.2 ms** | **4544.9 ms** | **Stack vs heap allocation** |
| Interpolation | 33.6 ms | 1.0 ms | |
| Division | 15.9 ms | 0.4 ms | |

### End-to-End Results

| Metric | Zig (Field192) | Rust (Field192) |
|--------|----------------|-----------------|
| Commit time | 243.5 ms | 4.5 ms |
| Prove time | 219.4 ms | 4549 ms |
| **Total prover** | **462.9 ms** | **4627 ms** |
| Verifier time | 3.98 ms | 2.09 ms |
| Proof size | ~32 KB | ~46 KB |
| Prover hashes | 3070 | 3070 |

### Analysis

The **10x prover speedup** primarily comes from:

1. **Proof-of-Work allocation overhead**: The Rust Blake3 sponge allocates heap memory on every `absorb()` and `squeeze_bytes()` call. For 22-bit PoW (~4 million iterations), this creates 12+ million allocations. The Zig transcript uses stack-allocated buffers with zero heap allocations in the hot loop. (Zig: 24ms vs Rust: 4545ms)

2. **Merkle hashing**: Blake3 vs SHA3.

However, Zig's Field192 arithmetic is notably **slower** than Rust's arkworks:
- FFT: 112.5ms (Zig) vs 1.0ms (Rust) - ~100x slower
- This is due to Zig using naive schoolbook multiplication and Fermat's Little Theorem for inversion, while arkworks uses optimized Montgomery representation

To switch Zig to the faster Field64 (for experimentation), change `pub const Field = Field192;` to `pub const Field = Field64;` in `field.zig`.

## Building and Running

### Prerequisites

- **Zig**: 0.14+ (tested with 0.14.0)
- **Rust**: 1.70+ with Cargo

### Zig

```bash
# Build
zig build -Doptimize=ReleaseFast

# Run benchmarks (component + protocol)
./zig-out/bin/stir-bench

# Run protocol benchmarks only
./zig-out/bin/stir-bench --protocol-only

# Custom parameters
./zig-out/bin/stir-bench -d 12 -f 6 -k 16 --reps 100

# Run tests
zig build test
```

**Benchmark options:**
```
-d, --degree <N>         Initial degree as log2 (default: 12)
-f, --final-degree <N>   Final degree as log2 (default: 6)
-k, --folding-factor <N> Folding factor (default: 16)
--reps <N>               Verifier repetitions (default: 100)
--protocol-only          Run only protocol benchmarks
--components-only        Run only component benchmarks
```

### Rust

```bash
# Build
cargo build --release

# Run STIR benchmark
cargo run --release --bin stir -- -d 12 -f 6 -k 16 --reps 100

# Run with profiling output
cargo run --release --bin stir -- -d 12 -f 6 -k 16 --reps 100 --profile

# Run FRI benchmark (for comparison)
cargo run --release --bin fri -- -d 12 -f 6 -k 16 --reps 100
```

**Benchmark options:**
```
-d, --initial-degree <N>  Initial degree as log2 (default: 20)
-f, --final-degree <N>    Final degree as log2 (default: 6)
-k, --folding-factor <N>  Folding factor (default: 16)
-r, --rate <N>            Starting rate (default: 2)
--reps <N>                Verifier repetitions (default: 1000)
--profile                 Enable detailed profiling output
```

## Reproducing Benchmarks

To reproduce the head-to-head comparison:

```bash
# Terminal 1: Run Zig benchmark
zig build -Doptimize=ReleaseFast
./zig-out/bin/stir-bench -d 12 -f 6 -k 16 --reps 100 --protocol-only

# Terminal 2: Run Rust benchmark with profiling
cargo build --release
./target/release/stir -d 12 -f 6 -k 16 --reps 100 --profile
```

## Project Structure

```
src/
├── bin/                  # Rust binaries
│   ├── stir.rs          # STIR benchmark
│   ├── fri.rs           # FRI benchmark
│   └── ...
├── stir/                 # Rust STIR implementation
├── fri/                  # Rust FRI implementation
├── crypto/               # Rust crypto primitives
│
├── benchmark.zig         # Zig benchmarks
├── prover.zig           # Zig STIR prover
├── verifier.zig         # Zig STIR verifier
├── field.zig            # Zig field arithmetic (Field64 + Field192)
├── fft.zig              # Zig FFT
├── merkle.zig           # Zig Merkle tree (Blake3)
├── transcript.zig       # Zig Fiat-Shamir transcript
├── polynomial.zig       # Zig polynomial operations
├── folding.zig          # Zig polynomial folding
├── interpolation.zig    # Zig interpolation
├── quotient.zig         # Zig quotient computation
├── domain.zig           # Zig evaluation domains
├── parameters.zig       # Zig protocol parameters
└── types.zig            # Zig type definitions
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

## Acknowledgments

- Original STIR paper authors: Gal Arnon, Alessandro Chiesa, Giacomo Fenzi, Eylon Yogev
- [arkworks](https://arkworks.rs) ecosystem for the Rust implementation foundation
