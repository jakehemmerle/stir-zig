//! Benchmarks for STIR components
//!
//! Run with: zig build bench
//! Or run directly: zig run src/benchmark.zig

const std = @import("std");
const time = std.time;

const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const Domain = @import("domain.zig").Domain;
const merkle = @import("merkle.zig");
const MerkleTree = merkle.MerkleTree;
const Transcript = @import("transcript.zig").Transcript;
const fft_mod = @import("fft.zig");
const interpolation = @import("interpolation.zig");
const folding = @import("folding.zig");
const utils = @import("utils.zig");
const Parameters = @import("parameters.zig").Parameters;
const FullParameters = @import("parameters.zig").FullParameters;
const StirProver = @import("prover.zig").StirProver;
const StirVerifier = @import("verifier.zig").StirVerifier;
const Proof = @import("prover.zig").Proof;
const Commitment = @import("prover.zig").Commitment;
const Point = @import("types.zig").Point;

const Allocator = std.mem.Allocator;

const ITERATIONS = 100;
const VERIFIER_REPS = 100;

/// Benchmark configuration
/// Default values match Rust's benchmark for fair comparison
const BenchmarkConfig = struct {
    security_level: usize = 128,
    protocol_security_level: usize = 106,
    initial_degree: usize = 18, // log2, so 2^18 = 262144 (Rust default is 20)
    final_degree: usize = 6, // log2, so 2^6 = 64
    rate: usize = 2,
    folding_factor: usize = 16,
    verifier_reps: usize = 1000, // Match Rust default
    run_component_benchmarks: bool = true,
    run_protocol_benchmarks: bool = true,
};

fn parseArgs() BenchmarkConfig {
    var config = BenchmarkConfig{};

    var args = std.process.args();
    _ = args.skip(); // skip program name

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--degree")) {
            if (args.next()) |val| {
                config.initial_degree = std.fmt.parseInt(usize, val, 10) catch config.initial_degree;
            }
        } else if (std.mem.eql(u8, arg, "-f") or std.mem.eql(u8, arg, "--final-degree")) {
            if (args.next()) |val| {
                config.final_degree = std.fmt.parseInt(usize, val, 10) catch config.final_degree;
            }
        } else if (std.mem.eql(u8, arg, "-r") or std.mem.eql(u8, arg, "--rate")) {
            if (args.next()) |val| {
                config.rate = std.fmt.parseInt(usize, val, 10) catch config.rate;
            }
        } else if (std.mem.eql(u8, arg, "-k") or std.mem.eql(u8, arg, "--folding-factor")) {
            if (args.next()) |val| {
                config.folding_factor = std.fmt.parseInt(usize, val, 10) catch config.folding_factor;
            }
        } else if (std.mem.eql(u8, arg, "--reps")) {
            if (args.next()) |val| {
                config.verifier_reps = std.fmt.parseInt(usize, val, 10) catch config.verifier_reps;
            }
        } else if (std.mem.eql(u8, arg, "--protocol-only")) {
            config.run_component_benchmarks = false;
        } else if (std.mem.eql(u8, arg, "--components-only")) {
            config.run_protocol_benchmarks = false;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            std.debug.print(
                \\STIR Zig Benchmarks
                \\
                \\Usage: stir-bench [OPTIONS]
                \\
                \\Options:
                \\  -d, --degree <N>         Initial degree as log2 (default: 12, meaning 2^12=4096)
                \\  -f, --final-degree <N>   Final degree as log2 (default: 6, meaning 2^6=64)
                \\  -r, --rate <N>           Starting rate (default: 2)
                \\  -k, --folding-factor <N> Folding factor (default: 16)
                \\  --reps <N>               Verifier repetitions (default: 100)
                \\  --protocol-only          Run only protocol benchmarks
                \\  --components-only        Run only component benchmarks
                \\  -h, --help               Show this help
                \\
            , .{});
            std.process.exit(0);
        }
    }

    return config;
}

fn getTimestampNs() i128 {
    const timer = time.Timer.start() catch return 0;
    return timer.read();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = parseArgs();

    std.debug.print("\n=== STIR Zig Benchmarks ===\n\n", .{});

    if (!config.run_component_benchmarks) {
        std.debug.print("(Skipping component benchmarks)\n\n", .{});
    } else {
        try runComponentBenchmarks(allocator);
    }

    if (config.run_protocol_benchmarks) {
        std.debug.print("\n--- End-to-End Protocol ---\n", .{});
        try runProtocolBenchmarkWithConfig(allocator, config);
    }

    std.debug.print("\n=== Benchmark Complete ===\n", .{});
}

fn runComponentBenchmarks(allocator: Allocator) !void {
    // Field arithmetic benchmarks
    std.debug.print("--- Field Arithmetic ---\n", .{});
    const a = Field.init(12345678901234567);
    const b = Field.init(98765432109876543);

    // Field multiplication
    {
        var timer = try time.Timer.start();
        for (0..ITERATIONS * 1000) |_| {
            _ = a.mul(b);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 1000);
        std.debug.print("Field multiplication: {d} ns\n", .{avg_ns});
    }

    // Field inversion
    {
        var timer = try time.Timer.start();
        for (0..ITERATIONS * 100) |_| {
            _ = a.inv();
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 100);
        std.debug.print("Field inversion: {d} ns\n", .{avg_ns});
    }

    // Field exponentiation
    {
        var timer = try time.Timer.start();
        for (0..ITERATIONS * 10) |_| {
            _ = a.pow(0xFFFFFFFFFFFFFFFF);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 10);
        std.debug.print("Field exponentiation (64-bit): {d} ns\n", .{avg_ns});
    }

    // FFT benchmarks
    std.debug.print("\n--- FFT ---\n", .{});

    inline for ([_]usize{ 64, 256, 1024, 4096 }) |size| {
        const log_size = @ctz(size);
        const root = Field.getRootOfUnity(log_size).?;

        var coeffs = try allocator.alloc(Field, size);
        defer allocator.free(coeffs);
        for (0..size) |i| {
            coeffs[i] = Field.init(@intCast(i + 1));
        }

        var timer = try time.Timer.start();
        for (0..ITERATIONS) |_| {
            const result = try fft_mod.fftRadix2(allocator, coeffs, root);
            allocator.free(result);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / ITERATIONS;
        std.debug.print("FFT size {d}: {d} ns ({d:.2} us)\n", .{ size, avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Polynomial operations
    std.debug.print("\n--- Polynomial Operations ---\n", .{});

    inline for ([_]usize{ 64, 256, 1024 }) |deg| {
        var coeffs = try allocator.alloc(Field, deg);
        defer allocator.free(coeffs);
        for (0..deg) |i| {
            coeffs[i] = Field.init(@intCast(i + 1));
        }

        var poly = try DensePolynomial.init(allocator, coeffs);
        defer poly.deinit();

        const point = Field.init(42);

        var timer = try time.Timer.start();
        for (0..ITERATIONS * 100) |_| {
            _ = poly.evaluate(point);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 100);
        std.debug.print("Poly eval (deg {d}): {d} ns ({d:.2} us)\n", .{ deg, avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Merkle tree benchmarks
    std.debug.print("\n--- Merkle Tree ---\n", .{});

    inline for ([_]usize{ 64, 256, 1024 }) |num_leaves| {
        var leaves = try allocator.alloc([]const Field, num_leaves);
        defer allocator.free(leaves);

        var leaf_data = try allocator.alloc(Field, num_leaves * 4);
        defer allocator.free(leaf_data);

        for (0..num_leaves) |i| {
            for (0..4) |j| {
                leaf_data[i * 4 + j] = Field.init(@intCast(i * 4 + j));
            }
            leaves[i] = leaf_data[i * 4 .. i * 4 + 4];
        }

        var timer = try time.Timer.start();
        for (0..ITERATIONS) |_| {
            var tree = try MerkleTree.init(allocator, leaves);
            tree.deinit();
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / ITERATIONS;
        std.debug.print("Merkle tree ({d} leaves): {d} ns ({d:.2} us)\n", .{ num_leaves, avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Interpolation benchmarks
    std.debug.print("\n--- Interpolation ---\n", .{});

    inline for ([_]usize{ 4, 8, 16 }) |num_points| {
        var points = try allocator.alloc(Point, num_points);
        defer allocator.free(points);

        for (0..num_points) |i| {
            points[i] = .{
                .x = Field.init(@intCast(i + 1)),
                .y = Field.init(@intCast(i * 2 + 3)),
            };
        }

        var timer = try time.Timer.start();
        for (0..ITERATIONS) |_| {
            var poly = try interpolation.naiveInterpolation(allocator, points);
            poly.deinit();
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / ITERATIONS;
        std.debug.print("Naive interpolation ({d} points): {d} ns ({d:.2} us)\n", .{ num_points, avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Folding benchmarks
    std.debug.print("\n--- Polynomial Folding ---\n", .{});

    inline for ([_]usize{ 64, 256, 1024 }) |deg| {
        var coeffs = try allocator.alloc(Field, deg);
        defer allocator.free(coeffs);
        for (0..deg) |i| {
            coeffs[i] = Field.init(@intCast(i + 1));
        }

        var poly = try DensePolynomial.init(allocator, coeffs);
        defer poly.deinit();

        const folding_randomness = Field.init(42);
        const folding_factor: usize = 4;

        var timer = try time.Timer.start();
        for (0..ITERATIONS) |_| {
            var folded = try folding.polyFold(allocator, poly, folding_factor, folding_randomness);
            folded.deinit();
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / ITERATIONS;
        std.debug.print("Poly fold (deg {d}, factor {d}): {d} ns ({d:.2} us)\n", .{ deg, folding_factor, avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Transcript benchmarks
    std.debug.print("\n--- Fiat-Shamir Transcript ---\n", .{});
    {
        var timer = try time.Timer.start();
        for (0..ITERATIONS * 100) |_| {
            var transcript = Transcript.init();
            transcript.absorbField(Field.init(42));
            _ = transcript.squeezeField();
            transcript.absorbField(Field.init(123));
            _ = transcript.squeezeFields(10);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 100);
        std.debug.print("Transcript ops: {d} ns ({d:.2} us)\n", .{ avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }

    // Domain benchmarks
    std.debug.print("\n--- Domain Operations ---\n", .{});
    {
        var timer = try time.Timer.start();
        for (0..ITERATIONS * 100) |_| {
            const domain = try Domain.init(1024, 2);
            _ = domain.scale(4);
            _ = domain.scaleOffset(2);
        }
        const elapsed = timer.read();
        const avg_ns = elapsed / (ITERATIONS * 100);
        std.debug.print("Domain creation + scaling: {d} ns ({d:.2} us)\n", .{ avg_ns, @as(f64, @floatFromInt(avg_ns)) / 1000.0 });
    }
}

/// Run protocol benchmark with user-specified configuration
fn runProtocolBenchmarkWithConfig(allocator: Allocator, config: BenchmarkConfig) !void {
    const starting_degree = @as(usize, 1) << @intCast(config.initial_degree);
    const stopping_degree = @as(usize, 1) << @intCast(config.final_degree);

    const params = Parameters{
        .security_level = config.security_level,
        .protocol_security_level = config.protocol_security_level,
        .starting_degree = starting_degree,
        .stopping_degree = stopping_degree,
        .folding_factor = config.folding_factor,
        .starting_rate = config.rate,
        .soundness_type = .Conjecture,
    };

    var full_params = try FullParameters.init(allocator, params);
    defer full_params.deinit();

    std.debug.print("\n  Degree 2^{d} ({d}), {d} rounds:\n", .{ config.initial_degree, starting_degree, full_params.num_rounds });
    displayParameters(full_params);

    // Create random polynomial
    var coeffs = try allocator.alloc(Field, starting_degree - 1);
    defer allocator.free(coeffs);
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    for (0..starting_degree - 1) |i| {
        // Field.init handles reduction, so just use a random u64
        coeffs[i] = Field.init(random.int(u64));
    }

    var polynomial = try DensePolynomial.init(allocator, coeffs);
    defer polynomial.deinit();

    var prover = StirProver.init(allocator, full_params);

    // Prover benchmark with detailed profiling
    merkle.resetHashCounter();

    std.debug.print("\n  --- Prover Profiling ---\n", .{});

    // Phase 1: Commit
    var commit_timer = try time.Timer.start();
    const commit_result = try prover.commit(polynomial);
    var witness = commit_result.witness;
    defer witness.deinit();
    const commitment = commit_result.commitment;
    const commit_elapsed = commit_timer.read();
    std.debug.print("    Commit: {d:.2} ms\n", .{@as(f64, @floatFromInt(commit_elapsed)) / 1_000_000.0});

    // Phase 2: Prove
    var prove_timer = try time.Timer.start();
    var proof = try prover.prove(witness);
    defer proof.deinit();
    const prove_elapsed = prove_timer.read();
    std.debug.print("    Prove: {d:.2} ms\n", .{@as(f64, @floatFromInt(prove_elapsed)) / 1_000_000.0});

    const prover_elapsed = commit_elapsed + prove_elapsed;
    const prover_hashes = merkle.getHashCount();

    const proof_size = estimateProofSize(proof);

    std.debug.print("    Prover time: {d:.2} ms\n", .{@as(f64, @floatFromInt(prover_elapsed)) / 1_000_000.0});
    std.debug.print("    Prover hashes: {d}\n", .{prover_hashes});
    std.debug.print("    Proof size: ~{d} bytes\n", .{proof_size});

    // Verifier benchmark
    var verifier = StirVerifier.init(allocator, full_params);
    merkle.resetHashCounter();

    const reps = config.verifier_reps;
    var verifier_timer = try time.Timer.start();
    for (0..reps) |_| {
        const result = verifier.verify(commitment, &proof) catch false;
        if (!result) {
            std.debug.print("    WARNING: Verification failed!\n", .{});
            break;
        }
    }
    const verifier_elapsed = verifier_timer.read();
    const verifier_hashes = merkle.getHashCount() / reps;

    std.debug.print("    Verifier time: {d:.2} ms (avg over {d} reps)\n", .{ @as(f64, @floatFromInt(verifier_elapsed / reps)) / 1_000_000.0, reps });
    std.debug.print("    Verifier hashes: {d}\n", .{verifier_hashes});
}

/// Display protocol parameters (similar to Rust's Stir::display)
fn displayParameters(params: FullParameters) void {
    std.debug.print("    Security level: {d}\n", .{params.params.security_level});
    std.debug.print("    Starting degree: {d}\n", .{params.startingDegree()});
    std.debug.print("    Stopping degree: {d}\n", .{params.stoppingDegree()});
    std.debug.print("    Folding factor: {d}\n", .{params.foldingFactor()});
    std.debug.print("    Starting rate: 1/{d}\n", .{@as(usize, 1) << @intCast(params.startingRate())});
    std.debug.print("    Num rounds: {d}\n", .{params.num_rounds});
    std.debug.print("    PoW bits: [", .{});
    for (params.pow_bits, 0..) |pb, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{pb});
    }
    std.debug.print("]\n", .{});

    // Benchmark PoW timing
    const pow_test_bits = params.pow_bits[0];
    var pow_timer = time.Timer.start() catch unreachable;
    var test_transcript = Transcript.init();
    test_transcript.absorbField(Field.init(12345));
    const pow_nonce = @import("transcript.zig").proofOfWork(&test_transcript, pow_test_bits);
    const pow_elapsed = pow_timer.read();
    std.debug.print("    PoW sample ({d} bits): {d:.2} ms (nonce: {?})\n", .{ pow_test_bits, @as(f64, @floatFromInt(pow_elapsed)) / 1_000_000.0, pow_nonce });
    std.debug.print("    Repetitions: [", .{});
    for (params.repetitions, 0..) |r, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d}", .{r});
    }
    std.debug.print("]\n", .{});
}

/// Estimate proof size in bytes
fn estimateProofSize(proof: Proof) usize {
    var size: usize = 0;

    // Round proofs
    for (proof.round_proofs) |rp| {
        // g_root (32 bytes)
        size += 32;
        // betas (8 bytes per field element)
        size += rp.betas.len * 8;
        // queries_to_prev_ans
        for (rp.queries_to_prev_ans) |ans| {
            size += ans.len * 8;
        }
        // queries_to_prev_proof (32 bytes per hash in path, per proof)
        if (rp.queries_to_prev_proof.proofs.len > 0) {
            size += rp.queries_to_prev_proof.proofs.len * rp.queries_to_prev_proof.proofs[0].path.len * 32;
        }
        // ans_polynomial
        size += rp.ans_polynomial.coeffs.len * 8;
        // shake_polynomial
        size += rp.shake_polynomial.coeffs.len * 8;
        // pow_nonce
        size += 8;
    }

    // Final polynomial
    size += proof.final_polynomial.coeffs.len * 8;

    // queries_to_final_ans
    for (proof.queries_to_final_ans) |ans| {
        size += ans.len * 8;
    }

    // queries_to_final_proof
    if (proof.queries_to_final_proof.proofs.len > 0) {
        size += proof.queries_to_final_proof.proofs.len * proof.queries_to_final_proof.proofs[0].path.len * 32;
    }

    // pow_nonce
    size += 8;

    return size;
}
