const std = @import("std");
const stir = @import("stir");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("STIR - Shift to Improve Rate\n", .{});
    std.debug.print("============================\n\n", .{});

    // Create test parameters
    const params = stir.Parameters{
        .security_level = 32,
        .protocol_security_level = 16,
        .starting_degree = 64,
        .stopping_degree = 8,
        .folding_factor = 4,
        .starting_rate = 2,
        .soundness_type = .Conjecture,
    };

    std.debug.print("Parameters:\n", .{});
    std.debug.print("  Security level: {d}\n", .{params.security_level});
    std.debug.print("  Starting degree: {d}\n", .{params.starting_degree});
    std.debug.print("  Stopping degree: {d}\n", .{params.stopping_degree});
    std.debug.print("  Folding factor: {d}\n", .{params.folding_factor});
    std.debug.print("  Starting rate: 2^-{d}\n\n", .{params.starting_rate});

    var full_params = try stir.FullParameters.init(allocator, params);
    defer full_params.deinit();

    std.debug.print("Derived parameters:\n", .{});
    std.debug.print("  Number of rounds: {d}\n", .{full_params.num_rounds});
    std.debug.print("  OOD samples: {d}\n\n", .{full_params.ood_samples});

    // Create a test polynomial
    const poly_degree: usize = 32;
    var coeffs = try allocator.alloc(stir.Field, poly_degree);
    defer allocator.free(coeffs);
    for (0..poly_degree) |i| {
        coeffs[i] = stir.Field.init(@intCast(i + 1));
    }

    var polynomial = try stir.DensePolynomial.init(allocator, coeffs);
    defer polynomial.deinit();

    std.debug.print("Test polynomial: degree {d}\n\n", .{polynomial.degree()});

    // Create prover and commit
    var prover_inst = stir.StirProver.init(allocator, full_params);

    std.debug.print("Committing to polynomial...\n", .{});
    const commit_result = try prover_inst.commit(polynomial);
    var witness = commit_result.witness;
    defer witness.deinit();

    std.debug.print("Commitment (Merkle root): ", .{});
    for (commit_result.commitment.root) |byte| {
        std.debug.print("{x:0>2}", .{byte});
    }
    std.debug.print("\n\n", .{});

    std.debug.print("STIR implementation complete!\n", .{});
    std.debug.print("Run `zig build test` to run all tests.\n", .{});
    std.debug.print("Run `zig build bench` to run benchmarks.\n", .{});
}

test "main module imports" {
    // Verify that stir module can be imported
    _ = stir.Field;
    _ = stir.DensePolynomial;
    _ = stir.MerkleTree;
    _ = stir.StirProver;
    _ = stir.StirVerifier;
}
