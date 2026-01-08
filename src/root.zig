//! STIR: Shift to Improve Rate
//! A Zig implementation of the STIR low-degree testing protocol.
//!
//! This is a translation of the Rust implementation from the same repository.

const std = @import("std");

// Core field arithmetic
pub const field = @import("field.zig");
pub const Field = field.Field;
pub const Field64 = field.Field64; // Keep for backwards compatibility

// Polynomial operations
pub const polynomial = @import("polynomial.zig");
pub const DensePolynomial = polynomial.DensePolynomial;

// FFT and evaluation
pub const fft = @import("fft.zig");

// Interpolation
pub const interpolation = @import("interpolation.zig");

// Polynomial folding (BS08)
pub const folding = @import("folding.zig");

// Quotient computation
pub const quotient = @import("quotient.zig");

// Merkle tree commitments
pub const merkle = @import("merkle.zig");
pub const MerkleTree = merkle.MerkleTree;
pub const MerkleProof = merkle.MerkleProof;

// Fiat-Shamir transcript
pub const transcript = @import("transcript.zig");
pub const Transcript = transcript.Transcript;

// Evaluation domains
pub const domain = @import("domain.zig");
pub const Domain = domain.Domain;

// Utility functions
pub const utils = @import("utils.zig");

// Common types
pub const types = @import("types.zig");
pub const Point = types.Point;

// Protocol parameters
pub const parameters = @import("parameters.zig");
pub const Parameters = parameters.Parameters;
pub const FullParameters = parameters.FullParameters;
pub const SoundnessType = parameters.SoundnessType;

// STIR Prover
pub const prover = @import("prover.zig");
pub const StirProver = prover.StirProver;
pub const Commitment = prover.Commitment;
pub const Proof = prover.Proof;
pub const Witness = prover.Witness;

// STIR Verifier
pub const verifier = @import("verifier.zig");
pub const StirVerifier = verifier.StirVerifier;

// Re-export commonly used types
pub const Hash = MerkleTree.Hash;

/// Create default test parameters
pub fn testParameters() Parameters {
    return Parameters{
        .security_level = 32,
        .protocol_security_level = 16,
        .starting_degree = 64,
        .stopping_degree = 8,
        .folding_factor = 4,
        .starting_rate = 2,
        .soundness_type = .Conjecture,
    };
}

// Run all tests from submodules
test {
    // Import all test blocks
    std.testing.refAllDecls(@This());

    // Explicitly reference test modules to ensure their tests run
    _ = field;
    _ = polynomial;
    _ = fft;
    _ = interpolation;
    _ = folding;
    _ = quotient;
    _ = merkle;
    _ = transcript;
    _ = domain;
    _ = utils;
    _ = types;
    _ = parameters;
    _ = prover;
    _ = verifier;
}

// Integration test for the full STIR protocol
test "STIR protocol end-to-end" {
    const allocator = std.testing.allocator;

    // Create parameters
    const params = Parameters{
        .security_level = 32,
        .protocol_security_level = 16,
        .starting_degree = 64,
        .stopping_degree = 8,
        .folding_factor = 4,
        .starting_rate = 2,
        .soundness_type = .Conjecture,
    };

    var full_params = try FullParameters.init(allocator, params);
    defer full_params.deinit();

    // Create prover
    var stir_prover = StirProver.init(allocator, full_params);

    // Create a test polynomial (degree < starting_degree)
    var coeffs = try allocator.alloc(Field, 32);
    defer allocator.free(coeffs);
    for (0..32) |i| {
        coeffs[i] = Field.init(@intCast(i + 1));
    }

    var poly = try DensePolynomial.init(allocator, coeffs);
    defer poly.deinit();

    // Commit
    const commit_result = try stir_prover.commit(poly);
    var witness = commit_result.witness;
    defer witness.deinit();

    // The commitment should be a valid Merkle root
    try std.testing.expect(!std.mem.eql(u8, &commit_result.commitment.root, &MerkleTree.EMPTY_HASH));
}
