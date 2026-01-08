const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const Domain = @import("domain.zig").Domain;
const MerkleTree = @import("merkle.zig").MerkleTree;
const MerkleProof = @import("merkle.zig").MerkleProof;
const MultiProof = @import("merkle.zig").MultiProof;
const Transcript = @import("transcript.zig").Transcript;
const proofOfWork = @import("transcript.zig").proofOfWork;
const FullParameters = @import("parameters.zig").FullParameters;
const fft_mod = @import("fft.zig");
const interpolation = @import("interpolation.zig");
const folding = @import("folding.zig");
const utils = @import("utils.zig");
const Point = @import("types.zig").Point;
const Allocator = std.mem.Allocator;

/// Commitment to a polynomial (Merkle root)
pub const Commitment = struct {
    root: MerkleTree.Hash,
};

/// Round proof structure
pub const RoundProof = struct {
    /// Merkle root of g polynomial
    g_root: MerkleTree.Hash,
    /// Out-of-domain evaluations
    betas: []Field,
    /// Merkle proof for queries to previous oracle
    queries_to_prev_ans: [][]Field,
    queries_to_prev_proof: MultiProof,
    /// Answer polynomial
    ans_polynomial: DensePolynomial,
    /// Shake polynomial for verification
    shake_polynomial: DensePolynomial,
    /// Proof of work nonce
    pow_nonce: ?usize,
    allocator: Allocator,

    pub fn deinit(self: *RoundProof) void {
        self.allocator.free(self.betas);
        for (self.queries_to_prev_ans) |ans| {
            self.allocator.free(ans);
        }
        self.allocator.free(self.queries_to_prev_ans);
        self.queries_to_prev_proof.deinit();
        self.ans_polynomial.deinit();
        self.shake_polynomial.deinit();
    }
};

/// Complete STIR proof
pub const Proof = struct {
    round_proofs: []RoundProof,
    final_polynomial: DensePolynomial,
    queries_to_final_ans: [][]Field,
    queries_to_final_proof: MultiProof,
    pow_nonce: ?usize,
    allocator: Allocator,

    pub fn deinit(self: *Proof) void {
        for (self.round_proofs) |*rp| {
            rp.deinit();
        }
        self.allocator.free(self.round_proofs);
        self.final_polynomial.deinit();
        for (self.queries_to_final_ans) |ans| {
            self.allocator.free(ans);
        }
        self.allocator.free(self.queries_to_final_ans);
        self.queries_to_final_proof.deinit();
    }
};

/// Witness for proving
pub const Witness = struct {
    domain: Domain,
    polynomial: DensePolynomial,
    merkle_tree: MerkleTree,
    folded_evals: [][]Field,
    allocator: Allocator,

    pub fn deinit(self: *Witness) void {
        self.polynomial.deinit();
        self.merkle_tree.deinit();
        utils.freeStacked(self.allocator, self.folded_evals);
    }
};

/// Extended witness for round computation
const WitnessExtended = struct {
    domain: Domain,
    polynomial: DensePolynomial,
    merkle_tree: MerkleTree,
    folded_evals: [][]Field,
    num_round: usize,
    folding_randomness: Field,
    allocator: Allocator,

    fn deinit(self: *WitnessExtended) void {
        self.polynomial.deinit();
        self.merkle_tree.deinit();
        utils.freeStacked(self.allocator, self.folded_evals);
    }
};

/// STIR Prover
pub const StirProver = struct {
    params: FullParameters,
    allocator: Allocator,

    pub fn init(allocator: Allocator, params: FullParameters) StirProver {
        return .{
            .params = params,
            .allocator = allocator,
        };
    }

    /// Commit to a polynomial
    pub fn commit(self: *StirProver, polynomial: DensePolynomial) !struct { commitment: Commitment, witness: Witness } {
        var timer = std.time.Timer.start() catch unreachable;

        const domain = try Domain.init(self.params.startingDegree(), self.params.startingRate());

        // Evaluate polynomial over domain
        const evals = try fft_mod.evaluateOverCoset(
            self.allocator,
            polynomial,
            domain.generator,
            domain.offset,
            domain.size,
        );
        defer self.allocator.free(evals);
        const fft_time = timer.read();

        // Stack evaluations for Merkle tree
        var folded_evals = try utils.stackEvaluations(self.allocator, evals, self.params.foldingFactor());

        // Build Merkle tree with stacked evaluations as leaves
        var leaves = try self.allocator.alloc([]const Field, folded_evals.len);
        defer self.allocator.free(leaves);
        for (folded_evals, 0..) |fe, i| {
            leaves[i] = fe;
        }

        var merkle_tree = try MerkleTree.init(self.allocator, leaves);
        const merkle_time = timer.read() - fft_time;

        const root = merkle_tree.root();

        const poly_clone = try polynomial.clone();

        std.debug.print("      [commit] FFT: {d:.2}ms, Merkle: {d:.2}ms\n", .{
            @as(f64, @floatFromInt(fft_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(merkle_time)) / 1_000_000.0,
        });

        return .{
            .commitment = .{ .root = root },
            .witness = .{
                .domain = domain,
                .polynomial = poly_clone,
                .merkle_tree = merkle_tree,
                .folded_evals = folded_evals,
                .allocator = self.allocator,
            },
        };
    }

    /// Generate a proof
    pub fn prove(self: *StirProver, witness: Witness) !Proof {
        std.debug.assert(witness.polynomial.degree() < self.params.startingDegree());
        var timer = std.time.Timer.start() catch unreachable;

        var transcript = Transcript.init();
        transcript.absorbRoot(witness.merkle_tree.root());
        const folding_randomness = transcript.squeezeField();

        var extended = WitnessExtended{
            .domain = witness.domain,
            .polynomial = try witness.polynomial.clone(),
            .merkle_tree = witness.merkle_tree,
            .folded_evals = witness.folded_evals,
            .num_round = 0,
            .folding_randomness = folding_randomness,
            .allocator = self.allocator,
        };

        var round_proofs = std.ArrayList(RoundProof){};
        defer round_proofs.deinit(self.allocator);

        var rounds_time: u64 = 0;
        for (0..self.params.num_rounds) |round_num| {
            var round_timer = std.time.Timer.start() catch unreachable;
            const result = try self.round(&transcript, &extended);
            extended.deinit();
            extended = result.new_witness;
            try round_proofs.append(self.allocator, result.round_proof);
            const round_elapsed = round_timer.read();
            rounds_time += round_elapsed;
            std.debug.print("      [prove] Round {d}: {d:.2}ms\n", .{ round_num, @as(f64, @floatFromInt(round_elapsed)) / 1_000_000.0 });
        }

        // Compute final polynomial
        var fold_timer = std.time.Timer.start() catch unreachable;
        const final_polynomial = try folding.polyFold(
            self.allocator,
            extended.polynomial,
            self.params.foldingFactor(),
            extended.folding_randomness,
        );
        const fold_time = fold_timer.read();

        // Final queries
        const final_repetitions = self.params.repetitions[self.params.num_rounds];
        const scaling_factor = extended.domain.size / self.params.foldingFactor();

        var indices = try self.allocator.alloc(usize, final_repetitions);
        defer self.allocator.free(indices);
        for (0..final_repetitions) |i| {
            indices[i] = transcript.squeezeInteger(scaling_factor);
        }

        var deduped_indices = try utils.dedup(self.allocator, indices);
        defer self.allocator.free(deduped_indices);

        var queries_to_final_ans = try self.allocator.alloc([]Field, deduped_indices.len);
        for (deduped_indices, 0..) |idx, i| {
            queries_to_final_ans[i] = try self.allocator.dupe(Field, extended.folded_evals[idx]);
        }

        const queries_to_final_proof = try extended.merkle_tree.generateMultiProof(deduped_indices);

        var pow_timer = std.time.Timer.start() catch unreachable;
        const pow_nonce = proofOfWork(&transcript, self.params.pow_bits[self.params.num_rounds]);
        const pow_time = pow_timer.read();

        const total_time = timer.read();
        std.debug.print("      [prove] Final fold: {d:.2}ms, PoW: {d:.2}ms, Total: {d:.2}ms\n", .{
            @as(f64, @floatFromInt(fold_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(pow_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(total_time)) / 1_000_000.0,
        });

        extended.polynomial.deinit();

        return Proof{
            .round_proofs = try round_proofs.toOwnedSlice(self.allocator),
            .final_polynomial = final_polynomial,
            .queries_to_final_ans = queries_to_final_ans,
            .queries_to_final_proof = queries_to_final_proof,
            .pow_nonce = pow_nonce,
            .allocator = self.allocator,
        };
    }

    fn round(self: *StirProver, transcript: *Transcript, witness: *WitnessExtended) !struct { new_witness: WitnessExtended, round_proof: RoundProof } {
        var timer = std.time.Timer.start() catch unreachable;

        // Fold the polynomial
        var g_poly = try folding.polyFold(
            self.allocator,
            witness.polynomial,
            self.params.foldingFactor(),
            witness.folding_randomness,
        );
        defer g_poly.deinit();
        const fold_time = timer.read();

        // Evaluate g over scaled domain
        const g_domain = witness.domain.scaleOffset(2);
        const g_evals = try fft_mod.evaluateOverCoset(
            self.allocator,
            g_poly,
            g_domain.generator,
            g_domain.offset,
            g_domain.size,
        );
        defer self.allocator.free(g_evals);
        const fft_time = timer.read() - fold_time;

        var g_folded_evals = try utils.stackEvaluations(self.allocator, g_evals, self.params.foldingFactor());

        // Build g's Merkle tree
        var g_leaves = try self.allocator.alloc([]const Field, g_folded_evals.len);
        defer self.allocator.free(g_leaves);
        for (g_folded_evals, 0..) |fe, i| {
            g_leaves[i] = fe;
        }

        var g_merkle = try MerkleTree.init(self.allocator, g_leaves);
        const merkle_time = timer.read() - fold_time - fft_time;

        const g_root = g_merkle.root();
        transcript.absorbRoot(g_root);

        // Out of domain sampling
        var betas = try self.allocator.alloc(Field, self.params.ood_samples);
        const ood_randomness = transcript.squeezeFields(2);
        for (0..self.params.ood_samples) |i| {
            betas[i] = g_poly.evaluate(ood_randomness[i]);
        }
        transcript.absorbFields(betas);

        // Combination randomness
        const comb_randomness = transcript.squeezeField();
        const new_folding_randomness = transcript.squeezeField();

        // Sample query indices
        const scaling_factor = witness.domain.size / self.params.foldingFactor();
        const num_repetitions = self.params.repetitions[witness.num_round];

        var raw_indices = try self.allocator.alloc(usize, num_repetitions);
        defer self.allocator.free(raw_indices);
        for (0..num_repetitions) |i| {
            raw_indices[i] = transcript.squeezeInteger(scaling_factor);
        }

        var stir_randomness_indices = try utils.dedup(self.allocator, raw_indices);
        defer self.allocator.free(stir_randomness_indices);

        const pre_pow_time = timer.read();
        const pow_nonce = proofOfWork(transcript, self.params.pow_bits[witness.num_round]);
        const pow_time = timer.read() - pre_pow_time;

        // Squeeze shake randomness (not used but must advance transcript)
        _ = transcript.squeezeField();

        // Query previous oracle
        var queries_to_prev_ans = try self.allocator.alloc([]Field, stir_randomness_indices.len);
        for (stir_randomness_indices, 0..) |idx, i| {
            queries_to_prev_ans[i] = try self.allocator.dupe(Field, witness.folded_evals[idx]);
        }

        const queries_to_prev_proof = try witness.merkle_tree.generateMultiProof(stir_randomness_indices);

        // Compute STIR randomness points
        const scaled_domain = witness.domain.scale(self.params.foldingFactor());
        var stir_randomness = try self.allocator.alloc(Field, stir_randomness_indices.len);
        defer self.allocator.free(stir_randomness);
        for (stir_randomness_indices, 0..) |idx, i| {
            stir_randomness[i] = scaled_domain.element(idx);
        }

        // Build quotient answers
        var quotient_answers = try self.allocator.alloc(Point, stir_randomness.len + self.params.ood_samples);
        defer self.allocator.free(quotient_answers);

        for (stir_randomness, 0..) |x, i| {
            quotient_answers[i] = .{ .x = x, .y = g_poly.evaluate(x) };
        }
        for (0..self.params.ood_samples) |i| {
            quotient_answers[stir_randomness.len + i] = .{ .x = ood_randomness[i], .y = betas[i] };
        }

        // Compute answer polynomial
        const pre_interp_time = timer.read();
        const ans_polynomial = try interpolation.naiveInterpolation(self.allocator, quotient_answers);
        const interp_time = timer.read() - pre_interp_time;

        // Compute quotient set
        var quotient_set = try self.allocator.alloc(Field, quotient_answers.len);
        defer self.allocator.free(quotient_set);
        for (quotient_answers, 0..) |qa, i| {
            quotient_set[i] = qa.x;
        }

        // Compute shake polynomial (for verification)
        var shake_poly_coeffs = try self.allocator.alloc(Field, 1);
        shake_poly_coeffs[0] = Field.ZERO;
        const shake_polynomial = DensePolynomial{ .coeffs = shake_poly_coeffs, .allocator = self.allocator };

        // Compute vanishing polynomial and quotient
        const pre_div_time = timer.read();
        var vanishing = try interpolation.vanishingPoly(self.allocator, quotient_set);
        defer vanishing.deinit();

        var numerator = try g_poly.add(ans_polynomial);
        defer numerator.deinit();

        const div_result = try numerator.divMod(vanishing);
        var quotient_polynomial = div_result.quotient;
        defer quotient_polynomial.deinit();
        var remainder = div_result.remainder;
        defer remainder.deinit();
        const div_time = timer.read() - pre_div_time;

        // Scaling polynomial: 1 + r*x + r^2*x^2 + ...
        var scaling_coeffs = try self.allocator.alloc(Field, quotient_set.len + 1);
        var r_pow = Field.ONE;
        for (0..quotient_set.len + 1) |i| {
            scaling_coeffs[i] = r_pow;
            r_pow = r_pow.mul(comb_randomness);
        }
        var scaling_poly = DensePolynomial{ .coeffs = scaling_coeffs, .allocator = self.allocator };
        defer scaling_poly.deinit();

        const witness_polynomial = try quotient_polynomial.mul(scaling_poly);

        const total_time = timer.read();
        std.debug.print("        [round] fold:{d:.1}ms fft:{d:.1}ms merkle:{d:.1}ms pow:{d:.1}ms interp:{d:.1}ms div:{d:.1}ms total:{d:.1}ms\n", .{
            @as(f64, @floatFromInt(fold_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(fft_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(merkle_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(pow_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(interp_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(div_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(total_time)) / 1_000_000.0,
        });

        return .{
            .new_witness = WitnessExtended{
                .domain = g_domain,
                .polynomial = witness_polynomial,
                .merkle_tree = g_merkle,
                .folded_evals = g_folded_evals,
                .num_round = witness.num_round + 1,
                .folding_randomness = new_folding_randomness,
                .allocator = self.allocator,
            },
            .round_proof = RoundProof{
                .g_root = g_root,
                .betas = betas,
                .queries_to_prev_ans = queries_to_prev_ans,
                .queries_to_prev_proof = queries_to_prev_proof,
                .ans_polynomial = ans_polynomial,
                .shake_polynomial = shake_polynomial,
                .pow_nonce = pow_nonce,
                .allocator = self.allocator,
            },
        };
    }
};

// Tests
test "prover commit" {
    const allocator = std.testing.allocator;

    const params = @import("parameters.zig").Parameters{
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

    var prover = StirProver.init(allocator, full_params);

    // Create a simple polynomial
    var coeffs = try allocator.alloc(Field, 32);
    defer allocator.free(coeffs);
    for (0..32) |i| {
        coeffs[i] = Field.init(@intCast(i + 1));
    }

    var polynomial = try DensePolynomial.init(allocator, coeffs);
    defer polynomial.deinit();

    const result = try prover.commit(polynomial);
    var witness = result.witness;
    defer witness.deinit();

    // Commitment should have a valid root
    try std.testing.expect(!std.mem.eql(u8, &result.commitment.root, &MerkleTree.EMPTY_HASH));
}
