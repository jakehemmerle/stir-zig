const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const Domain = @import("domain.zig").Domain;
const MerkleTree = @import("merkle.zig").MerkleTree;
const Transcript = @import("transcript.zig").Transcript;
const proofOfWorkVerify = @import("transcript.zig").proofOfWorkVerify;
const FullParameters = @import("parameters.zig").FullParameters;
const interpolation = @import("interpolation.zig");
const quotient_mod = @import("quotient.zig");
const utils = @import("utils.zig");
const prover = @import("prover.zig");
const Point = @import("types.zig").Point;
const Commitment = prover.Commitment;
const Proof = prover.Proof;
const RoundProof = prover.RoundProof;
const Allocator = std.mem.Allocator;

/// Virtual function for verification
const VirtualFunction = struct {
    comb_randomness: Field,
    interpolating_polynomial: DensePolynomial,
    quotient_set: []Field,
    allocator: Allocator,

    fn deinit(self: *VirtualFunction) void {
        self.interpolating_polynomial.deinit();
        self.allocator.free(self.quotient_set);
    }
};

/// Oracle type (initial or virtual)
const OracleType = union(enum) {
    Initial: void,
    Virtual: VirtualFunction,

    fn deinit(self: *OracleType) void {
        switch (self.*) {
            .Initial => {},
            .Virtual => |*vf| vf.deinit(),
        }
    }
};

/// Verification state
const VerificationState = struct {
    oracle: OracleType,
    domain_gen: Field,
    domain_size: usize,
    domain_offset: Field,
    root_of_unity: Field,
    folding_randomness: Field,
    num_round: usize,

    fn deinit(self: *VerificationState) void {
        self.oracle.deinit();
    }

    /// Query the oracle at an evaluation point
    fn query(
        self: VerificationState,
        evaluation_point: Field,
        value_of_prev_oracle: Field,
        common_factor_inverse: Field,
        denom_hint: Field,
        ans_eval: Field,
    ) Field {
        switch (self.oracle) {
            .Initial => return value_of_prev_oracle,
            .Virtual => |vf| {
                const num_terms = vf.quotient_set.len;
                const quotient_evaluation = quotient_mod.quotientWithHint(
                    value_of_prev_oracle,
                    evaluation_point,
                    vf.quotient_set,
                    denom_hint,
                    ans_eval,
                ) orelse Field.ZERO;

                const common_factor = evaluation_point.mul(vf.comb_randomness);

                const scale_factor = if (!common_factor.eq(Field.ONE))
                    Field.ONE.sub(common_factor.pow(num_terms + 1)).mul(common_factor_inverse)
                else
                    Field.init(@intCast(num_terms + 1));

                return quotient_evaluation.mul(scale_factor);
            },
        }
    }
};

/// STIR Verifier
pub const StirVerifier = struct {
    params: FullParameters,
    allocator: Allocator,

    pub fn init(allocator: Allocator, params: FullParameters) StirVerifier {
        return .{
            .params = params,
            .allocator = allocator,
        };
    }

    /// Verify a proof against a commitment
    pub fn verify(self: *StirVerifier, commitment: Commitment, proof_arg: *Proof) !bool {
        // Check final polynomial degree
        if (proof_arg.final_polynomial.degree() + 1 > self.params.stoppingDegree()) {
            return false;
        }

        // Verify all Merkle paths
        var current_root = commitment.root;
        for (proof_arg.round_proofs) |*round_proof| {
            const valid = round_proof.queries_to_prev_proof.verify(
                current_root,
                round_proof.queries_to_prev_ans,
            );
            if (!valid) return false;
            current_root = round_proof.g_root;
        }

        // Verify final queries
        const final_valid = proof_arg.queries_to_final_proof.verify(
            current_root,
            proof_arg.queries_to_final_ans,
        );
        if (!final_valid) return false;

        // Recompute Fiat-Shamir transcript
        var transcript = Transcript.init();
        transcript.absorbRoot(commitment.root);
        const folding_randomness = transcript.squeezeField();

        const domain = try Domain.init(self.params.startingDegree(), self.params.startingRate());

        var verification_state = VerificationState{
            .oracle = .Initial,
            .domain_gen = domain.generator,
            .domain_size = domain.size,
            .domain_offset = Field.ONE,
            .root_of_unity = domain.generator,
            .num_round = 0,
            .folding_randomness = folding_randomness,
        };

        // Process each round
        for (proof_arg.round_proofs) |*round_proof| {
            const new_state = try self.round(&transcript, round_proof, verification_state) orelse return false;
            verification_state.deinit();
            verification_state = new_state;
        }
        defer verification_state.deinit();

        // Final verification
        const final_repetitions = self.params.repetitions[self.params.num_rounds];
        const scaling_factor = verification_state.domain_size / self.params.foldingFactor();

        var raw_indices = try self.allocator.alloc(usize, final_repetitions);
        defer self.allocator.free(raw_indices);
        for (0..final_repetitions) |i| {
            raw_indices[i] = transcript.squeezeInteger(scaling_factor);
        }

        const final_indices = try utils.dedup(self.allocator, raw_indices);
        defer self.allocator.free(final_indices);

        // PoW verification
        if (!proofOfWorkVerify(&transcript, self.params.pow_bits[self.params.num_rounds], proof_arg.pow_nonce)) {
            return false;
        }

        // Compute folded evaluations and check against final polynomial
        const folded_answers = try self.computeFoldedEvaluations(
            verification_state,
            final_indices,
            proof_arg.queries_to_final_ans,
        );
        defer {
            for (folded_answers) |*fa| {
                self.allocator.free(fa.f_answers);
            }
            self.allocator.free(folded_answers);
        }

        for (folded_answers) |fa| {
            const expected = proof_arg.final_polynomial.evaluate(fa.point);
            if (!expected.eq(fa.value)) {
                return false;
            }
        }

        return true;
    }

    const FoldedAnswer = struct {
        point: Field,
        value: Field,
        f_answers: []Field,
    };

    fn computeFoldedEvaluations(
        self: *StirVerifier,
        state: VerificationState,
        indices: []const usize,
        oracle_answers: []const []const Field,
    ) ![]FoldedAnswer {
        const scaling_factor = state.domain_size / self.params.foldingFactor();
        const generator = state.domain_gen.pow(@intCast(scaling_factor));
        _ = generator.inv() orelse return error.InversionFailed;
        _ = Field.init(@intCast(self.params.foldingFactor())).inv() orelse return error.InversionFailed;

        var results = try self.allocator.alloc(FoldedAnswer, indices.len);
        errdefer self.allocator.free(results);

        for (indices, 0..) |idx, i| {
            const coset_offset = state.domain_offset.mul(state.domain_gen.pow(@intCast(idx)));
            _ = coset_offset.inv() orelse continue;

            // Build query set
            var query_set = try self.allocator.alloc(Field, self.params.foldingFactor());
            defer self.allocator.free(query_set);
            var scale = Field.ONE;
            for (0..self.params.foldingFactor()) |j| {
                query_set[j] = coset_offset.mul(scale);
                scale = scale.mul(generator);
            }

            // Query the oracle at each point in the coset
            var f_answers = try self.allocator.alloc(Field, self.params.foldingFactor());

            for (0..self.params.foldingFactor()) |j| {
                const common_factor_inverse = switch (state.oracle) {
                    .Initial => Field.ONE,
                    .Virtual => |vf| Field.ONE.sub(vf.comb_randomness.mul(query_set[j])).inv() orelse Field.ONE,
                };

                const denom_hint = switch (state.oracle) {
                    .Initial => Field.ONE,
                    .Virtual => |vf| blk: {
                        var d = Field.ONE;
                        for (vf.quotient_set) |x| {
                            d = d.mul(query_set[j].sub(x));
                        }
                        break :blk d.inv() orelse Field.ONE;
                    },
                };

                const ans_eval = switch (state.oracle) {
                    .Initial => Field.ONE,
                    .Virtual => |vf| vf.interpolating_polynomial.evaluate(query_set[j]),
                };

                f_answers[j] = state.query(
                    query_set[j],
                    oracle_answers[i][j],
                    common_factor_inverse,
                    denom_hint,
                    ans_eval,
                );
            }

            // Fold the answers using FFT interpolation
            var interpolated = try interpolation.fftInterpolate(
                self.allocator,
                generator,
                coset_offset,
                f_answers,
            );
            defer interpolated.deinit();

            const folded_value = interpolated.evaluate(state.folding_randomness);

            // Compute the actual query point (x^k)
            const scaled_offset = state.domain_offset.pow(@intCast(self.params.foldingFactor()));
            const query_point = scaled_offset.mul(
                state.domain_gen.pow(@intCast(self.params.foldingFactor() * idx)),
            );

            results[i] = .{
                .point = query_point,
                .value = folded_value,
                .f_answers = f_answers,
            };
        }

        return results;
    }

    fn round(
        self: *StirVerifier,
        transcript: *Transcript,
        round_proof: *RoundProof,
        state: VerificationState,
    ) !?VerificationState {
        // Redo Fiat-Shamir
        transcript.absorbRoot(round_proof.g_root);
        const ood_randomness = transcript.squeezeFields(2);
        transcript.absorbFields(round_proof.betas);
        const comb_randomness = transcript.squeezeField();
        const new_folding_randomness = transcript.squeezeField();

        const scaling_factor = state.domain_size / self.params.foldingFactor();
        const num_repetitions = self.params.repetitions[state.num_round];

        var raw_indices = try self.allocator.alloc(usize, num_repetitions);
        defer self.allocator.free(raw_indices);
        for (0..num_repetitions) |i| {
            raw_indices[i] = transcript.squeezeInteger(scaling_factor);
        }

        const stir_randomness_indices = try utils.dedup(self.allocator, raw_indices);
        defer self.allocator.free(stir_randomness_indices);

        // PoW verification
        if (!proofOfWorkVerify(transcript, self.params.pow_bits[state.num_round], round_proof.pow_nonce)) {
            return null;
        }

        _ = transcript.squeezeField(); // shake randomness

        // Compute folded answers
        var folded_answers = try self.computeFoldedEvaluations(
            state,
            stir_randomness_indices,
            round_proof.queries_to_prev_ans,
        );
        defer {
            for (folded_answers) |*fa| {
                self.allocator.free(fa.f_answers);
            }
            self.allocator.free(folded_answers);
        }

        // Build quotient answers
        var quotient_answers = try self.allocator.alloc(Point, self.params.ood_samples + folded_answers.len);
        defer self.allocator.free(quotient_answers);

        for (0..self.params.ood_samples) |i| {
            quotient_answers[i] = .{ .x = ood_randomness[i], .y = round_proof.betas[i] };
        }
        for (folded_answers, 0..) |fa, i| {
            quotient_answers[self.params.ood_samples + i] = .{ .x = fa.point, .y = fa.value };
        }

        // Clone interpolating polynomial
        const interpolating_polynomial = try round_proof.ans_polynomial.clone();

        // Build quotient set
        var quotient_set = try self.allocator.alloc(Field, quotient_answers.len);
        for (quotient_answers, 0..) |qa, i| {
            quotient_set[i] = qa.x;
        }

        return VerificationState{
            .oracle = .{
                .Virtual = VirtualFunction{
                    .comb_randomness = comb_randomness,
                    .interpolating_polynomial = interpolating_polynomial,
                    .quotient_set = quotient_set,
                    .allocator = self.allocator,
                },
            },
            .domain_size = state.domain_size / 2,
            .domain_gen = state.domain_gen.mul(state.domain_gen),
            .domain_offset = state.domain_offset.mul(state.domain_offset).mul(state.root_of_unity),
            .root_of_unity = state.root_of_unity,
            .folding_randomness = new_folding_randomness,
            .num_round = state.num_round + 1,
        };
    }
};

// Tests
test "verifier initialization" {
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

    const verifier = StirVerifier.init(allocator, full_params);
    _ = verifier;
}
