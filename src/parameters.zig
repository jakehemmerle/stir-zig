const std = @import("std");
const Field = @import("field.zig").Field;
const utils = @import("utils.zig");
const Allocator = std.mem.Allocator;

/// Soundness type for security analysis
pub const SoundnessType = enum {
    Provable,
    Conjecture,
};

/// Base parameters for STIR protocol
pub const Parameters = struct {
    /// Target security level in bits
    security_level: usize,
    /// Protocol security level
    protocol_security_level: usize,
    /// Starting polynomial degree
    starting_degree: usize,
    /// Stopping degree (when to output final polynomial)
    stopping_degree: usize,
    /// Folding factor (must be power of 2)
    folding_factor: usize,
    /// Starting rate as log2(1/rho)
    starting_rate: usize,
    /// Soundness type
    soundness_type: SoundnessType,

    /// Calculate number of repetitions for given log inverse rate
    pub fn repetitions(self: Parameters, log_inv_rate: usize) usize {
        const constant: usize = switch (self.soundness_type) {
            .Provable => 2,
            .Conjecture => 1,
        };
        const num = constant * self.protocol_security_level;
        return (num + log_inv_rate - 1) / log_inv_rate; // ceil division
    }

    /// Calculate proof of work bits for given log inverse rate
    pub fn powBits(self: Parameters, log_inv_rate: usize) usize {
        const reps = self.repetitions(log_inv_rate);
        const scaling_factor: f64 = switch (self.soundness_type) {
            .Provable => 2.0,
            .Conjecture => 1.0,
        };
        const achieved_security: f64 = (@as(f64, @floatFromInt(log_inv_rate)) / scaling_factor) * @as(f64, @floatFromInt(reps));
        const remaining = @as(f64, @floatFromInt(self.security_level)) - achieved_security;

        if (remaining <= 0.0) {
            return 0;
        }
        return @intFromFloat(@ceil(remaining));
    }
};

/// Full parameters including derived values
pub const FullParameters = struct {
    params: Parameters,
    /// Number of folding rounds
    num_rounds: usize,
    /// Rates at each round
    rates: []usize,
    /// Number of repetitions at each round
    repetitions: []usize,
    /// PoW bits at each round
    pow_bits: []usize,
    /// OOD (out-of-domain) samples
    ood_samples: usize,
    /// Degrees at each round
    degrees: []usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, params: Parameters) !FullParameters {
        std.debug.assert(utils.isPowerOfTwo(params.folding_factor));
        std.debug.assert(utils.isPowerOfTwo(params.starting_degree));
        std.debug.assert(utils.isPowerOfTwo(params.stopping_degree));

        // Calculate number of rounds
        var d = params.starting_degree;
        var degrees_list = std.ArrayList(usize){};
        try degrees_list.ensureTotalCapacity(allocator, 16);
        defer degrees_list.deinit(allocator);
        try degrees_list.append(allocator, d);

        var num_rounds: usize = 0;
        while (d > params.stopping_degree) {
            std.debug.assert(d % params.folding_factor == 0);
            d /= params.folding_factor;
            try degrees_list.append(allocator, d);
            num_rounds += 1;
        }

        num_rounds -= 1;
        _ = degrees_list.pop(); // Remove last degree

        const degrees = try allocator.dupe(usize, degrees_list.items);

        // Calculate rates
        const log_folding = @ctz(params.folding_factor);
        var rates = try allocator.alloc(usize, num_rounds + 1);
        rates[0] = params.starting_rate;
        for (1..num_rounds + 1) |i| {
            rates[i] = params.starting_rate + i * (log_folding - 1);
        }

        // Calculate pow_bits and repetitions
        var pow_bits = try allocator.alloc(usize, num_rounds + 1);
        var repetitions = try allocator.alloc(usize, num_rounds + 1);

        for (0..num_rounds + 1) |i| {
            pow_bits[i] = params.powBits(rates[i]);
            repetitions[i] = params.repetitions(rates[i]);
        }

        // Cap repetitions
        for (0..num_rounds) |i| {
            repetitions[i] = @min(repetitions[i], degrees[i] / params.folding_factor);
        }

        return FullParameters{
            .params = params,
            .num_rounds = num_rounds,
            .rates = rates,
            .repetitions = repetitions,
            .pow_bits = pow_bits,
            .ood_samples = 2,
            .degrees = degrees,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FullParameters) void {
        self.allocator.free(self.rates);
        self.allocator.free(self.repetitions);
        self.allocator.free(self.pow_bits);
        self.allocator.free(self.degrees);
    }

    /// Access base parameter fields
    pub fn securityLevel(self: FullParameters) usize {
        return self.params.security_level;
    }

    pub fn startingDegree(self: FullParameters) usize {
        return self.params.starting_degree;
    }

    pub fn stoppingDegree(self: FullParameters) usize {
        return self.params.stopping_degree;
    }

    pub fn foldingFactor(self: FullParameters) usize {
        return self.params.folding_factor;
    }

    pub fn startingRate(self: FullParameters) usize {
        return self.params.starting_rate;
    }
};

// Tests
test "parameters basic" {
    const params = Parameters{
        .security_level = 128,
        .protocol_security_level = 80,
        .starting_degree = 1024,
        .stopping_degree = 16,
        .folding_factor = 4,
        .starting_rate = 3,
        .soundness_type = .Conjecture,
    };

    try std.testing.expect(params.repetitions(3) > 0);
    try std.testing.expect(params.powBits(3) >= 0);
}

test "full parameters" {
    const allocator = std.testing.allocator;

    const params = Parameters{
        .security_level = 128,
        .protocol_security_level = 80,
        .starting_degree = 256,
        .stopping_degree = 16,
        .folding_factor = 4,
        .starting_rate = 3,
        .soundness_type = .Conjecture,
    };

    var full = try FullParameters.init(allocator, params);
    defer full.deinit();

    // Check that we have the right number of rounds
    try std.testing.expect(full.num_rounds > 0);
    try std.testing.expectEqual(@as(usize, 256), full.startingDegree());
    try std.testing.expectEqual(@as(usize, 16), full.stoppingDegree());
}
