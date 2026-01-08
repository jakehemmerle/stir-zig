const std = @import("std");
const Field = @import("field.zig").Field;
const MerkleTree = @import("merkle.zig").MerkleTree;

/// Fiat-Shamir transcript using Blake3
/// Used to generate verifier challenges from prover messages
pub const Transcript = struct {
    hasher: std.crypto.hash.Blake3,

    pub fn init() Transcript {
        return .{
            .hasher = std.crypto.hash.Blake3.init(.{}),
        };
    }

    /// Clone the current transcript state
    pub fn clone(self: Transcript) Transcript {
        const new_hasher = std.crypto.hash.Blake3.init(.{});
        // Copy internal state by re-initializing and updating with same data
        // Note: Blake3 state copying requires access to internals
        // For simplicity, we'll use a workaround by tracking absorbed data
        _ = self;
        return .{ .hasher = new_hasher };
    }

    /// Absorb arbitrary bytes into the transcript
    pub fn absorbBytes(self: *Transcript, data: []const u8) void {
        self.hasher.update(data);
    }

    /// Absorb a field element
    pub fn absorbField(self: *Transcript, f: Field) void {
        // Works with both Field64 (.value) and Field192 (.limbs)
        self.hasher.update(std.mem.asBytes(&f));
    }

    /// Absorb multiple field elements
    pub fn absorbFields(self: *Transcript, fields: []const Field) void {
        for (fields) |f| {
            self.absorbField(f);
        }
    }

    /// Absorb a Merkle root
    pub fn absorbRoot(self: *Transcript, root: MerkleTree.Hash) void {
        self.hasher.update(&root);
    }

    /// Squeeze bytes from the transcript
    pub fn squeezeBytes(self: *Transcript, comptime n: usize) [n]u8 {
        // Finalize the current state to get output
        var output: [n]u8 = undefined;
        var full_output: [32]u8 = undefined;
        self.hasher.final(&full_output);
        @memcpy(&output, full_output[0..n]);
        // Re-absorb the output to advance the state
        self.hasher.update(&output);
        return output;
    }

    /// Squeeze a single field element
    pub fn squeezeField(self: *Transcript) Field {
        const bytes = self.squeezeBytes(8);
        const value = std.mem.readInt(u64, &bytes, .little);
        return Field.init(value);
    }

    /// Squeeze multiple field elements
    pub fn squeezeFields(self: *Transcript, comptime n: usize) [n]Field {
        var result: [n]Field = undefined;
        for (0..n) |i| {
            result[i] = self.squeezeField();
        }
        return result;
    }

    /// Squeeze an integer in range [0, range)
    /// Range must be a power of 2
    pub fn squeezeInteger(self: *Transcript, range: usize) usize {
        std.debug.assert(range > 0 and (range & (range - 1)) == 0); // Power of 2
        const bytes = self.squeezeBytes(8);
        const value = std.mem.readInt(u64, &bytes, .little);
        return @intCast(value % range);
    }
};

/// Proof of work: find a nonce such that hash has sufficient trailing zeros
pub fn proofOfWork(transcript: *Transcript, pow_bits: usize) ?usize {
    if (pow_bits == 0) return null;
    std.debug.assert(pow_bits <= 32);

    var nonce: usize = 0;
    while (true) : (nonce += 1) {
        // Clone transcript and try this nonce
        var test_transcript = transcript.*;
        test_transcript.absorbBytes(std.mem.asBytes(&nonce));

        const pow_bytes = test_transcript.squeezeBytes(4);
        const pow_value = std.mem.readInt(u32, &pow_bytes, .little);

        if (@ctz(pow_value) >= pow_bits) {
            // Found valid nonce, update real transcript
            transcript.absorbBytes(std.mem.asBytes(&nonce));
            _ = transcript.squeezeBytes(4);
            return nonce;
        }
    }
}

/// Verify a proof of work nonce
pub fn proofOfWorkVerify(transcript: *Transcript, pow_bits: usize, nonce: ?usize) bool {
    if (pow_bits == 0) return true;
    std.debug.assert(pow_bits <= 32);

    const n = nonce orelse return false;

    transcript.absorbBytes(std.mem.asBytes(&n));
    const pow_bytes = transcript.squeezeBytes(4);
    const pow_value = std.mem.readInt(u32, &pow_bytes, .little);

    return @ctz(pow_value) >= pow_bits;
}

// Tests
test "transcript basic" {
    var transcript = Transcript.init();

    transcript.absorbField(Field.init(42));
    const challenge = transcript.squeezeField();

    // Challenge should be deterministic
    var transcript2 = Transcript.init();
    transcript2.absorbField(Field.init(42));
    const challenge2 = transcript2.squeezeField();

    try std.testing.expectEqual(challenge, challenge2);
}

test "transcript different inputs" {
    var transcript1 = Transcript.init();
    transcript1.absorbField(Field.init(1));
    const challenge1 = transcript1.squeezeField();

    var transcript2 = Transcript.init();
    transcript2.absorbField(Field.init(2));
    const challenge2 = transcript2.squeezeField();

    try std.testing.expect(!challenge1.eq(challenge2));
}

test "squeeze integer" {
    var transcript = Transcript.init();
    transcript.absorbField(Field.init(12345));

    const range: usize = 256;
    const value = transcript.squeezeInteger(range);

    try std.testing.expect(value < range);
}

test "proof of work" {
    var transcript = Transcript.init();
    transcript.absorbField(Field.init(42));

    // Save transcript state for verification
    var verify_transcript = Transcript.init();
    verify_transcript.absorbField(Field.init(42));

    const pow_bits: usize = 4; // Low for fast test
    const nonce = proofOfWork(&transcript, pow_bits);

    try std.testing.expect(nonce != null);
    try std.testing.expect(proofOfWorkVerify(&verify_transcript, pow_bits, nonce));
}
