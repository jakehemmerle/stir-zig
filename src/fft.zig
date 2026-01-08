const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const Allocator = std.mem.Allocator;

/// FFT evaluation that works when polynomial degree is larger than domain size
/// This is the recursive FFT from the Rust implementation
pub fn fft(allocator: Allocator, coeffs: []const Field, generator: Field, coset_offset: Field, size: usize) ![]Field {
    const BASE_THRESHOLD = 1;

    if (size <= BASE_THRESHOLD) {
        // Direct evaluation
        var poly = try DensePolynomial.init(allocator, coeffs);
        defer poly.deinit();

        var evaluations = try allocator.alloc(Field, size);
        var scale = Field.ONE;
        for (0..size) |i| {
            evaluations[i] = poly.evaluate(coset_offset.mul(scale));
            scale = scale.mul(generator);
        }
        return evaluations;
    }

    // Pad coefficients to next power of two
    const next_pow2 = std.math.ceilPowerOfTwo(usize, coeffs.len) catch coeffs.len;
    var padded_coeffs = try allocator.alloc(Field, next_pow2);
    defer allocator.free(padded_coeffs);

    @memcpy(padded_coeffs[0..coeffs.len], coeffs);
    @memset(padded_coeffs[coeffs.len..], Field.ZERO);

    // Split into odd and even
    const half_len = next_pow2 / 2;
    var odd = try allocator.alloc(Field, half_len);
    defer allocator.free(odd);
    var even = try allocator.alloc(Field, half_len);
    defer allocator.free(even);

    for (0..half_len) |i| {
        even[i] = padded_coeffs[2 * i];
        odd[i] = padded_coeffs[2 * i + 1];
    }

    const gen2 = generator.mul(generator);
    const off2 = coset_offset.mul(coset_offset);
    const size2 = size / 2;

    var odd_evals = try fft(allocator, odd, gen2, off2, size2);
    defer allocator.free(odd_evals);
    var even_evals = try fft(allocator, even, gen2, off2, size2);
    defer allocator.free(even_evals);

    var result = try allocator.alloc(Field, size);
    var scale = Field.ONE;
    for (0..size) |i| {
        const even_val = even_evals[i % even_evals.len];
        const odd_val = odd_evals[i % odd_evals.len];
        result[i] = even_val.add(coset_offset.mul(scale).mul(odd_val));
        scale = scale.mul(generator);
    }

    return result;
}

/// Standard radix-2 FFT for power-of-two sizes
pub fn fftRadix2(allocator: Allocator, coeffs: []const Field, root_of_unity: Field) ![]Field {
    const n = coeffs.len;
    std.debug.assert(n > 0 and (n & (n - 1)) == 0); // Must be power of 2

    var result = try allocator.dupe(Field, coeffs);

    // Bit-reversal permutation
    const log_n = @ctz(n);
    for (0..n) |i| {
        const j = bitReverse(i, log_n);
        if (i < j) {
            const temp = result[i];
            result[i] = result[j];
            result[j] = temp;
        }
    }

    // Cooley-Tukey iterative FFT
    var len: usize = 2;
    while (len <= n) : (len *= 2) {
        const half_len = len / 2;
        // Compute the primitive root for this stage
        var w = root_of_unity;
        const steps = n / len;
        for (0..@ctz(steps)) |_| {
            w = w.mul(w);
        }

        var start: usize = 0;
        while (start < n) : (start += len) {
            var omega = Field.ONE;
            for (0..half_len) |j| {
                const u = result[start + j];
                const v = result[start + j + half_len].mul(omega);
                result[start + j] = u.add(v);
                result[start + j + half_len] = u.sub(v);
                omega = omega.mul(w);
            }
        }
    }

    return result;
}

/// Inverse FFT
pub fn ifftRadix2(allocator: Allocator, evals: []const Field, root_of_unity: Field) ![]Field {
    const n = evals.len;
    const root_inv = root_of_unity.inv() orelse return error.InversionFailed;

    const result = try fftRadix2(allocator, evals, root_inv);

    // Scale by 1/n
    const n_inv = Field.init(@intCast(n)).inv() orelse return error.InversionFailed;
    for (result) |*r| {
        r.* = r.mul(n_inv);
    }

    return result;
}

fn bitReverse(x: usize, bits: usize) usize {
    var result: usize = 0;
    var val = x;
    for (0..bits) |_| {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

/// Evaluate polynomial over a coset (offset * <generator>)
pub fn evaluateOverCoset(allocator: Allocator, poly: DensePolynomial, generator: Field, coset_offset: Field, size: usize) ![]Field {
    // For power of two sizes, use radix-2 FFT with coset adjustment
    if (size > 0 and (size & (size - 1)) == 0) {
        // Multiply coefficients by powers of coset_offset
        var adjusted = try allocator.alloc(Field, size);
        defer allocator.free(adjusted);

        var offset_power = Field.ONE;
        for (0..size) |i| {
            if (i < poly.coeffs.len) {
                adjusted[i] = poly.coeffs[i].mul(offset_power);
            } else {
                adjusted[i] = Field.ZERO;
            }
            offset_power = offset_power.mul(coset_offset);
        }

        return fftRadix2(allocator, adjusted, generator);
    }

    // Fall back to general FFT
    return fft(allocator, poly.coeffs, generator, coset_offset, size);
}

// Tests
test "fft basic" {
    const allocator = std.testing.allocator;

    // Test with simple polynomial p(x) = 1 + x
    var coeffs = [_]Field{ Field.init(1), Field.init(1) };

    const root = Field.getRootOfUnity(1).?; // 2nd root of unity
    var result = try fftRadix2(allocator, &coeffs, root);
    defer allocator.free(result);

    // p(1) = 2, p(-1) = 0 (where -1 is represented as root)
    try std.testing.expectEqual(Field.init(2), result[0]);
    // result[1] = p(root) where root^2 = 1 and root != 1
}

test "fft roundtrip" {
    const allocator = std.testing.allocator;

    var coeffs = [_]Field{ Field.init(1), Field.init(2), Field.init(3), Field.init(4) };
    const root = Field.getRootOfUnity(2).?; // 4th root of unity

    const evals = try fftRadix2(allocator, &coeffs, root);
    defer allocator.free(evals);

    var recovered = try ifftRadix2(allocator, evals, root);
    defer allocator.free(recovered);

    for (0..4) |i| {
        try std.testing.expectEqual(coeffs[i], recovered[i]);
    }
}

test "bit reverse" {
    try std.testing.expectEqual(@as(usize, 0), bitReverse(0, 3));
    try std.testing.expectEqual(@as(usize, 4), bitReverse(1, 3));
    try std.testing.expectEqual(@as(usize, 2), bitReverse(2, 3));
    try std.testing.expectEqual(@as(usize, 6), bitReverse(3, 3));
}
