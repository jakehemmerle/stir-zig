const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const fft_mod = @import("fft.zig");
const Point = @import("types.zig").Point;
const Allocator = std.mem.Allocator;

/// Computes the vanishing polynomial that is zero at all given points
/// Z(x) = prod_{a in points} (x - a)
pub fn vanishingPoly(allocator: Allocator, points: []const Field) !DensePolynomial {
    // Start with the constant polynomial 1
    var result_coeffs = try allocator.alloc(Field, 1);
    result_coeffs[0] = Field.ONE;
    var result = DensePolynomial{ .coeffs = result_coeffs, .allocator = allocator };

    for (points) |a| {
        // Multiply by (x - a)
        // (x - a) has coefficients [-a, 1]
        var linear_coeffs = [_]Field{ a.neg(), Field.ONE };
        var linear = try DensePolynomial.init(allocator, &linear_coeffs);
        defer linear.deinit();

        const new_result = try result.mul(linear);
        result.deinit();
        result = new_result;
    }

    return result;
}

/// Naive Lagrange interpolation
/// Given points [(x_0, y_0), ..., (x_n, y_n)], computes the polynomial p(x)
/// such that p(x_i) = y_i
pub fn naiveInterpolation(allocator: Allocator, points: []const Point) !DensePolynomial {
    if (points.len == 0) {
        var zero = try allocator.alloc(Field, 1);
        zero[0] = Field.ZERO;
        return DensePolynomial{ .coeffs = zero, .allocator = allocator };
    }

    // Compute vanishing polynomial
    var xs = try allocator.alloc(Field, points.len);
    defer allocator.free(xs);
    for (points, 0..) |p, i| {
        xs[i] = p.x;
    }
    var vanishing = try vanishingPoly(allocator, xs);
    defer vanishing.deinit();

    // Start with zero polynomial
    var result_coeffs = try allocator.alloc(Field, 1);
    result_coeffs[0] = Field.ZERO;
    var result = DensePolynomial{ .coeffs = result_coeffs, .allocator = allocator };

    for (points) |point| {
        const a = point.x;
        const y = point.y;

        // Compute vanishing / (x - a)
        var vanishing_adjusted = try vanishing.divideByLinear(a);
        defer vanishing_adjusted.deinit();

        // Evaluate at a to get the denominator
        const denom = vanishing_adjusted.evaluate(a);
        const scale_factor = y.div(denom) orelse continue;

        // Scale and add
        var scaled = try vanishing_adjusted.scale(scale_factor);
        defer scaled.deinit();

        const new_result = try result.add(scaled);
        result.deinit();
        result = new_result;
    }

    return result;
}

/// Evaluate interpolation at a single point without computing the full polynomial
/// Uses Lagrange's formula directly
pub fn evaluateInterpolation(points: []const Point, point: Field) Field {
    // Check if point is one of the interpolation points
    for (points) |p| {
        if (p.x.eq(point)) {
            return p.y;
        }
    }

    // Compute denominators using Lagrange formula
    // L_i(x) = prod_{j != i} (x - x_j) / (x_i - x_j)
    // p(point) = sum_i y_i * L_i(point)

    var result = Field.ZERO;
    var numerator_product = Field.ONE;

    // First compute product of (point - x_j) for all j
    for (points) |p| {
        numerator_product = numerator_product.mul(point.sub(p.x));
    }

    for (points, 0..) |pi, i| {
        // Compute denominator: prod_{j != i} (x_i - x_j)
        var denom = Field.ONE;
        for (points, 0..) |pj, j| {
            if (i != j) {
                denom = denom.mul(pi.x.sub(pj.x));
            }
        }

        // Compute (point - x_i) inverse for numerator adjustment
        const point_minus_xi = point.sub(pi.x);
        const point_minus_xi_inv = point_minus_xi.inv() orelse continue;

        // L_i(point) = numerator_product / (point - x_i) / denom
        const li = numerator_product.mul(point_minus_xi_inv).mul(denom.inv() orelse continue);

        result = result.add(pi.y.mul(li));
    }

    return result;
}

/// FFT-based interpolation for evaluations on a coset
pub fn fftInterpolate(allocator: Allocator, generator: Field, coset_offset: Field, evals: []const Field) !DensePolynomial {
    const n = evals.len;
    std.debug.assert(n > 0 and (n & (n - 1)) == 0); // Must be power of 2

    // Inverse FFT
    const generator_inv = generator.inv() orelse return error.InversionFailed;
    var coeffs = try fft_mod.fftRadix2(allocator, evals, generator_inv);
    defer allocator.free(coeffs);

    // Scale by 1/n and adjust for coset
    const n_inv = Field.init(@intCast(n)).inv() orelse return error.InversionFailed;
    const coset_inv = coset_offset.inv() orelse return error.InversionFailed;

    var result = try allocator.alloc(Field, n);
    var coset_power = Field.ONE;
    for (0..n) |i| {
        result[i] = coeffs[i].mul(n_inv).mul(coset_power);
        coset_power = coset_power.mul(coset_inv);
    }

    return DensePolynomial{ .coeffs = result, .allocator = allocator };
}

// Tests
test "vanishing polynomial" {
    const allocator = std.testing.allocator;

    var points = [_]Field{ Field.init(5), Field.init(10), Field.init(9), Field.init(7) };
    var vanishing = try vanishingPoly(allocator, &points);
    defer vanishing.deinit();

    // Vanishing polynomial should be zero at all points
    for (points) |p| {
        try std.testing.expectEqual(Field.ZERO, vanishing.evaluate(p));
    }
}

test "naive interpolation" {
    const allocator = std.testing.allocator;

    var points = [_]Point{
        .{ .x = Field.init(5), .y = Field.init(10) },
        .{ .x = Field.init(9), .y = Field.init(7) },
    };

    var poly = try naiveInterpolation(allocator, &points);
    defer poly.deinit();

    // Check that polynomial passes through all points
    for (points) |p| {
        try std.testing.expectEqual(p.y, poly.evaluate(p.x));
    }
}

test "evaluate interpolation" {
    var points = [_]Point{
        .{ .x = Field.init(5), .y = Field.init(10) },
        .{ .x = Field.init(9), .y = Field.init(7) },
    };

    // At interpolation points
    try std.testing.expectEqual(Field.init(10), evaluateInterpolation(&points, Field.init(5)));
    try std.testing.expectEqual(Field.init(7), evaluateInterpolation(&points, Field.init(9)));
}

test "fft interpolation roundtrip" {
    const allocator = std.testing.allocator;

    // Create a polynomial and evaluate it over a coset
    var coeffs = [_]Field{ Field.init(1), Field.init(2), Field.init(3), Field.init(4) };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    const root = Field.getRootOfUnity(2).?; // 4th root of unity
    const coset_offset = Field.init(3);

    // Evaluate over coset
    const evals = try fft_mod.evaluateOverCoset(allocator, poly, root, coset_offset, 4);
    defer allocator.free(evals);

    // Interpolate back
    var recovered = try fftInterpolate(allocator, root, coset_offset, evals);
    defer recovered.deinit();

    // Should get back original coefficients
    for (0..4) |i| {
        try std.testing.expect(coeffs[i].eq(recovered.coeffs[i]));
    }
}
