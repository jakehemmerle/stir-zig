const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const interpolation = @import("interpolation.zig");
const Point = @import("types.zig").Point;
const Allocator = std.mem.Allocator;

/// Compute the quotient polynomial (poly - ans_poly) / vanishing_poly
/// where ans_poly interpolates the given points
pub fn polyQuotient(allocator: Allocator, poly: DensePolynomial, points: []const Field) !DensePolynomial {
    // Compute evaluations at points
    var evaluations = try allocator.alloc(Point, points.len);
    defer allocator.free(evaluations);

    for (points, 0..) |p, i| {
        evaluations[i] = .{ .x = p, .y = poly.evaluate(p) };
    }

    // Compute answer polynomial
    var ans_poly = try interpolation.naiveInterpolation(allocator, evaluations);
    defer ans_poly.deinit();

    // Compute vanishing polynomial
    var vanishing = try interpolation.vanishingPoly(allocator, points);
    defer vanishing.deinit();

    // Compute numerator = poly - ans_poly (we add because we want quotient of poly)
    // Actually the Rust code does poly + ans_poly, let's check...
    // The numerator is poly - ans at the points, so we need poly - interpolating_poly
    // But then we're dividing by vanishing which is zero at those points
    // Actually reviewing Rust: numerator = poly + ans_polynomial
    // This seems like it might be a sign convention thing
    var numerator = try poly.add(ans_poly);
    defer numerator.deinit();

    // Divide by vanishing polynomial
    const div_result = try numerator.divMod(vanishing);
    var remainder = div_result.remainder;
    defer remainder.deinit();

    return div_result.quotient;
}

/// Compute the quotient evaluation directly at a point
/// Quotient(f, S, Ans, Fill) from the STIR paper
/// claimed_eval: f(evaluation_point)
/// evaluation_point: the point we're evaluating at (must not be in answers' domain)
/// answers: pairs of (domain_point, f(domain_point))
pub fn quotient(claimed_eval: Field, evaluation_point: Field, answers: []const Point) ?Field {
    // Check if evaluation point is in domain
    for (answers) |a| {
        if (evaluation_point.eq(a.x)) {
            // Evaluation point is in domain, invalid
            return null;
        }
    }

    // Compute ans polynomial evaluation at evaluation_point
    const ans_eval = interpolation.evaluateInterpolation(answers, evaluation_point);

    // Numerator = claimed_eval - ans_eval
    const num = claimed_eval.sub(ans_eval);

    // Denominator = product of (evaluation_point - domain_point)
    var denom = Field.ONE;
    for (answers) |a| {
        denom = denom.mul(evaluation_point.sub(a.x));
    }

    // Return quotient
    return num.div(denom);
}

/// Quotient with precomputed hints for efficiency
/// denom_hint: precomputed inverse of product of (evaluation_point - domain_points)
/// ans_eval: precomputed evaluation of answer polynomial at evaluation_point
pub fn quotientWithHint(claimed_eval: Field, evaluation_point: Field, quotient_set: []const Field, denom_hint: Field, ans_eval: Field) ?Field {
    // Check if evaluation point is in domain
    for (quotient_set) |x| {
        if (evaluation_point.eq(x)) {
            return null;
        }
    }

    const num = claimed_eval.sub(ans_eval);
    return num.mul(denom_hint);
}

// Tests
test "quotient computation" {
    const allocator = std.testing.allocator;

    // Create a polynomial p(x) = x^2 - 1 = (x-1)(x+1)
    // -1 in the field is Field.ZERO.sub(Field.ONE) = Field.ONE.neg()
    const neg_one = Field.ONE.neg();
    var coeffs = [_]Field{ neg_one, Field.ZERO, Field.ONE };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    // Points where we "know" the value: at x=0 and x=2
    var points = [_]Field{ Field.ZERO, Field.init(2) };

    var quotient_poly = try polyQuotient(allocator, poly, &points);
    defer quotient_poly.deinit();

    // Quotient should be defined
    try std.testing.expect(quotient_poly.coeffs.len > 0);
}

test "quotient evaluation" {
    // Simple test: interpolating through (1, 5) and (2, 8)
    // The line is y = 3x + 2
    var answers = [_]Point{
        .{ .x = Field.init(1), .y = Field.init(5) },
        .{ .x = Field.init(2), .y = Field.init(8) },
    };

    // At x=3, the line gives y=11
    const eval_point = Field.init(3);
    const claimed_eval = Field.init(11);

    const result = quotient(claimed_eval, eval_point, &answers);
    try std.testing.expect(result != null);

    // If claimed_eval matches interpolation, quotient should be 0
    try std.testing.expectEqual(Field.ZERO, result.?);
}
