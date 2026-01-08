const std = @import("std");
const Field = @import("field.zig").Field;
const Allocator = std.mem.Allocator;

/// Dense univariate polynomial represented by its coefficients
/// coeffs[i] is the coefficient of x^i
pub const DensePolynomial = struct {
    coeffs: []Field,
    allocator: Allocator,

    pub fn init(allocator: Allocator, coeffs: []const Field) !DensePolynomial {
        const owned = try allocator.dupe(Field, coeffs);
        return .{
            .coeffs = owned,
            .allocator = allocator,
        };
    }

    pub fn initZero(allocator: Allocator, deg: usize) !DensePolynomial {
        const coeffs = try allocator.alloc(Field, deg + 1);
        @memset(coeffs, Field.ZERO);
        return .{
            .coeffs = coeffs,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DensePolynomial) void {
        self.allocator.free(self.coeffs);
    }

    pub fn clone(self: DensePolynomial) !DensePolynomial {
        return init(self.allocator, self.coeffs);
    }

    pub fn degree(self: DensePolynomial) usize {
        if (self.coeffs.len == 0) return 0;

        var d = self.coeffs.len - 1;
        while (d > 0 and self.coeffs[d].eq(Field.ZERO)) {
            d -= 1;
        }
        return d;
    }

    pub fn isZero(self: DensePolynomial) bool {
        for (self.coeffs) |c| {
            if (!c.eq(Field.ZERO)) return false;
        }
        return true;
    }

    pub fn evaluate(self: DensePolynomial, point: Field) Field {
        // Horner's method
        if (self.coeffs.len == 0) return Field.ZERO;

        var result = self.coeffs[self.coeffs.len - 1];
        var i: usize = self.coeffs.len - 1;
        while (i > 0) {
            i -= 1;
            result = result.mul(point).add(self.coeffs[i]);
        }
        return result;
    }

    pub fn add(self: DensePolynomial, other: DensePolynomial) !DensePolynomial {
        const max_len = @max(self.coeffs.len, other.coeffs.len);
        var result = try self.allocator.alloc(Field, max_len);

        for (0..max_len) |i| {
            const a = if (i < self.coeffs.len) self.coeffs[i] else Field.ZERO;
            const b = if (i < other.coeffs.len) other.coeffs[i] else Field.ZERO;
            result[i] = a.add(b);
        }

        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    pub fn sub(self: DensePolynomial, other: DensePolynomial) !DensePolynomial {
        const max_len = @max(self.coeffs.len, other.coeffs.len);
        var result = try self.allocator.alloc(Field, max_len);

        for (0..max_len) |i| {
            const a = if (i < self.coeffs.len) self.coeffs[i] else Field.ZERO;
            const b = if (i < other.coeffs.len) other.coeffs[i] else Field.ZERO;
            result[i] = a.sub(b);
        }

        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    /// Multiply two polynomials using naive O(n^2) algorithm
    pub fn mul(self: DensePolynomial, other: DensePolynomial) !DensePolynomial {
        if (self.coeffs.len == 0 or other.coeffs.len == 0) {
            return initZero(self.allocator, 0);
        }

        const result_len = self.coeffs.len + other.coeffs.len - 1;
        var result = try self.allocator.alloc(Field, result_len);
        @memset(result, Field.ZERO);

        for (self.coeffs, 0..) |a, i| {
            for (other.coeffs, 0..) |b, j| {
                result[i + j] = result[i + j].add(a.mul(b));
            }
        }

        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    /// Scale polynomial by a constant
    pub fn scale(self: DensePolynomial, scalar: Field) !DensePolynomial {
        var result = try self.allocator.alloc(Field, self.coeffs.len);
        for (self.coeffs, 0..) |c, i| {
            result[i] = c.mul(scalar);
        }
        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    /// Scale and shift: scalar * x^shift * self
    pub fn scaleAndShift(self: DensePolynomial, scalar: Field, shift: usize) !DensePolynomial {
        var result = try self.allocator.alloc(Field, self.coeffs.len + shift);
        @memset(result[0..shift], Field.ZERO);
        for (self.coeffs, 0..) |c, i| {
            result[i + shift] = c.mul(scalar);
        }
        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    /// Divide polynomial by (x - root), assuming it divides evenly
    pub fn divideByLinear(self: DensePolynomial, root: Field) !DensePolynomial {
        if (self.coeffs.len <= 1) {
            return initZero(self.allocator, 0);
        }

        var result = try self.allocator.alloc(Field, self.coeffs.len - 1);
        result[self.coeffs.len - 2] = self.coeffs[self.coeffs.len - 1];

        var i: usize = self.coeffs.len - 2;
        while (i > 0) {
            i -= 1;
            result[i] = self.coeffs[i + 1].add(root.mul(result[i + 1]));
        }

        return .{
            .coeffs = result,
            .allocator = self.allocator,
        };
    }

    /// Polynomial division: returns (quotient, remainder)
    pub fn divMod(self: DensePolynomial, divisor: DensePolynomial) !struct { quotient: DensePolynomial, remainder: DensePolynomial } {
        if (divisor.isZero()) {
            return error.DivisionByZero;
        }

        const self_deg = self.degree();
        const divisor_deg = divisor.degree();

        if (self_deg < divisor_deg) {
            return .{
                .quotient = try initZero(self.allocator, 0),
                .remainder = try self.clone(),
            };
        }

        var remainder = try self.clone();
        defer remainder.deinit();

        const quotient_len = self_deg - divisor_deg + 1;
        var quotient = try self.allocator.alloc(Field, quotient_len);
        @memset(quotient, Field.ZERO);

        const leading_inv = divisor.coeffs[divisor_deg].inv() orelse return error.DivisionByZero;

        var i: usize = self_deg;
        while (i >= divisor_deg) {
            const coeff = remainder.coeffs[i].mul(leading_inv);
            quotient[i - divisor_deg] = coeff;

            for (0..divisor_deg + 1) |j| {
                remainder.coeffs[i - divisor_deg + j] = remainder.coeffs[i - divisor_deg + j].sub(coeff.mul(divisor.coeffs[j]));
            }

            if (i == 0) break;
            i -= 1;
        }

        const rem_result = try self.allocator.dupe(Field, remainder.coeffs);

        return .{
            .quotient = .{ .coeffs = quotient, .allocator = self.allocator },
            .remainder = .{ .coeffs = rem_result, .allocator = self.allocator },
        };
    }
};

// Tests
test "polynomial evaluation" {
    const allocator = std.testing.allocator;

    // p(x) = 1 + 2x + 3x^2
    var coeffs = [_]Field{ Field.init(1), Field.init(2), Field.init(3) };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    // p(2) = 1 + 4 + 12 = 17
    const result = poly.evaluate(Field.init(2));
    try std.testing.expectEqual(Field.init(17), result);
}

test "polynomial addition" {
    const allocator = std.testing.allocator;

    var coeffs1 = [_]Field{ Field.init(1), Field.init(2) };
    var coeffs2 = [_]Field{ Field.init(3), Field.init(4), Field.init(5) };

    var p1 = try DensePolynomial.init(allocator, &coeffs1);
    defer p1.deinit();
    var p2 = try DensePolynomial.init(allocator, &coeffs2);
    defer p2.deinit();

    var sum = try p1.add(p2);
    defer sum.deinit();

    try std.testing.expectEqual(Field.init(4), sum.coeffs[0]);
    try std.testing.expectEqual(Field.init(6), sum.coeffs[1]);
    try std.testing.expectEqual(Field.init(5), sum.coeffs[2]);
}

test "polynomial multiplication" {
    const allocator = std.testing.allocator;

    // (1 + x) * (1 + x) = 1 + 2x + x^2
    var coeffs = [_]Field{ Field.init(1), Field.init(1) };
    var p = try DensePolynomial.init(allocator, &coeffs);
    defer p.deinit();

    var product = try p.mul(p);
    defer product.deinit();

    try std.testing.expectEqual(Field.init(1), product.coeffs[0]);
    try std.testing.expectEqual(Field.init(2), product.coeffs[1]);
    try std.testing.expectEqual(Field.init(1), product.coeffs[2]);
}

test "polynomial degree" {
    const allocator = std.testing.allocator;

    var coeffs = [_]Field{ Field.init(1), Field.init(2), Field.ZERO };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    try std.testing.expectEqual(@as(usize, 1), poly.degree());
}
