const std = @import("std");
const Field = @import("field.zig").Field;
const DensePolynomial = @import("polynomial.zig").DensePolynomial;
const interpolation = @import("interpolation.zig");
const Allocator = std.mem.Allocator;

/// Bivariate polynomial represented as a matrix of coefficients
/// p(x, y) = sum_{i,j} coeffs[i][j] * x^i * y^j
pub const BivariatePolynomial = struct {
    coeffs: [][]Field,
    allocator: Allocator,

    pub fn deinit(self: *BivariatePolynomial) void {
        for (self.coeffs) |row| {
            self.allocator.free(row);
        }
        self.allocator.free(self.coeffs);
    }

    pub fn rows(self: BivariatePolynomial) usize {
        return self.coeffs.len;
    }

    pub fn cols(self: BivariatePolynomial) usize {
        if (self.coeffs.len == 0) return 0;
        return self.coeffs[0].len;
    }

    pub fn evaluate(self: BivariatePolynomial, x: Field, y: Field) Field {
        var result = Field.ZERO;
        var x_pow = Field.ONE;
        for (self.coeffs) |row| {
            var y_pow = Field.ONE;
            for (row) |c| {
                result = result.add(c.mul(x_pow).mul(y_pow));
                y_pow = y_pow.mul(y);
            }
            x_pow = x_pow.mul(x);
        }
        return result;
    }

    /// Fold by column: compute sum_j alpha^j * col_j as a polynomial
    pub fn foldByCol(self: BivariatePolynomial, alpha: Field) !DensePolynomial {
        const num_cols = self.cols();
        const num_rows = self.rows();

        if (num_cols == 0 or num_rows == 0) {
            var zero = try self.allocator.alloc(Field, 1);
            zero[0] = Field.ZERO;
            return DensePolynomial{ .coeffs = zero, .allocator = self.allocator };
        }

        var result = try self.allocator.alloc(Field, num_rows);
        @memset(result, Field.ZERO);

        var alpha_pow = Field.ONE;
        for (0..num_cols) |j| {
            for (0..num_rows) |i| {
                result[i] = result[i].add(self.coeffs[i][j].mul(alpha_pow));
            }
            alpha_pow = alpha_pow.mul(alpha);
        }

        return DensePolynomial{ .coeffs = result, .allocator = self.allocator };
    }
};

/// Convert a polynomial to its coefficient matrix representation (BS08)
/// Interprets polynomial coefficients as a rows x cols matrix
pub fn toCoefficientMatrix(allocator: Allocator, f: DensePolynomial, num_rows: usize, num_cols: usize) !BivariatePolynomial {
    const max_coeffs = num_rows * num_cols;
    if (f.coeffs.len > max_coeffs) {
        return error.PolynomialTooLarge;
    }

    var matrix = try allocator.alloc([]Field, num_rows);
    errdefer allocator.free(matrix);

    for (0..num_rows) |i| {
        matrix[i] = try allocator.alloc(Field, num_cols);
        for (0..num_cols) |j| {
            const idx = i * num_cols + j;
            if (idx < f.coeffs.len) {
                matrix[i][j] = f.coeffs[idx];
            } else {
                matrix[i][j] = Field.ZERO;
            }
        }
    }

    return BivariatePolynomial{
        .coeffs = matrix,
        .allocator = allocator,
    };
}

/// Fold a polynomial using the BS08 technique
/// Takes f(x) and returns g(y) = sum_j alpha^j * q_j(y) where f(x) = sum_j q_j(x^cols) * x^j
pub fn polyFold(allocator: Allocator, f: DensePolynomial, folding_factor: usize, folding_randomness: Field) !DensePolynomial {
    const degree_plus_one = f.coeffs.len;
    const num_rows = (degree_plus_one + folding_factor - 1) / folding_factor;

    var bivar = try toCoefficientMatrix(allocator, f, num_rows, folding_factor);
    defer bivar.deinit();

    return bivar.foldByCol(folding_randomness);
}

/// Fold evaluations at points forming a coset
/// f_answers is a vector of (point, f(point)) pairs where points form a coset under x^k
pub fn fold(allocator: Allocator, f_answers: []const struct { x: Field, y: Field }, folding_factor: usize, folding_randomness: Field) !Field {
    _ = allocator;
    std.debug.assert(f_answers.len == folding_factor);
    return interpolation.evaluateInterpolation(f_answers, folding_randomness);
}

// Tests
test "coefficient matrix" {
    const allocator = std.testing.allocator;

    // Polynomial with coefficients [0, 1, 2, 3, 4, 5]
    var coeffs = [_]Field{
        Field.init(0), Field.init(1), Field.init(2),
        Field.init(3), Field.init(4), Field.init(5),
    };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    var matrix = try toCoefficientMatrix(allocator, poly, 3, 2);
    defer matrix.deinit();

    // Matrix should be:
    // [0, 1]
    // [2, 3]
    // [4, 5]
    for (0..3) |r| {
        for (0..2) |c| {
            try std.testing.expectEqual(Field.init(@intCast(2 * r + c)), matrix.coeffs[r][c]);
        }
    }
}

test "bivariate evaluation" {
    const allocator = std.testing.allocator;

    var coeffs = [_]Field{
        Field.init(0), Field.init(1), Field.init(2),
        Field.init(3), Field.init(4), Field.init(5),
    };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    var matrix = try toCoefficientMatrix(allocator, poly, 3, 2);
    defer matrix.deinit();

    // p(x) evaluated at point should equal matrix.evaluate(point^cols, point)
    const point = Field.init(7);
    const poly_eval = poly.evaluate(point);
    const cols: u64 = 2;
    const matrix_eval = matrix.evaluate(point.pow(cols), point);

    try std.testing.expectEqual(poly_eval, matrix_eval);
}

test "poly fold" {
    const allocator = std.testing.allocator;

    var coeffs = [_]Field{
        Field.init(1), Field.init(2), Field.init(3), Field.init(4),
        Field.init(5), Field.init(6), Field.init(7), Field.init(8),
    };
    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    const folding_factor: usize = 2;
    const folding_randomness = Field.init(5);

    var folded = try polyFold(allocator, poly, folding_factor, folding_randomness);
    defer folded.deinit();

    // The folded polynomial should have degree (original_degree / folding_factor)
    try std.testing.expect(folded.coeffs.len <= poly.coeffs.len / folding_factor + 1);
}
