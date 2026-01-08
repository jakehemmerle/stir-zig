const std = @import("std");
const Field = @import("field.zig").Field;
const Allocator = std.mem.Allocator;

/// Check if a number is a power of 2
pub fn isPowerOfTwo(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

/// Transpose a 2D array
pub fn transpose(allocator: Allocator, matrix: []const []const Field) ![][]Field {
    if (matrix.len == 0) return &[_][]Field{};

    const rows = matrix.len;
    const cols = matrix[0].len;

    var result = try allocator.alloc([]Field, cols);
    errdefer allocator.free(result);

    for (0..cols) |c| {
        result[c] = try allocator.alloc(Field, rows);
        for (0..rows) |r| {
            result[c][r] = matrix[r][c];
        }
    }

    return result;
}

/// Free a transposed matrix
pub fn freeTransposed(allocator: Allocator, matrix: [][]Field) void {
    for (matrix) |row| {
        allocator.free(row);
    }
    allocator.free(matrix);
}

/// Stack evaluations for folding
/// Takes [f(w^0), f(w^1), ..., f(w^{n-1})] and produces
/// [[f(w^0), f(w^k), f(w^{2k}), ...], [f(w^1), f(w^{k+1}), ...], ...]
/// where k = n / folding_factor
pub fn stackEvaluations(allocator: Allocator, evals: []const Field, folding_factor: usize) ![][]Field {
    std.debug.assert(evals.len % folding_factor == 0);
    const size_of_new_domain = evals.len / folding_factor;

    var stacked = try allocator.alloc([]Field, size_of_new_domain);
    errdefer allocator.free(stacked);

    for (0..size_of_new_domain) |i| {
        stacked[i] = try allocator.alloc(Field, folding_factor);
        for (0..folding_factor) |j| {
            stacked[i][j] = evals[i + j * size_of_new_domain];
        }
    }

    return stacked;
}

/// Free stacked evaluations
pub fn freeStacked(allocator: Allocator, stacked: [][]Field) void {
    for (stacked) |row| {
        allocator.free(row);
    }
    allocator.free(stacked);
}

/// Remove duplicates and sort (like BTreeSet in Rust)
pub fn dedup(allocator: Allocator, values: []const usize) ![]usize {
    if (values.len == 0) return &[_]usize{};

    // Copy and sort
    var sorted = try allocator.dupe(usize, values);
    defer allocator.free(sorted);
    std.mem.sort(usize, sorted, {}, std.sort.asc(usize));

    // Count unique
    var unique_count: usize = 1;
    for (1..sorted.len) |i| {
        if (sorted[i] != sorted[i - 1]) {
            unique_count += 1;
        }
    }

    // Build result
    var result = try allocator.alloc(usize, unique_count);
    result[0] = sorted[0];
    var idx: usize = 1;
    for (1..sorted.len) |i| {
        if (sorted[i] != sorted[i - 1]) {
            result[idx] = sorted[i];
            idx += 1;
        }
    }

    return result;
}

// Tests
test "is power of two" {
    try std.testing.expect(isPowerOfTwo(1));
    try std.testing.expect(isPowerOfTwo(2));
    try std.testing.expect(isPowerOfTwo(4));
    try std.testing.expect(isPowerOfTwo(1024));

    try std.testing.expect(!isPowerOfTwo(0));
    try std.testing.expect(!isPowerOfTwo(3));
    try std.testing.expect(!isPowerOfTwo(5));
}

test "stack evaluations" {
    const allocator = std.testing.allocator;

    var evals = [_]Field{
        Field.init(0), Field.init(1), Field.init(2), Field.init(3),
        Field.init(4), Field.init(5), Field.init(6), Field.init(7),
    };

    var stacked = try stackEvaluations(allocator, &evals, 4);
    defer freeStacked(allocator, stacked);

    try std.testing.expectEqual(@as(usize, 2), stacked.len);
    try std.testing.expectEqual(@as(usize, 4), stacked[0].len);

    // stacked[0] = [evals[0], evals[2], evals[4], evals[6]]
    try std.testing.expectEqual(Field.init(0), stacked[0][0]);
    try std.testing.expectEqual(Field.init(2), stacked[0][1]);
    try std.testing.expectEqual(Field.init(4), stacked[0][2]);
    try std.testing.expectEqual(Field.init(6), stacked[0][3]);
}

test "dedup" {
    const allocator = std.testing.allocator;

    var values = [_]usize{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
    const result = try dedup(allocator, &values);
    defer allocator.free(result);

    // Should be sorted and unique: [1, 2, 3, 4, 5, 6, 9]
    const expected = [_]usize{ 1, 2, 3, 4, 5, 6, 9 };
    try std.testing.expectEqualSlices(usize, &expected, result);
}
