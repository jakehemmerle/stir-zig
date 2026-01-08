const std = @import("std");
const Field = @import("field.zig").Field;
const Allocator = std.mem.Allocator;

/// Evaluation domain for polynomial operations
/// Represents a multiplicative coset: offset * <generator>
pub const Domain = struct {
    /// Size of the domain (power of 2)
    size: usize,
    /// Primitive generator of the domain (root of unity)
    generator: Field,
    /// Inverse of the generator
    generator_inv: Field,
    /// Coset offset
    offset: Field,
    /// Inverse of offset
    offset_inv: Field,
    /// The original root of unity for the full domain
    root_of_unity: Field,
    /// Inverse of root of unity
    root_of_unity_inv: Field,

    /// Create a new domain for polynomials of the given degree with the specified rate
    /// size = degree * 2^log_rho_inv
    pub fn init(degree: usize, log_rho_inv: usize) !Domain {
        const size = degree << @intCast(log_rho_inv);

        // Size must be power of 2
        if (size == 0 or (size & (size - 1)) != 0) {
            return error.InvalidDomainSize;
        }

        const log_size = @ctz(size);
        const generator = Field.getRootOfUnity(@intCast(log_size)) orelse return error.NoRootOfUnity;
        const generator_inv = generator.inv() orelse return error.InversionFailed;

        return Domain{
            .size = size,
            .generator = generator,
            .generator_inv = generator_inv,
            .offset = Field.ONE,
            .offset_inv = Field.ONE,
            .root_of_unity = generator,
            .root_of_unity_inv = generator_inv,
        };
    }

    /// Get the i-th element of the domain
    pub fn element(self: Domain, i: usize) Field {
        return self.offset.mul(self.generator.pow(@intCast(i % self.size)));
    }

    /// Scale the domain by taking every `power`-th element
    /// New domain is <generator^power> with size/power elements
    pub fn scale(self: Domain, power: usize) Domain {
        std.debug.assert(self.size % power == 0);
        const new_size = self.size / power;

        const new_generator = self.generator.pow(@intCast(power));
        const new_generator_inv = self.generator_inv.pow(@intCast(power));
        const new_offset = self.offset.pow(@intCast(power));
        const new_offset_inv = self.offset_inv.pow(@intCast(power));

        return Domain{
            .size = new_size,
            .generator = new_generator,
            .generator_inv = new_generator_inv,
            .offset = new_offset,
            .offset_inv = new_offset_inv,
            .root_of_unity = self.root_of_unity,
            .root_of_unity_inv = self.root_of_unity_inv,
        };
    }

    /// Scale with offset: creates domain w * offset^power * <generator^power>
    /// This ensures the new domain doesn't overlap with the original
    pub fn scaleOffset(self: Domain, power: usize) Domain {
        std.debug.assert(self.size % power == 0);
        const new_size = self.size / power;

        const new_generator = self.generator.pow(@intCast(power));
        const new_generator_inv = self.generator_inv.pow(@intCast(power));

        // New offset = old_offset^power * root_of_unity (to shift the coset)
        const new_offset = self.offset.pow(@intCast(power)).mul(self.root_of_unity);
        const new_offset_inv = self.offset_inv.pow(@intCast(power)).mul(self.root_of_unity_inv);

        return Domain{
            .size = new_size,
            .generator = new_generator,
            .generator_inv = new_generator_inv,
            .offset = new_offset,
            .offset_inv = new_offset_inv,
            .root_of_unity = self.root_of_unity,
            .root_of_unity_inv = self.root_of_unity_inv,
        };
    }

    /// Get all elements of the domain
    pub fn elements(self: Domain, allocator: Allocator) ![]Field {
        var result = try allocator.alloc(Field, self.size);
        var current = self.offset;
        for (0..self.size) |i| {
            result[i] = current;
            current = current.mul(self.generator);
        }
        return result;
    }
};

// Tests
test "domain creation" {
    const domain = try Domain.init(16, 2);

    try std.testing.expectEqual(@as(usize, 64), domain.size);
    try std.testing.expectEqual(Field.ONE, domain.offset);
}

test "domain elements" {
    const allocator = std.testing.allocator;
    const domain = try Domain.init(4, 1);

    var elems = try domain.elements(allocator);
    defer allocator.free(elems);

    try std.testing.expectEqual(@as(usize, 8), elems.len);

    // First element should be offset (1)
    try std.testing.expectEqual(Field.ONE, elems[0]);

    // All elements raised to domain.size should equal offset^size
    for (elems) |e| {
        const powered = e.pow(@intCast(domain.size));
        try std.testing.expectEqual(Field.ONE, powered);
    }
}

test "domain scale" {
    const domain = try Domain.init(16, 2);
    const scaled = domain.scale(4);

    try std.testing.expectEqual(@as(usize, 16), scaled.size);

    // generator^4 should have order size/4
    const order_check = scaled.generator.pow(@intCast(scaled.size));
    try std.testing.expectEqual(Field.ONE, order_check);
}

test "domain non overlapping" {
    const allocator = std.testing.allocator;
    const folding_factor: usize = 4;

    const l0 = try Domain.init(64, 2);
    const l0_k = l0.scale(folding_factor);
    const l1 = l0.scaleOffset(2);

    const l0_k_elems = try l0_k.elements(allocator);
    defer allocator.free(l0_k_elems);
    const l1_elems = try l1.elements(allocator);
    defer allocator.free(l1_elems);

    // Check no overlap
    for (l0_k_elems) |e1| {
        for (l1_elems) |e2| {
            try std.testing.expect(!e1.eq(e2));
        }
    }
}
