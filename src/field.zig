const std = @import("std");
const builtin = @import("builtin");

/// 192-bit field with Montgomery representation (matching Rust's ark-ff Field192)
/// Modulus: 4787605948707450321761805915146316350821882368518086721537
/// = 0xc340f039bc837d70_16ce1859e23e15db_0000000000000001
/// Values are stored in Montgomery form: a is represented as aR mod N where R = 2^192
pub const Field192Mont = struct {
    limbs: [3]u64, // Stored in Montgomery form

    // Modulus N
    pub const MODULUS: [3]u64 = .{
        0x0000000000000001, // limb 0 (least significant)
        0x16ce1859e23e15db, // limb 1
        0xc340f039bc837d70, // limb 2 (most significant)
    };

    // Montgomery constant N' = -N^(-1) mod 2^64
    // Since N[0] = 1, N^(-1) mod 2^64 = 1, so N' = -1 mod 2^64
    pub const N_PRIME: u64 = 0xffffffffffffffff;

    // R^2 mod N (for converting to Montgomery form)
    pub const R_SQUARED: [3]u64 = .{
        0xed4789c79fd854b7,
        0x13089c72bf1056c9,
        0x2d1759d8d513e359,
    };

    // Constants in Montgomery form (multiplied by R mod N)
    pub const ZERO: Field192Mont = .{ .limbs = .{ 0, 0, 0 } };
    pub const ONE: Field192Mont = .{ .limbs = .{
        0xffffffffffffffff,
        0xe931e7a61dc1ea24,
        0x3cbf0fc6437c828f,
    } }; // R mod N
    pub const GENERATOR: Field192Mont = .{ .limbs = .{
        0xfffffffffffffffd,
        0xbb95b6f25945be6e,
        0xb63d2f52ca7587af,
    } }; // 3R mod N

    pub const TWO_ADICITY: u32 = 64;

    // 2^64-th root of unity in Montgomery form
    pub const TWO_ADIC_ROOT_OF_UNITY: [3]u64 = .{
        0x778883e246bb988c,
        0xb94f96f791b0dd61,
        0xb58e3f8fe66d5f45,
    };

    /// Create a field element from a u64 value (converts to Montgomery form)
    pub fn init(value: u64) Field192Mont {
        // Convert to Montgomery form: value * R mod N = REDC(value * R^2)
        const standard: [3]u64 = .{ value, 0, 0 };
        return .{ .limbs = montMul(standard, R_SQUARED) };
    }

    /// Create from limbs (assumes NOT in Montgomery form, will convert)
    pub fn fromLimbs(limbs: [3]u64) Field192Mont {
        // Reduce first, then convert to Montgomery form
        var reduced = limbs;
        while (!lessThan(reduced, MODULUS)) {
            reduced = sub192(reduced, MODULUS);
        }
        return .{ .limbs = montMul(reduced, R_SQUARED) };
    }

    /// Create from raw Montgomery limbs (assumes already in Montgomery form)
    fn fromMontLimbs(limbs: [3]u64) Field192Mont {
        return .{ .limbs = limbs };
    }

    /// Convert from Montgomery form to standard form
    pub fn toStandard(self: Field192Mont) [3]u64 {
        // REDC(aR * 1) = aR * R^(-1) = a
        return montMul(self.limbs, .{ 1, 0, 0 });
    }

    fn lessThan(a: [3]u64, b: [3]u64) bool {
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    }

    fn sub192(a: [3]u64, b: [3]u64) [3]u64 {
        var result: [3]u64 = undefined;
        var borrow: u64 = 0;

        const r0 = @subWithOverflow(a[0], b[0]);
        result[0] = r0[0];
        borrow = r0[1];

        const r1a = @subWithOverflow(a[1], b[1]);
        const r1b = @subWithOverflow(r1a[0], borrow);
        result[1] = r1b[0];
        borrow = r1a[1] + r1b[1];

        result[2] = a[2] -% b[2] -% borrow;
        return result;
    }

    fn add192WithCarry(a: [3]u64, b: [3]u64) struct { result: [3]u64, carry: bool } {
        var result: [3]u64 = undefined;
        var carry: u64 = 0;

        const r0 = @addWithOverflow(a[0], b[0]);
        result[0] = r0[0];
        carry = r0[1];

        const r1a = @addWithOverflow(a[1], b[1]);
        const r1b = @addWithOverflow(r1a[0], carry);
        result[1] = r1b[0];
        carry = r1a[1] + r1b[1];

        const r2a = @addWithOverflow(a[2], b[2]);
        const r2b = @addWithOverflow(r2a[0], carry);
        result[2] = r2b[0];
        const final_carry = (r2a[1] + r2b[1]) > 0;

        return .{ .result = result, .carry = final_carry };
    }

    /// Montgomery multiplication: computes REDC(a * b) = a * b * R^(-1) mod N
    /// Using CIOS (Coarsely Integrated Operand Scanning) algorithm
    fn montMul(a: [3]u64, b: [3]u64) [3]u64 {
        var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

        // Process each limb of b
        inline for (0..3) |i| {
            // Multiplication phase: t += a * b[i]
            var carry: u64 = 0;
            inline for (0..3) |j| {
                const prod: u128 = @as(u128, a[j]) * @as(u128, b[i]) +
                    @as(u128, t[j]) + @as(u128, carry);
                t[j] = @truncate(prod);
                carry = @truncate(prod >> 64);
            }
            const sum1: u128 = @as(u128, t[3]) + @as(u128, carry);
            t[3] = @truncate(sum1);
            t[4] = @truncate(sum1 >> 64);

            // Reduction phase
            // m = t[0] * N' mod 2^64 = -t[0] (since N' = -1)
            const m: u64 = t[0] *% N_PRIME;

            // Add m * N and shift right by 64 bits
            // First term: t[0] + m * N[0] should be divisible by 2^64
            const prod0: u128 = @as(u128, m) * @as(u128, MODULUS[0]) + @as(u128, t[0]);
            carry = @truncate(prod0 >> 64);

            // Remaining terms
            inline for (1..3) |j| {
                const prodj: u128 = @as(u128, m) * @as(u128, MODULUS[j]) +
                    @as(u128, t[j]) + @as(u128, carry);
                t[j - 1] = @truncate(prodj);
                carry = @truncate(prodj >> 64);
            }

            const sum2: u128 = @as(u128, t[3]) + @as(u128, carry);
            t[2] = @truncate(sum2);
            carry = @truncate(sum2 >> 64);
            t[3] = t[4] +% carry;
            t[4] = 0;
        }

        // Final conditional subtraction
        var result: [3]u64 = .{ t[0], t[1], t[2] };
        if (t[3] > 0 or !lessThan(result, MODULUS)) {
            result = sub192(result, MODULUS);
        }
        return result;
    }

    pub fn add(self: Field192Mont, other: Field192Mont) Field192Mont {
        const added = add192WithCarry(self.limbs, other.limbs);
        var result = added.result;

        // If overflow or result >= modulus, subtract modulus
        if (added.carry or !lessThan(result, MODULUS)) {
            result = sub192(result, MODULUS);
        }
        return .{ .limbs = result };
    }

    pub fn sub(self: Field192Mont, other: Field192Mont) Field192Mont {
        if (lessThan(self.limbs, other.limbs)) {
            // self < other, so add modulus first
            const tmp = add192WithCarry(self.limbs, MODULUS).result;
            return .{ .limbs = sub192(tmp, other.limbs) };
        }
        return .{ .limbs = sub192(self.limbs, other.limbs) };
    }

    pub fn neg(self: Field192Mont) Field192Mont {
        if (self.limbs[0] == 0 and self.limbs[1] == 0 and self.limbs[2] == 0) {
            return ZERO;
        }
        return .{ .limbs = sub192(MODULUS, self.limbs) };
    }

    pub fn mul(self: Field192Mont, other: Field192Mont) Field192Mont {
        // Montgomery multiplication preserves the form:
        // (aR) * (bR) * R^(-1) = abR mod N
        return .{ .limbs = montMul(self.limbs, other.limbs) };
    }

    pub fn inv(self: Field192Mont) ?Field192Mont {
        if (self.limbs[0] == 0 and self.limbs[1] == 0 and self.limbs[2] == 0) {
            return null;
        }
        // Use Fermat's little theorem: a^(-1) = a^(p-2) mod p
        // In Montgomery form: (aR)^(p-2) * R^(p-3) mod N
        // We need to adjust for Montgomery: result should be a^(-1) * R
        // a^(-1) * R = (aR)^(-1) * R^2 = REDC((aR)^(p-2) * R^2)
        const p_minus_2 = subOne(subOne(MODULUS));
        const raw_result = self.powBig(p_minus_2);
        // The result is already in Montgomery form after powBig
        return raw_result;
    }

    fn subOne(x: [3]u64) [3]u64 {
        var result = x;
        if (result[0] > 0) {
            result[0] -= 1;
        } else {
            result[0] = std.math.maxInt(u64);
            if (result[1] > 0) {
                result[1] -= 1;
            } else {
                result[1] = std.math.maxInt(u64);
                result[2] -= 1;
            }
        }
        return result;
    }

    pub fn pow(self: Field192Mont, exp: u64) Field192Mont {
        var result = ONE;
        var base = self;
        var e = exp;

        while (e > 0) {
            if (e & 1 == 1) {
                result = result.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        return result;
    }

    fn powBig(self: Field192Mont, exp: [3]u64) Field192Mont {
        var result = ONE;
        var base = self;

        for (0..3) |limb_idx| {
            var e = exp[limb_idx];
            for (0..64) |_| {
                if (e & 1 == 1) {
                    result = result.mul(base);
                }
                base = base.mul(base);
                e >>= 1;
            }
        }
        return result;
    }

    pub fn div(self: Field192Mont, other: Field192Mont) ?Field192Mont {
        const inv_other = other.inv() orelse return null;
        return self.mul(inv_other);
    }

    pub fn eq(self: Field192Mont, other: Field192Mont) bool {
        return self.limbs[0] == other.limbs[0] and
            self.limbs[1] == other.limbs[1] and
            self.limbs[2] == other.limbs[2];
    }

    pub fn getRootOfUnity(log_size: u32) ?Field192Mont {
        if (log_size > TWO_ADICITY) return null;
        // Start with the 2^64-th root of unity (already in Montgomery form)
        var root = Field192Mont{ .limbs = TWO_ADIC_ROOT_OF_UNITY };
        var i: u32 = TWO_ADICITY;
        while (i > log_size) : (i -= 1) {
            root = root.mul(root);
        }
        return root;
    }

    pub fn getRootOfUnityForSize(size: usize) ?Field192Mont {
        if (size == 0 or (size & (size - 1)) != 0) return null;
        const log_size = @ctz(size);
        return getRootOfUnity(@intCast(log_size));
    }

    pub fn format(
        self: Field192Mont,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        // Display in standard form for readability
        const standard = self.toStandard();
        try writer.print("{x}_{x}_{x}", .{ standard[2], standard[1], standard[0] });
    }
};

/// 192-bit field WITHOUT Montgomery (original implementation for comparison)
/// Modulus: 4787605948707450321761805915146316350821882368518086721537
/// = 0xc340f039bc837d70_16ce1859e23e15db_0000000000000001
/// This is stored as 3 x 64-bit limbs in little-endian order
pub const Field192 = struct {
    limbs: [3]u64,

    // Modulus: 4787605948707450321761805915146316350821882368518086721537
    // In little-endian limbs:
    pub const MODULUS: [3]u64 = .{
        0x0000000000000001, // limb 0 (least significant)
        0x16ce1859e23e15db, // limb 1
        0xc340f039bc837d70, // limb 2 (most significant)
    };

    pub const ZERO: Field192 = .{ .limbs = .{ 0, 0, 0 } };
    pub const ONE: Field192 = .{ .limbs = .{ 1, 0, 0 } };
    pub const GENERATOR: Field192 = .{ .limbs = .{ 3, 0, 0 } };

    // Two-adicity: largest k such that 2^k divides p-1
    // p-1 = 2^64 * 259536638529657107390708680683681617371
    pub const TWO_ADICITY: u32 = 64;

    // 2^64-th root of unity = g^((p-1)/2^64) where g=3 is the generator
    pub const TWO_ADIC_ROOT_OF_UNITY: [3]u64 = .{
        0xf6042b33ce8ed6f6,
        0x436f4f786c7c9206,
        0x4ab9cc27b535e88e,
    };

    pub fn init(value: u64) Field192 {
        var result = Field192{ .limbs = .{ value, 0, 0 } };
        return result.reduce();
    }

    pub fn fromLimbs(limbs: [3]u64) Field192 {
        var result = Field192{ .limbs = limbs };
        return result.reduce();
    }

    fn lessThan(a: [3]u64, b: [3]u64) bool {
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    }

    fn reduce(self: Field192) Field192 {
        var result = self.limbs;
        while (!lessThan(result, MODULUS)) {
            result = sub192(result, MODULUS);
        }
        return .{ .limbs = result };
    }

    // Returns result and carry flag (true if overflow)
    fn add192WithCarry(a: [3]u64, b: [3]u64) struct { result: [3]u64, carry: bool } {
        var result: [3]u64 = undefined;
        var carry: u64 = 0;

        const r0 = @addWithOverflow(a[0], b[0]);
        result[0] = r0[0];
        carry = r0[1];

        const r1a = @addWithOverflow(a[1], b[1]);
        const r1b = @addWithOverflow(r1a[0], carry);
        result[1] = r1b[0];
        carry = r1a[1] + r1b[1];

        const r2a = @addWithOverflow(a[2], b[2]);
        const r2b = @addWithOverflow(r2a[0], carry);
        result[2] = r2b[0];
        const final_carry = (r2a[1] + r2b[1]) > 0;

        return .{ .result = result, .carry = final_carry };
    }

    fn add192(a: [3]u64, b: [3]u64) [3]u64 {
        return add192WithCarry(a, b).result;
    }

    fn sub192(a: [3]u64, b: [3]u64) [3]u64 {
        var result: [3]u64 = undefined;
        var borrow: u64 = 0;

        const r0 = @subWithOverflow(a[0], b[0]);
        result[0] = r0[0];
        borrow = r0[1];

        const r1a = @subWithOverflow(a[1], b[1]);
        const r1b = @subWithOverflow(r1a[0], borrow);
        result[1] = r1b[0];
        borrow = r1a[1] + r1b[1];

        result[2] = a[2] -% b[2] -% borrow;
        return result;
    }

    pub fn add(self: Field192, other: Field192) Field192 {
        const added = add192WithCarry(self.limbs, other.limbs);
        var result = added.result;

        if (added.carry) {
            // Overflow: true value is result + 2^192
            // 2^192 mod p = R, so we add R and reduce
            // R = 2^192 - p (precomputed in reduce384)
            const R: [3]u64 = .{
                0xffffffffffffffff,
                0xe931e7a61dc1ea24,
                0x3cbf0fc6437c828f,
            };
            const added2 = add192WithCarry(result, R);
            result = added2.result;
            // added2.carry should be false since result < 2^192 and R < p < 2^192
            // and result + R < 2^192 + p (since result came from overflow which means
            // original sum was >= 2^192, so result < 2p - 2^192 < p)
        }

        // Reduce if >= modulus
        if (!lessThan(result, MODULUS)) {
            result = sub192(result, MODULUS);
        }
        return .{ .limbs = result };
    }

    pub fn sub(self: Field192, other: Field192) Field192 {
        if (lessThan(self.limbs, other.limbs)) {
            // self < other, so add modulus first
            const tmp = add192(self.limbs, MODULUS);
            return .{ .limbs = sub192(tmp, other.limbs) };
        }
        return .{ .limbs = sub192(self.limbs, other.limbs) };
    }

    pub fn neg(self: Field192) Field192 {
        if (self.limbs[0] == 0 and self.limbs[1] == 0 and self.limbs[2] == 0) {
            return ZERO;
        }
        return .{ .limbs = sub192(MODULUS, self.limbs) };
    }

    // Multiply two 192-bit numbers, get 384-bit result, then reduce
    pub fn mul(self: Field192, other: Field192) Field192 {
        // Full 384-bit multiplication using schoolbook method
        var result: [6]u64 = .{ 0, 0, 0, 0, 0, 0 };

        for (0..3) |i| {
            var carry: u64 = 0;
            for (0..3) |j| {
                const product: u128 = @as(u128, self.limbs[i]) * @as(u128, other.limbs[j]) +
                    @as(u128, result[i + j]) + @as(u128, carry);
                result[i + j] = @truncate(product);
                carry = @truncate(product >> 64);
            }
            result[i + 3] = carry;
        }

        return reduce384(result);
    }

    fn reduce384(x: [6]u64) Field192 {
        // Reduce 384-bit value modulo p
        // x = x[0] + x[1]*2^64 + x[2]*2^128 + x[3]*2^192 + x[4]*2^256 + x[5]*2^320
        // x = x_low + x_high * 2^192  where x_low = x[0..2], x_high = x[3..5]
        //
        // We use: 2^192 mod p = R where R is precomputed
        // Then: x mod p = (x_low + x_high * R) mod p
        //
        // Since x_high < 2^192 and R < p < 2^192, x_high * R < 2^384
        // We may need to apply reduction twice

        // First, if high part is zero, just reduce low part
        if (x[3] == 0 and x[4] == 0 and x[5] == 0) {
            var result: [3]u64 = .{ x[0], x[1], x[2] };
            while (!lessThan(result, MODULUS)) {
                result = sub192(result, MODULUS);
            }
            return .{ .limbs = result };
        }

        // 2^192 mod p precomputed (calculated in Python):
        // p = 0xc340f039bc837d70_16ce1859e23e15db_0000000000000001
        // R = 2^192 mod p = 0x3cbf0fc6437c828f_e931e7a61dc1ea24_ffffffffffffffff
        const R: [3]u64 = .{
            0xffffffffffffffff,
            0xe931e7a61dc1ea24,
            0x3cbf0fc6437c828f,
        };

        // x_high = [x[3], x[4], x[5]]
        // Compute x_high * R (this gives a 384-bit result)
        var product: [6]u64 = .{ 0, 0, 0, 0, 0, 0 };
        const x_high: [3]u64 = .{ x[3], x[4], x[5] };

        for (0..3) |i| {
            var carry: u64 = 0;
            for (0..3) |j| {
                const p128: u128 = @as(u128, x_high[i]) * @as(u128, R[j]) +
                    @as(u128, product[i + j]) + @as(u128, carry);
                product[i + j] = @truncate(p128);
                carry = @truncate(p128 >> 64);
            }
            product[i + 3] = carry;
        }

        // Now add x_low to product
        // result = x_low + product (up to 385 bits, but we know it's < 2*p*2^192 so fits)
        var sum: [6]u64 = undefined;
        var carry: u64 = 0;

        const add0 = @addWithOverflow(product[0], x[0]);
        sum[0] = add0[0];
        carry = add0[1];

        const add1a = @addWithOverflow(product[1], x[1]);
        const add1b = @addWithOverflow(add1a[0], carry);
        sum[1] = add1b[0];
        carry = add1a[1] + add1b[1];

        const add2a = @addWithOverflow(product[2], x[2]);
        const add2b = @addWithOverflow(add2a[0], carry);
        sum[2] = add2b[0];
        carry = add2a[1] + add2b[1];

        const add3 = @addWithOverflow(product[3], carry);
        sum[3] = add3[0];
        carry = add3[1];

        const add4 = @addWithOverflow(product[4], carry);
        sum[4] = add4[0];
        carry = add4[1];

        sum[5] = product[5] +% carry;

        // If sum still has high bits, recurse (apply reduction again)
        if (sum[3] != 0 or sum[4] != 0 or sum[5] != 0) {
            return reduce384(sum);
        }

        // Final reduction of the 192-bit result
        var result: [3]u64 = .{ sum[0], sum[1], sum[2] };
        while (!lessThan(result, MODULUS)) {
            result = sub192(result, MODULUS);
        }
        return .{ .limbs = result };
    }

    pub fn inv(self: Field192) ?Field192 {
        if (self.limbs[0] == 0 and self.limbs[1] == 0 and self.limbs[2] == 0) {
            return null;
        }
        // Use Fermat's little theorem: a^(-1) = a^(p-2) mod p
        const p_minus_2 = subOne(subOne(MODULUS));
        return self.powBig(p_minus_2);
    }

    pub fn subOne(x: [3]u64) [3]u64 {
        var result = x;
        if (result[0] > 0) {
            result[0] -= 1;
        } else {
            result[0] = std.math.maxInt(u64);
            if (result[1] > 0) {
                result[1] -= 1;
            } else {
                result[1] = std.math.maxInt(u64);
                result[2] -= 1;
            }
        }
        return result;
    }

    pub fn pow(self: Field192, exp: u64) Field192 {
        var result = ONE;
        var base = self;
        var e = exp;

        while (e > 0) {
            if (e & 1 == 1) {
                result = result.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        return result;
    }

    fn powBig(self: Field192, exp: [3]u64) Field192 {
        var result = ONE;
        var base = self;
        var iteration: usize = 0;

        for (0..3) |limb_idx| {
            var e = exp[limb_idx];
            for (0..64) |bit_idx| {
                if (e & 1 == 1) {
                    result = result.mul(base);
                }
                const old_base = base;
                base = base.mul(base);

                // Debug: check if squaring produces something >= p
                if (iteration < 10 or (iteration % 20 == 0)) {
                    _ = old_base;
                    // std.debug.print("iter {d}: base^2 limb2={x}\n", .{ iteration, base.limbs[2] });
                }

                e >>= 1;
                iteration += 1;
                _ = bit_idx;
            }
        }
        return result;
    }

    pub fn div(self: Field192, other: Field192) ?Field192 {
        const inv_other = other.inv() orelse return null;
        return self.mul(inv_other);
    }

    pub fn eq(self: Field192, other: Field192) bool {
        return self.limbs[0] == other.limbs[0] and
            self.limbs[1] == other.limbs[1] and
            self.limbs[2] == other.limbs[2];
    }

    pub fn getRootOfUnity(log_size: u32) ?Field192 {
        if (log_size > TWO_ADICITY) return null;
        // Start with the 2^64-th root of unity and square appropriately
        var root = Field192{ .limbs = TWO_ADIC_ROOT_OF_UNITY };
        var i: u32 = TWO_ADICITY;
        while (i > log_size) : (i -= 1) {
            root = root.mul(root);
        }
        return root;
    }

    pub fn getRootOfUnityForSize(size: usize) ?Field192 {
        if (size == 0 or (size & (size - 1)) != 0) return null;
        const log_size = @ctz(size);
        return getRootOfUnity(@intCast(log_size));
    }

    pub fn format(
        self: Field192,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{x}_{x}_{x}", .{ self.limbs[2], self.limbs[1], self.limbs[0] });
    }
};

/// Goldilocks field: p = 2^64 - 2^32 + 1 = 18446744069414584321
/// This is the same modulus used in Field64 in the Rust implementation
pub const Field64 = struct {
    value: u64,

    pub const MODULUS: u64 = 18446744069414584321;
    pub const GENERATOR: u64 = 7;
    // Two-adicity: The largest k such that 2^k divides p-1
    // p - 1 = 2^32 * 4294967295, so TWO_ADICITY = 32
    pub const TWO_ADICITY: u32 = 32;
    // 2^32-th root of unity
    pub const TWO_ADIC_ROOT_OF_UNITY: u64 = 1753635133440165772;

    pub const ZERO: Field64 = .{ .value = 0 };
    pub const ONE: Field64 = .{ .value = 1 };

    pub fn init(value: u64) Field64 {
        return .{ .value = value % MODULUS };
    }

    pub fn fromU128(value: u128) Field64 {
        return .{ .value = @intCast(value % MODULUS) };
    }

    pub fn add(self: Field64, other: Field64) Field64 {
        var sum: u64 = undefined;
        const overflow = @addWithOverflow(self.value, other.value);
        sum = overflow[0];
        if (overflow[1] != 0 or sum >= MODULUS) {
            sum -%= MODULUS;
        }
        return .{ .value = sum };
    }

    pub fn sub(self: Field64, other: Field64) Field64 {
        if (self.value >= other.value) {
            return .{ .value = self.value - other.value };
        } else {
            return .{ .value = MODULUS - (other.value - self.value) };
        }
    }

    pub fn mul(self: Field64, other: Field64) Field64 {
        const product: u128 = @as(u128, self.value) * @as(u128, other.value);
        return reduce128(product);
    }

    fn reduce128(x: u128) Field64 {
        // Barrett reduction for Goldilocks field
        // Since p = 2^64 - 2^32 + 1, we can use special reduction
        const low: u64 = @truncate(x);
        const high: u64 = @truncate(x >> 64);

        // x = low + high * 2^64
        // 2^64 = 2^32 - 1 (mod p)
        // So x = low + high * (2^32 - 1) (mod p)
        const high_times_2_32: u128 = @as(u128, high) << 32;
        const adjusted = @as(u128, low) +% high_times_2_32 -% @as(u128, high);

        // May need another round of reduction
        if (adjusted >= @as(u128, MODULUS) * 2) {
            const low2: u64 = @truncate(adjusted);
            const high2: u64 = @truncate(adjusted >> 64);
            const high2_times_2_32: u128 = @as(u128, high2) << 32;
            const result = @as(u128, low2) +% high2_times_2_32 -% @as(u128, high2);
            return .{ .value = @intCast(result % MODULUS) };
        }

        return .{ .value = @intCast(adjusted % MODULUS) };
    }

    pub fn neg(self: Field64) Field64 {
        if (self.value == 0) {
            return ZERO;
        }
        return .{ .value = MODULUS - self.value };
    }

    pub fn inv(self: Field64) ?Field64 {
        if (self.value == 0) {
            return null;
        }
        // Extended Euclidean Algorithm
        return self.pow(MODULUS - 2);
    }

    pub fn div(self: Field64, other: Field64) ?Field64 {
        const inv_other = other.inv() orelse return null;
        return self.mul(inv_other);
    }

    pub fn pow(self: Field64, exp: u64) Field64 {
        var result = ONE;
        var base = self;
        var e = exp;

        while (e > 0) {
            if (e & 1 == 1) {
                result = result.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }

        return result;
    }

    pub fn eq(self: Field64, other: Field64) bool {
        return self.value == other.value;
    }

    /// Get a primitive root of unity for domain of size 2^log_size
    pub fn getRootOfUnity(log_size: u32) ?Field64 {
        if (log_size > TWO_ADICITY) {
            return null;
        }
        // Start with the 2^32-th root of unity and square it appropriately
        var root = Field64.init(TWO_ADIC_ROOT_OF_UNITY);
        var i: u32 = TWO_ADICITY;
        while (i > log_size) : (i -= 1) {
            root = root.mul(root);
        }
        return root;
    }

    /// Get a root of unity for a domain of the given size
    pub fn getRootOfUnityForSize(size: usize) ?Field64 {
        if (size == 0 or (size & (size - 1)) != 0) {
            // Size must be a power of 2
            return null;
        }
        const log_size = @ctz(size);
        return getRootOfUnity(@intCast(log_size));
    }

    pub fn format(
        self: Field64,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d}", .{self.value});
    }
};

/// The field type used throughout the STIR implementation.
/// Field192Mont uses Montgomery representation (matching Rust's ark-ff).
/// Change to Field192 for non-Montgomery, or Field64 for faster benchmarks.
pub const Field = Field192Mont;

// Tests
test "field arithmetic" {
    const a = Field64.init(5);
    const b = Field64.init(3);

    // Addition
    try std.testing.expectEqual(Field64.init(8), a.add(b));

    // Subtraction
    try std.testing.expectEqual(Field64.init(2), a.sub(b));

    // Multiplication
    try std.testing.expectEqual(Field64.init(15), a.mul(b));

    // Negation
    const neg_a = a.neg();
    try std.testing.expectEqual(Field64.ZERO, a.add(neg_a));

    // Inverse
    const inv_a = a.inv().?;
    try std.testing.expectEqual(Field64.ONE, a.mul(inv_a));

    // Power
    try std.testing.expectEqual(Field64.init(125), a.pow(3));
}

test "field modular reduction" {
    // Test with values near modulus
    const a = Field64.init(Field64.MODULUS - 1);
    const b = Field64.init(2);
    const result = a.add(b);
    try std.testing.expectEqual(Field64.init(1), result);
}

test "root of unity" {
    // Test that root of unity has correct order
    const log_size: u32 = 4;
    const size: u64 = 1 << log_size;
    const root = Field64.getRootOfUnity(log_size).?;

    // root^size should be 1
    try std.testing.expectEqual(Field64.ONE, root.pow(size));

    // root^(size/2) should not be 1
    try std.testing.expect(!root.pow(size / 2).eq(Field64.ONE));
}

test "field multiplication edge cases" {
    const max_val = Field64.init(Field64.MODULUS - 1);
    const result = max_val.mul(max_val);
    // (p-1)^2 mod p = 1
    try std.testing.expectEqual(Field64.ONE, result);
}

// Field192 tests
test "field192 basic arithmetic" {
    const a = Field192.init(5);
    const b = Field192.init(3);

    // Addition
    try std.testing.expect(a.add(b).eq(Field192.init(8)));

    // Subtraction
    try std.testing.expect(a.sub(b).eq(Field192.init(2)));

    // Multiplication
    try std.testing.expect(a.mul(b).eq(Field192.init(15)));

    // Negation
    const neg_a = a.neg();
    try std.testing.expect(a.add(neg_a).eq(Field192.ZERO));
}

test "field192 multiplication" {
    // Test small values
    const two = Field192.init(2);
    const three = Field192.init(3);
    const six = two.mul(three);
    try std.testing.expect(six.eq(Field192.init(6)));

    // Test larger value
    const big = Field192.init(1 << 32);
    const result = big.mul(big);
    // (2^32)^2 = 2^64, should reduce properly
    try std.testing.expect(result.limbs[1] == 1);
    try std.testing.expect(result.limbs[0] == 0);
}

test "field192 pow" {
    const a = Field192.init(7);

    // Test a^2
    const a2 = a.mul(a);
    try std.testing.expect(a2.eq(Field192.init(49)));

    // Test a^4
    const a4 = a2.mul(a2);
    try std.testing.expect(a4.eq(Field192.init(2401)));

    // Test pow function for small exponents
    try std.testing.expect(a.pow(1).eq(Field192.init(7)));
    try std.testing.expect(a.pow(2).eq(Field192.init(49)));
    try std.testing.expect(a.pow(3).eq(Field192.init(343)));
    try std.testing.expect(a.pow(4).eq(Field192.init(2401)));
    try std.testing.expect(a.pow(10).eq(Field192.init(282475249)));

    // Test larger power
    const a_100 = a.pow(100);
    std.debug.print("\n7^100 = [{x}, {x}, {x}]\n", .{ a_100.limbs[0], a_100.limbs[1], a_100.limbs[2] });
}

test "field192 squaring chain" {
    // Test that repeated squaring works correctly
    var x = Field192.init(7);
    std.debug.print("\nSquaring chain:\n", .{});

    for (0..10) |i| {
        const x_squared = x.mul(x);
        const x_pow2 = x.pow(2);

        if (!x_squared.eq(x_pow2)) {
            std.debug.print("Mismatch at iteration {d}!\n", .{i});
            std.debug.print("x = [{x}, {x}, {x}]\n", .{ x.limbs[0], x.limbs[1], x.limbs[2] });
            std.debug.print("x*x = [{x}, {x}, {x}]\n", .{ x_squared.limbs[0], x_squared.limbs[1], x_squared.limbs[2] });
            std.debug.print("x^2 = [{x}, {x}, {x}]\n", .{ x_pow2.limbs[0], x_pow2.limbs[1], x_pow2.limbs[2] });
        }
        try std.testing.expect(x_squared.eq(x_pow2));

        // Also verify result is less than modulus
        if (!Field192.lessThan(x_squared.limbs, Field192.MODULUS)) {
            std.debug.print("Result >= MODULUS at iteration {d}!\n", .{i});
        }
        try std.testing.expect(Field192.lessThan(x_squared.limbs, Field192.MODULUS));

        x = x_squared;
        std.debug.print("iter {d}: x = [{x}, {x}, {x}]\n", .{ i, x.limbs[0], x.limbs[1], x.limbs[2] });
    }
}

test "field192 inverse" {
    const a = Field192.init(7);

    // First verify that a^(p-1) = 1 (Fermat's little theorem)
    const p_minus_1 = Field192.subOne(Field192.MODULUS);
    const fermat = a.powBig(p_minus_1);
    std.debug.print("\n7^(p-1) = [{x}, {x}, {x}]\n", .{ fermat.limbs[0], fermat.limbs[1], fermat.limbs[2] });
    try std.testing.expect(fermat.eq(Field192.ONE));

    // Now test inverse
    const inv_a = a.inv().?;
    const product = a.mul(inv_a);

    std.debug.print("7 * 7^(-1) = [{x}, {x}, {x}]\n", .{ product.limbs[0], product.limbs[1], product.limbs[2] });

    try std.testing.expect(product.eq(Field192.ONE));
}

test "field192 mul edge case" {
    // (p-1)^2 mod p should equal 1
    const p_minus_1_limbs = Field192.subOne(Field192.MODULUS);
    const p_minus_1 = Field192{ .limbs = p_minus_1_limbs };
    const result = p_minus_1.mul(p_minus_1);
    try std.testing.expect(result.eq(Field192.ONE));
}

test "field192 reduction constant" {
    // Verify that reduce384 works correctly for 2^192
    // 2^192 mod p should equal R
    const x: [6]u64 = .{ 0, 0, 0, 1, 0, 0 }; // represents 2^192
    const result = Field192.reduce384(x);

    // Expected R = 2^192 mod p (calculated in Python)
    const expected_R: [3]u64 = .{
        0xffffffffffffffff,
        0xe931e7a61dc1ea24,
        0x3cbf0fc6437c828f,
    };

    std.debug.print("\n2^192 mod p:\n", .{});
    std.debug.print("result = [{x}, {x}, {x}]\n", .{ result.limbs[0], result.limbs[1], result.limbs[2] });
    std.debug.print("expected = [{x}, {x}, {x}]\n", .{ expected_R[0], expected_R[1], expected_R[2] });

    try std.testing.expect(result.limbs[0] == expected_R[0]);
    try std.testing.expect(result.limbs[1] == expected_R[1]);
    try std.testing.expect(result.limbs[2] == expected_R[2]);
}

test "field192 root of unity" {
    // Test that the 2^64-th root of unity has correct order
    var root = Field192{ .limbs = Field192.TWO_ADIC_ROOT_OF_UNITY };

    // root^(2^64) should be 1
    for (0..64) |_| {
        root = root.mul(root);
    }
    try std.testing.expect(root.eq(Field192.ONE));

    // Test getRootOfUnity for smaller sizes
    const root_16 = Field192.getRootOfUnity(4).?; // 2^4 = 16
    var r = root_16;
    for (0..4) |_| {
        r = r.mul(r);
    }
    try std.testing.expect(r.eq(Field192.ONE));

    // root_16^8 should not be 1 (primitive)
    r = root_16;
    for (0..3) |_| {
        r = r.mul(r);
    }
    try std.testing.expect(!r.eq(Field192.ONE));
}

test "field192 large mul" {
    // Test: (p-1) * 2 = 2p - 2 mod p = -2 mod p = p - 2
    const p_minus_1_limbs = Field192.subOne(Field192.MODULUS);
    const p_minus_1 = Field192{ .limbs = p_minus_1_limbs };
    const two = Field192.init(2);
    const result = p_minus_1.mul(two);

    // Expected: p - 2 = subOne(subOne(MODULUS))
    const expected = Field192.subOne(Field192.subOne(Field192.MODULUS));

    std.debug.print("\n(p-1) * 2:\n", .{});
    std.debug.print("result = [{x}, {x}, {x}]\n", .{ result.limbs[0], result.limbs[1], result.limbs[2] });
    std.debug.print("expected = [{x}, {x}, {x}]\n", .{ expected[0], expected[1], expected[2] });

    try std.testing.expect(result.limbs[0] == expected[0]);
    try std.testing.expect(result.limbs[1] == expected[1]);
    try std.testing.expect(result.limbs[2] == expected[2]);
}

// ============== Montgomery Field192 Tests ==============

test "field192mont basic arithmetic" {
    const a = Field192Mont.init(5);
    const b = Field192Mont.init(3);

    // Addition: 5 + 3 = 8
    const sum = a.add(b);
    try std.testing.expect(sum.eq(Field192Mont.init(8)));

    // Subtraction: 5 - 3 = 2
    const diff = a.sub(b);
    try std.testing.expect(diff.eq(Field192Mont.init(2)));

    // Multiplication: 5 * 3 = 15
    const prod = a.mul(b);
    try std.testing.expect(prod.eq(Field192Mont.init(15)));

    // Negation: 5 + (-5) = 0
    const neg_a = a.neg();
    try std.testing.expect(a.add(neg_a).eq(Field192Mont.ZERO));
}

test "field192mont conversion to/from standard" {
    // Test that converting to Montgomery and back gives original value
    const original: [3]u64 = .{ 12345, 67890, 11111 };
    const mont = Field192Mont.fromLimbs(original);
    const back = mont.toStandard();

    std.debug.print("\nMontgomery conversion test:\n", .{});
    std.debug.print("original = [{x}, {x}, {x}]\n", .{ original[0], original[1], original[2] });
    std.debug.print("montgomery = [{x}, {x}, {x}]\n", .{ mont.limbs[0], mont.limbs[1], mont.limbs[2] });
    std.debug.print("back = [{x}, {x}, {x}]\n", .{ back[0], back[1], back[2] });

    try std.testing.expect(back[0] == original[0]);
    try std.testing.expect(back[1] == original[1]);
    try std.testing.expect(back[2] == original[2]);
}

test "field192mont ONE constant" {
    // ONE in Montgomery form should convert to [1, 0, 0] in standard form
    const one_standard = Field192Mont.ONE.toStandard();
    std.debug.print("\nONE in standard form: [{x}, {x}, {x}]\n", .{ one_standard[0], one_standard[1], one_standard[2] });
    try std.testing.expect(one_standard[0] == 1);
    try std.testing.expect(one_standard[1] == 0);
    try std.testing.expect(one_standard[2] == 0);

    // a * 1 = a
    const a = Field192Mont.init(42);
    try std.testing.expect(a.mul(Field192Mont.ONE).eq(a));
}

test "field192mont GENERATOR constant" {
    // GENERATOR in Montgomery form should convert to [3, 0, 0] in standard form
    const gen_standard = Field192Mont.GENERATOR.toStandard();
    std.debug.print("\nGENERATOR in standard form: [{x}, {x}, {x}]\n", .{ gen_standard[0], gen_standard[1], gen_standard[2] });
    try std.testing.expect(gen_standard[0] == 3);
    try std.testing.expect(gen_standard[1] == 0);
    try std.testing.expect(gen_standard[2] == 0);
}

test "field192mont multiplication" {
    // Test small values
    const two = Field192Mont.init(2);
    const three = Field192Mont.init(3);
    const six = two.mul(three);
    try std.testing.expect(six.eq(Field192Mont.init(6)));

    // Test larger value: 2^32 * 2^32 = 2^64
    const big = Field192Mont.init(1 << 32);
    const result = big.mul(big);
    const result_std = result.toStandard();
    // 2^64 should have limb[1] = 1, limb[0] = 0
    try std.testing.expect(result_std[0] == 0);
    try std.testing.expect(result_std[1] == 1);
    try std.testing.expect(result_std[2] == 0);
}

test "field192mont inverse" {
    const a = Field192Mont.init(7);

    // Test inverse: a * a^(-1) = 1
    const inv_a = a.inv().?;
    const product = a.mul(inv_a);

    std.debug.print("\n7 * 7^(-1) in Montgomery:\n", .{});
    const prod_std = product.toStandard();
    std.debug.print("result = [{x}, {x}, {x}]\n", .{ prod_std[0], prod_std[1], prod_std[2] });

    try std.testing.expect(product.eq(Field192Mont.ONE));
}

test "field192mont pow" {
    const a = Field192Mont.init(7);

    // Test powers
    try std.testing.expect(a.pow(1).eq(Field192Mont.init(7)));
    try std.testing.expect(a.pow(2).eq(Field192Mont.init(49)));
    try std.testing.expect(a.pow(3).eq(Field192Mont.init(343)));
    try std.testing.expect(a.pow(4).eq(Field192Mont.init(2401)));
}

test "field192mont root of unity" {
    // Test that the 2^64-th root of unity has correct order
    var root = Field192Mont{ .limbs = Field192Mont.TWO_ADIC_ROOT_OF_UNITY };

    // root^(2^64) should be 1
    for (0..64) |_| {
        root = root.mul(root);
    }
    try std.testing.expect(root.eq(Field192Mont.ONE));

    // Test getRootOfUnity for smaller sizes
    const root_16 = Field192Mont.getRootOfUnity(4).?; // 2^4 = 16
    var r = root_16;
    for (0..4) |_| {
        r = r.mul(r);
    }
    try std.testing.expect(r.eq(Field192Mont.ONE));

    // root_16^8 should not be 1 (primitive)
    r = root_16;
    for (0..3) |_| {
        r = r.mul(r);
    }
    try std.testing.expect(!r.eq(Field192Mont.ONE));
}

test "field192mont mul edge case" {
    // (p-1)^2 mod p should equal 1
    const p_minus_1_limbs = Field192Mont.subOne(Field192Mont.MODULUS);
    const p_minus_1 = Field192Mont.fromLimbs(p_minus_1_limbs);
    const result = p_minus_1.mul(p_minus_1);
    try std.testing.expect(result.eq(Field192Mont.ONE));
}

test "field192mont matches field192" {
    // Verify that Montgomery and non-Montgomery give same results
    const a_mont = Field192Mont.init(12345);
    const b_mont = Field192Mont.init(67890);
    const a_std = Field192.init(12345);
    const b_std = Field192.init(67890);

    // Addition
    const sum_mont = a_mont.add(b_mont).toStandard();
    const sum_std = a_std.add(b_std).limbs;
    try std.testing.expect(sum_mont[0] == sum_std[0]);
    try std.testing.expect(sum_mont[1] == sum_std[1]);
    try std.testing.expect(sum_mont[2] == sum_std[2]);

    // Multiplication
    const prod_mont = a_mont.mul(b_mont).toStandard();
    const prod_std = a_std.mul(b_std).limbs;
    try std.testing.expect(prod_mont[0] == prod_std[0]);
    try std.testing.expect(prod_mont[1] == prod_std[1]);
    try std.testing.expect(prod_mont[2] == prod_std[2]);

    // Power
    const pow_mont = a_mont.pow(17).toStandard();
    const pow_std = a_std.pow(17).limbs;
    try std.testing.expect(pow_mont[0] == pow_std[0]);
    try std.testing.expect(pow_mont[1] == pow_std[1]);
    try std.testing.expect(pow_mont[2] == pow_std[2]);
}
