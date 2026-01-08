const std = @import("std");
const Field = @import("field.zig").Field;
const Allocator = std.mem.Allocator;

/// Global hash counter for benchmarking
pub var hash_counter: usize = 0;

pub fn resetHashCounter() void {
    hash_counter = 0;
}

pub fn getHashCount() usize {
    return hash_counter;
}

/// Simple Blake3-based Merkle tree implementation
pub const MerkleTree = struct {
    /// All nodes in the tree, stored level by level (root at index 0)
    nodes: []Hash,
    /// Number of leaves
    leaf_count: usize,
    allocator: Allocator,

    pub const Hash = [32]u8;
    pub const EMPTY_HASH: Hash = [_]u8{0} ** 32;

    pub fn init(allocator: Allocator, leaves: []const []const Field) !MerkleTree {
        if (leaves.len == 0) {
            return error.EmptyTree;
        }

        // Round up to power of 2
        const leaf_count = leaves.len;
        const padded_count = std.math.ceilPowerOfTwo(usize, leaf_count) catch leaf_count;

        // Total nodes in a complete binary tree with padded_count leaves
        const total_nodes = 2 * padded_count - 1;

        var nodes = try allocator.alloc(Hash, total_nodes);
        errdefer allocator.free(nodes);

        // Hash leaves and place at the end (leaves are at the last padded_count positions)
        const leaf_start = padded_count - 1;
        for (0..padded_count) |i| {
            if (i < leaf_count) {
                nodes[leaf_start + i] = hashLeaf(leaves[i]);
            } else {
                nodes[leaf_start + i] = EMPTY_HASH;
            }
        }

        // Build internal nodes bottom-up
        if (padded_count > 1) {
            var level_size = padded_count / 2;
            var level_start = leaf_start - level_size;
            var child_start = leaf_start;

            while (level_size >= 1) {
                for (0..level_size) |i| {
                    const left = nodes[child_start + 2 * i];
                    const right = nodes[child_start + 2 * i + 1];
                    nodes[level_start + i] = hashPair(left, right);
                }

                if (level_size == 1) break;

                child_start = level_start;
                level_size /= 2;
                level_start -= level_size;
            }
        }

        return MerkleTree{
            .nodes = nodes,
            .leaf_count = leaf_count,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MerkleTree) void {
        self.allocator.free(self.nodes);
    }

    pub fn root(self: MerkleTree) Hash {
        return self.nodes[0];
    }

    /// Generate a proof for the leaf at the given index
    pub fn generateProof(self: MerkleTree, index: usize) !MerkleProof {
        if (index >= self.leaf_count) {
            return error.IndexOutOfBounds;
        }

        const padded_count = std.math.ceilPowerOfTwo(usize, self.leaf_count) catch self.leaf_count;
        const log_size = @ctz(padded_count);

        var path = try self.allocator.alloc(Hash, log_size);
        errdefer self.allocator.free(path);

        var current_idx = padded_count - 1 + index; // Index in node array

        for (0..log_size) |level| {
            // Sibling index
            const sibling_idx = if (current_idx % 2 == 0) current_idx - 1 else current_idx + 1;
            path[level] = self.nodes[sibling_idx];
            // Parent index
            current_idx = (current_idx - 1) / 2;
        }

        return MerkleProof{
            .path = path,
            .leaf_index = index,
            .allocator = self.allocator,
        };
    }

    /// Generate proofs for multiple indices
    pub fn generateMultiProof(self: MerkleTree, indices: []const usize) !MultiProof {
        var proofs = try self.allocator.alloc(MerkleProof, indices.len);
        errdefer self.allocator.free(proofs);

        for (indices, 0..) |idx, i| {
            proofs[i] = try self.generateProof(idx);
        }

        return MultiProof{
            .proofs = proofs,
            .allocator = self.allocator,
        };
    }

    fn hashLeaf(leaf: []const Field) Hash {
        hash_counter += 1;
        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(&[_]u8{0x00}); // Domain separator for leaves

        for (leaf) |f| {
            // Works with both Field64 and Field192
            const bytes = std.mem.asBytes(&f);
            hasher.update(bytes);
        }

        var result: Hash = undefined;
        hasher.final(&result);
        return result;
    }

    fn hashPair(left: Hash, right: Hash) Hash {
        hash_counter += 1;
        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(&[_]u8{0x01}); // Domain separator for internal nodes
        hasher.update(&left);
        hasher.update(&right);
        var result: Hash = undefined;
        hasher.final(&result);
        return result;
    }
};

pub const MerkleProof = struct {
    path: []MerkleTree.Hash,
    leaf_index: usize,
    allocator: Allocator,

    pub fn deinit(self: *MerkleProof) void {
        self.allocator.free(self.path);
    }

    /// Verify the proof against a given root and leaf
    pub fn verify(self: MerkleProof, expected_root: MerkleTree.Hash, leaf: []const Field) bool {
        var current_hash = hashLeaf(leaf);

        // We need to track position in the node array, not the leaf index
        // The path length tells us log2(padded_count)
        const padded_count = @as(usize, 1) << @intCast(self.path.len);
        var node_idx = padded_count - 1 + self.leaf_index;

        for (self.path) |sibling| {
            // In our tree layout (indices starting at 0):
            // - Odd node indices (3, 5, 7...) are LEFT children
            // - Even node indices (4, 6, 8...) are RIGHT children
            if (node_idx % 2 == 1) {
                // Current is left child, sibling is right
                current_hash = hashPair(current_hash, sibling);
            } else {
                // Current is right child, sibling is left
                current_hash = hashPair(sibling, current_hash);
            }
            // Move to parent
            node_idx = (node_idx - 1) / 2;
        }

        return std.mem.eql(u8, &current_hash, &expected_root);
    }

    fn hashLeaf(leaf: []const Field) MerkleTree.Hash {
        hash_counter += 1;
        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(&[_]u8{0x00});
        for (leaf) |f| {
            // Works with both Field64 and Field192
            hasher.update(std.mem.asBytes(&f));
        }
        var result: MerkleTree.Hash = undefined;
        hasher.final(&result);
        return result;
    }

    fn hashPair(left: MerkleTree.Hash, right: MerkleTree.Hash) MerkleTree.Hash {
        hash_counter += 1;
        var hasher = std.crypto.hash.Blake3.init(.{});
        hasher.update(&[_]u8{0x01});
        hasher.update(&left);
        hasher.update(&right);
        var result: MerkleTree.Hash = undefined;
        hasher.final(&result);
        return result;
    }
};

pub const MultiProof = struct {
    proofs: []MerkleProof,
    allocator: Allocator,

    pub fn deinit(self: *MultiProof) void {
        for (self.proofs) |*p| {
            p.deinit();
        }
        self.allocator.free(self.proofs);
    }

    pub fn verify(self: MultiProof, expected_root: MerkleTree.Hash, leaves: []const []const Field) bool {
        if (self.proofs.len != leaves.len) return false;

        for (self.proofs, leaves) |proof, leaf| {
            if (!proof.verify(expected_root, leaf)) {
                return false;
            }
        }

        return true;
    }
};

// Tests
test "merkle tree basic" {
    const allocator = std.testing.allocator;

    // Create some leaves
    var leaf1 = [_]Field{Field.init(1)};
    var leaf2 = [_]Field{Field.init(2)};
    var leaf3 = [_]Field{Field.init(3)};
    var leaf4 = [_]Field{Field.init(4)};

    var leaves: [4][]const Field = .{
        &leaf1,
        &leaf2,
        &leaf3,
        &leaf4,
    };

    var tree = try MerkleTree.init(allocator, &leaves);
    defer tree.deinit();

    // Root should be deterministic
    const root = tree.root();
    try std.testing.expect(!std.mem.eql(u8, &root, &MerkleTree.EMPTY_HASH));
}

test "merkle proof verification" {
    const allocator = std.testing.allocator;

    var leaf1 = [_]Field{Field.init(1)};
    var leaf2 = [_]Field{Field.init(2)};
    var leaf3 = [_]Field{Field.init(3)};
    var leaf4 = [_]Field{Field.init(4)};

    var leaves: [4][]const Field = .{
        &leaf1,
        &leaf2,
        &leaf3,
        &leaf4,
    };

    var tree = try MerkleTree.init(allocator, &leaves);
    defer tree.deinit();

    const root = tree.root();

    // Generate and verify proof for each leaf
    for (0..4) |i| {
        var proof = try tree.generateProof(i);
        defer proof.deinit();

        try std.testing.expect(proof.verify(root, leaves[i]));
    }
}

test "merkle proof invalid" {
    const allocator = std.testing.allocator;

    var leaf1 = [_]Field{Field.init(1)};
    var leaf2 = [_]Field{Field.init(2)};

    var leaves: [2][]const Field = .{ &leaf1, &leaf2 };

    var tree = try MerkleTree.init(allocator, &leaves);
    defer tree.deinit();

    const root = tree.root();

    var proof = try tree.generateProof(0);
    defer proof.deinit();

    // Wrong leaf should fail verification
    var wrong_leaf = [_]Field{Field.init(999)};
    try std.testing.expect(!proof.verify(root, &wrong_leaf));
}
