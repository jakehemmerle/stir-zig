const Field = @import("field.zig").Field;

/// A point (x, y) used in interpolation
pub const Point = struct {
    x: Field,
    y: Field,
};
