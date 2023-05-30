@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> in: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute
@workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) cell: vec3<u32>) {
    if (in[cell_index(cell.xy)] == u32(1)) {
        out[cell_index(cell.xy)] = u32(0);
    } else {
        out[cell_index(cell.xy)] = u32(1);
    }
}

fn cell_index(cell: vec2<u32>) -> u32 {
    return cell.y * u32(grid.x) + cell.x;
}
