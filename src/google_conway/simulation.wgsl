@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> in: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute
@workgroup_size(8, 8)
fn compute_main(@builtin(global_invocation_id) cell: vec3<u32>) {
    let active_neighbors =
        cell_active(cell.x + 1u, cell.y + 1u) +
        cell_active(cell.x + 1u, cell.y     ) +
        cell_active(cell.x + 1u, cell.y - 1u) +
        cell_active(cell.x     , cell.y - 1u) +
        cell_active(cell.x - 1u, cell.y - 1u) +
        cell_active(cell.x - 1u, cell.y     ) +
        cell_active(cell.x - 1u, cell.y + 1u) +
        cell_active(cell.x     , cell.y + 1u);

    let i = cell_index(cell.xy);

    switch active_neighbors {
        case 2u: {
            out[i] = in[i];
        }
        case 3u: {
            out[i] = 1u;
        }
        default: {
            out[i] = 0u;
        }
    }
}

fn cell_index(cell: vec2<u32>) -> u32 {
    return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
}

fn cell_active(x: u32, y: u32) -> u32 {
    return in[cell_index(vec2(x, y))];
}
