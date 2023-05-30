@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> state: array<u32>;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @builtin(instance_index) instance: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) cell: vec2<f32>,
}

struct FragmentInput {
  @location(0) cell: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let i = f32(input.instance);
    let cell = vec2<f32>(i % grid.x, floor(i / grid.x));

    var output: VertexOutput;
    output.position = vec4<f32>(
        (f32(state[input.instance]) * input.position + 1.0) / grid - 1.0 + (cell / grid * 2.0),
        0.0,
        1.0,
    );
    output.cell = cell;

    return output;

}

@fragment
fn fragment_main(input: FragmentInput) -> @location(0) vec4<f32> {
    let c = input.cell / grid;
    return vec4<f32>(c, 1.0 - c.x, 1.0);
}
