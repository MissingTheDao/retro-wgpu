//! Stage 1 (core): headless offscreen tests on every backend this machine has. Each test
//! checks pixels analytically and against a golden image the test itself wrote on first run
//! (tests/golden; RETRO_WGPU_BLESS=1 rewrites them).

mod common;
use common::*;
use retro_wgpu::*;

macro_rules! per_backend {
    ($name:ident) => {
        mod $name {
            #[test]
            fn vulkan() {
                super::$name(retro_wgpu::Backend::Vulkan);
            }
            #[test]
            fn gl() {
                super::$name(retro_wgpu::Backend::Gl);
            }
            #[cfg(windows)]
            #[test]
            fn dx12() {
                super::$name(retro_wgpu::Backend::Dx12);
            }
        }
    };
}

const S: (u32, u32) = (64, 64);

fn new(backend: Backend) -> Option<Ctx> {
    ctx(backend, S, SceneSettings::default())
}

fn eight_lights(backend: Backend) {
    let Some(mut c) = new(backend) else { return };
    
    // Layout: Position (F32x3) + Normal (F32x3)
    let l = c.layout(&[
        (Semantic::Position, AttrFormat::F32x3),
        (Semantic::Normal, AttrFormat::F32x3),
    ]);

    // Helper to create a quad facing camera (normal 0,0,1) at z=0.5
    let make_quad = |x0: f32, x1: f32| {
        let mut w = VertexWriter::new();
        for (x, y) in [(x0, -1.0), (x1, -1.0), (x1, 1.0), (x0, 1.0)] {
            w.f32s(&[x, y, 0.5]).f32s(&[0.0, 0.0, 1.0]);
        }
        w.bytes
    };

    let st = RenderState::default();
    let mut lighting = Lighting {
        enabled: true,
        ambient: [0.0, 0.0, 0.0],
        count: 8,
        normalize: true,
        ..Lighting::OFF
    };

    // Configure lights
    // Lights 0-3: Red [0.1, 0.0, 0.0]
    // Lights 4-7: Green [0.0, 0.1, 0.0]
    for k in 0..8 {
        lighting.lights[k] = DirLight {
            direction: [0.0, 0.0, -1.0],
            color: if k < 4 { [0.1, 0.0, 0.0] } else { [0.0, 0.1, 0.0] }
        };
    }

    let material = Material {
        diffuse: [1.0, 1.0, 1.0, 1.0],
        ambient: [0.0, 0.0, 0.0, 1.0],
        emissive: [0.0, 0.0, 0.0, 0.0],
        diffuse_from_vertex: false,
        ambient_from_vertex: false,
        emissive_from_vertex: false,
    };

    let draw_params = DrawParams {
        lighting,
        material,
        ..Default::default()
    };

    // Frame 1: Left (3 lights), Right (8 lights)
    c.r.clear(Some([0.0, 0.0, 0.0, 1.0]), Some(1.0), Some(0));
    c.draw(l, Topology::TriangleFan, &make_quad(-1.0, 0.0), Indices::None, &st, &DrawParams { lighting: Lighting { count: 3, ..lighting }, ..draw_params });
    c.draw(l, Topology::TriangleFan, &make_quad(0.0, 1.0), Indices::None, &st, &draw_params);
    let img1 = c.frame();
    expect_px(&img1, 16, 32, [77, 0, 0, 255], 2, "left 3 lights");
    expect_px(&img1, 48, 32, [102, 102, 0, 255], 2, "right 8 lights");

    // Frame 2: Right (8 lights), Left (3 lights)
    c.r.clear(Some([0.0, 0.0, 0.0, 1.0]), Some(1.0), Some(0));
    c.draw(l, Topology::TriangleFan, &make_quad(0.0, 1.0), Indices::None, &st, &draw_params);
    c.draw(l, Topology::TriangleFan, &make_quad(-1.0, 0.0), Indices::None, &st, &DrawParams { lighting: Lighting { count: 3, ..lighting }, ..draw_params });
    let img2 = c.frame();
    expect_px(&img2, 16, 32, [77, 0, 0, 255], 2, "left 3 lights (2nd frame)");
    expect_px(&img2, 48, 32, [102, 102, 0, 255], 2, "right 8 lights (2nd frame)");
}
per_backend!(eight_lights);

fn flat_vs_gouraud(backend: Backend) {
    let Some(mut c) = new(backend) else { return };

    // Layout: Position (F32x3) + Color (Unorm8x4)
    let l = c.layout(&[
        (Semantic::Position, AttrFormat::F32x3),
        (Semantic::Color, AttrFormat::Unorm8x4),
    ]);

    // Helper to create a triangle with colors
    let make_tri = |x_offset: f32| {
        let mut w = VertexWriter::new();
        // Vertices: (-1,-1), (0,-1), (-1,1) shifted by x_offset
        // First vertex red, second green, third blue
        let verts = [
            ([-1.0 + x_offset, -1.0, 0.5], [255u8, 0, 0, 255]),
            ([0.0 + x_offset, -1.0, 0.5], [0, 255, 0, 255]),
            ([-1.0 + x_offset, 1.0, 0.5], [0, 0, 255, 255]),
        ];
        for (pos, col) in verts {
            w.f32s(&pos).u8x4(col);
        }
        w.bytes
    };

    let st_flat = RenderState { flat_shading: true, ..Default::default() };
    let st_gouraud = RenderState { flat_shading: false, ..Default::default() };
    
    // Unlit material
    let material = Material {
        diffuse: [1.0, 1.0, 1.0, 1.0],
        ambient: [0.0, 0.0, 0.0, 1.0],
        emissive: [0.0, 0.0, 0.0, 0.0],
        diffuse_from_vertex: true,
        ambient_from_vertex: true,
        emissive_from_vertex: false,
    };

    let draw_params = DrawParams {
        material,
        ..Default::default()
    };

    c.r.clear(Some([0.0, 0.0, 0.0, 1.0]), Some(1.0), Some(0));
    // Left triangle (flat shading)
    c.draw(l, Topology::TriangleList, &make_tri(0.0), Indices::None, &st_flat, &draw_params);
    // Right triangle (gouraud shading), shifted right by 1.0 in x
    c.draw(l, Topology::TriangleList, &make_tri(1.0), Indices::None, &st_gouraud, &draw_params);
    
    let img = c.frame();

    // Flat shading: entire triangle is colored by the first vertex (red)
    expect_px(&img, 4, 40, [255, 0, 0, 255], 0, "flat pixel 1");
    expect_px(&img, 20, 56, [255, 0, 0, 255], 0, "flat pixel 2");
    expect_px(&img, 4, 10, [255, 0, 0, 255], 0, "flat pixel 3");

    // Gouraud shading: centroid of right triangle
    // Right triangle vertices: (0,-1), (1,-1), (0,1) in clip space relative to shift?
    // make_tri(1.0) shifts vertices: (-0, -1), (1, -1), (-0, 1).
    // Centroid approx: ( (-0 + 1 + -0)/3, (-1 + -1 + 1)/3 ) = (0.33, -0.33) in clip space.
    // In 64x64 image: x = (0.33 + 1)/2 * 64 = 42.6, y = (-0.33 + 1)/2 * 64 = 32.
    // The prompt asks for pixel (32+8, 40) which is (40, 40).
    // Let's check pixel (40, 40).
    // x=40 -> clip x = (40/64)*2 - 1 = 0.25
    // y=40 -> clip y = -(40/64)*2 + 1 = -0.25
    // Is (0.25, -0.25) inside triangle (-0,-1), (1,-1), (0,1)?
    // Yes.
    // Colors: V0(-0,-1) Red, V1(1,-1) Green, V2(0,1) Blue.
    // Interpolated color should have Green and Blue components > 30.
    let px = img.pixel(40, 40);
    assert!(px[1] > 30 && px[2] > 30, "gouraud pixel (40,40) got {:?}, expected mixed green/blue", px);
}
per_backend!(flat_vs_gouraud);
