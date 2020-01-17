#[macro_use]
extern crate glium;
extern crate cgmath;

use cgmath::SquareMatrix;
#[allow(unused_imports)]
use glium::{glutin, Surface};
use std::time::Instant;
use glutin::dpi::LogicalSize;
use std::collections::HashMap;
use std::io::{self, Write};


#[inline(always)]
fn bit_count(x: u32) -> usize {
/* first let res = x&0xAAAAAAAA >> 1 + x&55555555
 * after that the (2k)th and (2k+1)th bits of the res
 * will be the number of 1s that contained by the (2k)th 
 * and (2k+1)th bits of x
 * we can use a similar way to caculate the number of 1s
 * that contained by the (4k)th and (4k+1)th and (4k+2)th 
 * and (4k+3)th bits of x, so as 8, 16, 32
 */ 
    let mut var_a = (85 << 8) | 85;
    var_a = (var_a << 16) | var_a;
    let mut res = ((x>>1) & var_a) + (x & var_a);

    var_a = (51 << 8) | 51;
    var_a = (var_a << 16) | var_a;
    res = ((res>>2) & var_a) + (res & var_a);

    var_a = (15 << 8) | 15;
    var_a = (var_a << 16) | var_a;
    res = ((res>>4) & var_a) + (res & var_a);

    var_a = (255 << 16) | 255;
    res = ((res>>8) & var_a) + (res & var_a);

    var_a = (255 << 8) | 255;
    res = ((res>>16) & var_a) + (res & var_a);
    return res as usize;
}

fn find_closest_group(groups: &Vec<u64>, group: u64) -> u64 {
    let mut min : u64 = 0;
    let mut min_distance = std::usize::MAX;
    for template in groups {
        let xor = template ^ group;
        let distance = bit_count((xor >> 32) as u32) + bit_count((xor & 0xffffffff) as u32);
        if distance < min_distance {
            min_distance = distance;
            min = *template;
        }
        if distance == 1 { return min }
    }
    min
}

#[inline(always)]
fn grey_scale(rgba: &(u8, u8, u8)) -> usize {
    ((rgba.0 as usize + rgba.1 as usize + rgba.2 as usize) / 3)
}

fn render(raw_slice: &[u8], width: u32, height: u32) {
    let mut transforms = HashMap::new();
    transforms.insert(u64::from_str_radix(&("0".repeat(8*3)+&"1".repeat(8)+&"0".repeat(8*4)).as_str(), 2).unwrap(), (true, "─"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(4)+&"1".repeat(8)+&"0".repeat(8).repeat(3)).as_str(), 2).unwrap(), (true, "─"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(3)+&"1".repeat(8).repeat(2)+&"0".repeat(8).repeat(3)).as_str(), 2).unwrap(), (true, "━"));
    transforms.insert(u64::from_str_radix(&("00010000".repeat(8)).as_str(), 2).unwrap(), (true, "│"));
    transforms.insert(u64::from_str_radix(&("00001000".repeat(8)).as_str(), 2).unwrap(), (true, "│"));
    transforms.insert(u64::from_str_radix(&("00011000".repeat(8)).as_str(), 2).unwrap(), (true, "┃"));

    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(3)+&"0".repeat(8)+&"1".repeat(8).repeat(4)).as_str(), 2).unwrap(), (false, "─"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(4)+&"0".repeat(8)+&"1".repeat(8).repeat(3)).as_str(), 2).unwrap(), (false, "─"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(3)+&"0".repeat(8).repeat(2)+&"1".repeat(8).repeat(3)).as_str(), 2).unwrap(), (false, "━"));
    transforms.insert(u64::from_str_radix(&("11101111".repeat(8)).as_str(), 2).unwrap(), (false, "│"));
    transforms.insert(u64::from_str_radix(&("11110111".repeat(8)).as_str(), 2).unwrap(), (false, "│"));
    transforms.insert(u64::from_str_radix(&("11100111".repeat(8)).as_str(), 2).unwrap(), (false, "┃"));

    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(4)+&"0".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▀"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(7)+&"1".repeat(8)).as_str(), 2).unwrap(), (true, "▁"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(6)+&"1".repeat(8).repeat(2)).as_str(), 2).unwrap(), (true, "▂"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(5)+&"1".repeat(8).repeat(3)).as_str(), 2).unwrap(), (true, "▃"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(4)+&"1".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▄"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(3)+&"1".repeat(8).repeat(5)).as_str(), 2).unwrap(), (true, "▅"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(2)+&"1".repeat(8).repeat(6)).as_str(), 2).unwrap(), (true, "▆"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(1)+&"1".repeat(8).repeat(7)).as_str(), 2).unwrap(), (true, "▇"));

    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(4)+&"1".repeat(8).repeat(4)).as_str(), 2).unwrap(), (false, "▀"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(7)+&"0".repeat(8)).as_str(), 2).unwrap(), (false, "▁"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(6)+&"0".repeat(8).repeat(2)).as_str(), 2).unwrap(), (false, "▂"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(5)+&"0".repeat(8).repeat(3)).as_str(), 2).unwrap(), (false, "▃"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(4)+&"0".repeat(8).repeat(4)).as_str(), 2).unwrap(), (false, "▄"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(3)+&"0".repeat(8).repeat(5)).as_str(), 2).unwrap(), (false, "▅"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(2)+&"0".repeat(8).repeat(6)).as_str(), 2).unwrap(), (false, "▆"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(1)+&"0".repeat(8).repeat(7)).as_str(), 2).unwrap(), (false, "▇"));

    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(8)).as_str(), 2).unwrap(), (true, "█"));
    transforms.insert(u64::from_str_radix(&("11111110".repeat(8)).as_str(), 2).unwrap(), (true, "▉"));
    transforms.insert(u64::from_str_radix(&("11111100".repeat(8)).as_str(), 2).unwrap(), (true, "▊"));
    transforms.insert(u64::from_str_radix(&("11111000".repeat(8)).as_str(), 2).unwrap(), (true, "▋"));
    transforms.insert(u64::from_str_radix(&("11110000".repeat(8)).as_str(), 2).unwrap(), (true, "▌"));
    transforms.insert(u64::from_str_radix(&("11100000".repeat(8)).as_str(), 2).unwrap(), (true, "▍"));
    transforms.insert(u64::from_str_radix(&("11000000".repeat(8)).as_str(), 2).unwrap(), (true, "▎"));
    transforms.insert(u64::from_str_radix(&("10000000".repeat(8)).as_str(), 2).unwrap(), (true, "▏"));
    transforms.insert(u64::from_str_radix(&("00001111".repeat(8)).as_str(), 2).unwrap(), (true, "▐"));
    transforms.insert(u64::from_str_radix(&(("1000100000100010").repeat(4)).as_str(), 2).unwrap(), (true, "░"));
    transforms.insert(u64::from_str_radix(&(("1010101001010100").repeat(4)).as_str(), 2).unwrap(), (true, "▒"));
    transforms.insert(u64::from_str_radix(&(("0111011111011101").repeat(4)).as_str(), 2).unwrap(), (true, "▓"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8)+&"0".repeat(8).repeat(7)).as_str(), 2).unwrap(), (true, "▔"));
    transforms.insert(u64::from_str_radix(&("00000001".repeat(8)).as_str(), 2).unwrap(), (true, "▕"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(4)+&"11110000".repeat(4)).as_str(), 2).unwrap(), (true, "▖"));
    transforms.insert(u64::from_str_radix(&("0".repeat(8).repeat(4)+&"00001111".repeat(4)).as_str(), 2).unwrap(), (true, "▗"));
    transforms.insert(u64::from_str_radix(&("11110000".repeat(4)+&"0".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▘"));
    transforms.insert(u64::from_str_radix(&("11110000".repeat(4)+&"1".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▙"));
    transforms.insert(u64::from_str_radix(&("11110000".repeat(4)+&"00001111".repeat(4)).as_str(), 2).unwrap(), (true, "▚"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(4)+&"11110000".repeat(4)).as_str(), 2).unwrap(), (true, "▛"));
    transforms.insert(u64::from_str_radix(&("1".repeat(8).repeat(4)+&"00001111".repeat(4)).as_str(), 2).unwrap(), (true, "▜"));
    transforms.insert(u64::from_str_radix(&("00001111".repeat(4)+&"0".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▝"));
    transforms.insert(u64::from_str_radix(&("00001111".repeat(4)+&"11110000".repeat(4)).as_str(), 2).unwrap(), (true, "▞"));
    transforms.insert(u64::from_str_radix(&("00001111".repeat(4)+&"1".repeat(8).repeat(4)).as_str(), 2).unwrap(), (true, "▟"));

    let mut transforms_keys : Vec<u64> = Vec::new();
    for k in transforms.keys() {
        transforms_keys.push(*k);
    }
    const AVERAGE_SIZE: usize = 8;
    let mut sorted: [(usize, usize, (u8, u8, u8)); AVERAGE_SIZE] = [(0, 0, (0, 0, 0)); AVERAGE_SIZE];
    let mut grey_scales_start: [usize; 32] = [0; 32];
    let mut grey_scales_end: [usize; 32] = [0; 32];
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    handle.write_all(format!("\x1b[{};0f", 0).as_bytes()).unwrap();
    for y in 0..(height / 16) {
        let mut x = 0;
        while x < (width / 8) {
           let mut sum_grey_scale : usize = 0;
           let mut i = 0;
           let mut dy : usize = 0;
           let mut dx : usize;
           while dy < 8 {
               dx = 0;
               while dx < 8 {
                   let _x = x * 8 + (dx as u32);
                   let _y = y * 16 + (dy as u32) * 2;
                   let start = (((height - 1 - _y) * width + _x)*4) as usize;
                   let block = (raw_slice[start], raw_slice[start + 1], raw_slice[start + 2]);
                   let grey = grey_scale(&block);
                   /* do not write every pixel in sorted so that sort_by is faster
                    * the downside is that this reduce quality a lot */
                   if i % AVERAGE_SIZE == dy { sorted[dy] = (grey, i, block) };
                   if i < 32 {
                       grey_scales_start[i] = grey;
                   }
                   else {
                       grey_scales_end[i - 32] = grey;
                   }
                   sum_grey_scale += grey;
                   i += 1;
                   dx += 1;
               }
               dy += 1
           }
           let average_grey_scale : usize = sum_grey_scale / 64;
           sorted.sort_unstable_by(|a, b| a.0.cmp(&b.0));
           let average_min = sorted[AVERAGE_SIZE / 4].2;
           let average_max = sorted[(3 * AVERAGE_SIZE) / 4].2;
           let mut group = 0;
           for grey in &grey_scales_start {
               group = group << 1 | (if grey >= &average_grey_scale { 1 } else { 0 });
           }
           for grey in &grey_scales_end {
               group = group << 1 | (if grey >= &average_grey_scale { 1 } else { 0 });
           }
           let transform = match transforms.get(&group) {
               Some(t) => t,
               _ => {
                   let closest = find_closest_group(&transforms_keys, group);
                   match transforms.get(&closest) {
                       Some(x) => {
                           x
                       },
                       _ => &(true, " ")
                   }
                }
           }.clone();
           let fg = if transform.0 { average_max } else { average_min };
           let bg = if transform.0 { average_min } else { average_max };
           let result = transform.1;
           handle.write_all(format!("\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m{}", fg.0, fg.1, fg.2, bg.0, bg.1, bg.2, result).as_bytes()).unwrap();
           x += 1;
        }
        handle.write_all(b"\x1b[0m\n").unwrap();
    }
}

fn main() {
    let win_size = LogicalSize {
        width: 30.0,
        height: 30.0,
    };
    let shadow_map_size = 1024;

    // Create the main window
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(win_size)
        .with_title("ThisWindowMustFloat");
    let cb = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    // Create the boxes to render in the scene
    let (model_vertex_buffer, model_index_buffer) = create_box(&display);
    let mut model_data = [
        ModelData::color([0.4, 0.4, 0.4]).translate([0.0, -2.5, 0.0]).scale(5.0),
        ModelData::color([0.6, 0.1, 0.1]).translate([0.0, 0.252, 0.0]).scale(0.5),
        ModelData::color([0.1, 0.6, 0.1]).translate([0.9, 0.5, 0.1]).scale(0.5),
        ModelData::color([0.1, 0.1, 0.6]).translate([-0.8, 0.75, 0.1]).scale(0.5),
    ];

    let shadow_map_shaders = glium::Program::from_source(
        &display,
        // Vertex Shader
        "
            #version 330 core
            in vec4 position;
            uniform mat4 depth_mvp;
            void main() {
              gl_Position = depth_mvp * position;
            }
        ",
        // Fragement Shader
        "
            #version 330 core
            layout(location = 0) out float fragmentdepth;
            void main(){
                fragmentdepth = gl_FragCoord.z;
            }
        ",
        None).unwrap();

    let render_shaders = glium::Program::from_source(
        &display,
        // Vertex Shader
        "
            #version 330 core

            uniform mat4 mvp;
            uniform mat4 depth_bias_mvp;
            uniform mat4 model_matrix;
            uniform vec4 model_color;

            in vec4 position;
            in vec4 normal;

            out vec4 shadow_coord;
            out vec4 model_normal;

            void main() {
            	gl_Position =  mvp * position;
            	model_normal = model_matrix * normal;
            	shadow_coord = depth_bias_mvp * position;
            }
        ",
        // Fragement Shader
        "
            #version 330 core

            uniform sampler2DShadow shadow_map;
            uniform vec3 light_loc;
            uniform vec4 model_color;

            in vec4 shadow_coord;
            in vec4 model_normal;

            out vec4 color;

            void main() {
                vec3 light_color = vec3(1,1,1);
            	float bias = 0.0; // Geometry does not require bias

            	float lum = max(dot(normalize(model_normal.xyz), normalize(light_loc)), 0.0);

            	float visibility = texture(shadow_map, vec3(shadow_coord.xy, (shadow_coord.z-bias)/shadow_coord.w));

            	color = vec4(max(lum * visibility, 0.05) * model_color.rgb * light_color, 1.0);
            }
        ",
        None).unwrap();

    // Debug Resources (for displaying shadow map)
    let debug_vertex_buffer = glium::VertexBuffer::new(
        &display,
        &[
            DebugVertex::new([0.25, -1.0], [0.0, 0.0]),
            DebugVertex::new([0.25, -0.25], [0.0, 1.0]),
            DebugVertex::new([1.0, -0.25], [1.0, 1.0]),
            DebugVertex::new([1.0, -1.0], [1.0, 0.0]),
        ],
    ).unwrap();
    let debug_index_buffer = glium::IndexBuffer::new(
        &display,
        glium::index::PrimitiveType::TrianglesList,
        &[0u16, 1, 2, 0, 2, 3],
    ).unwrap();
    let debug_shadow_map_shaders = glium::Program::from_source(
        &display,
        // Vertex Shader
        "
			#version 140
			in vec2 position;
			in vec2 tex_coords;
			out vec2 v_tex_coords;
			void main() {
				gl_Position = vec4(position, 0.0, 1.0);
				v_tex_coords = tex_coords;
			}
        ",
        // Fragement Shader
        "
			#version 140
			uniform sampler2D tex;
			in vec2 v_tex_coords;
			out vec4 f_color;
			void main() {
				f_color = vec4(texture(tex, v_tex_coords).rgb, 1.0);
			}
        ",
        None).unwrap();

    let shadow_texture = glium::texture::DepthTexture2d::empty(&display, shadow_map_size, shadow_map_size).unwrap();

    let mut start = Instant::now();

    let mut light_t: f64 = 8.7;
    let mut light_rotating = false;
    let mut camera_t: f64 = 8.22;
    let mut camera_rotating = false;

    println!("This example demonstrates real-time shadow mapping. Press C to toggle camera");
    println!("rotation; press L to toggle light rotation.");

    event_loop.run(move |event, _, control_flow| {
        let elapsed_dur = start.elapsed();
        let secs = (elapsed_dur.as_secs() as f64) + (elapsed_dur.subsec_nanos() as f64) * 1e-9;
        start = Instant::now();

        if camera_rotating { camera_t += secs * 0.7; }
        if light_rotating { light_t += secs * 0.7; }

        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        // Handle events
        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { input, .. } => if input.state == glutin::event::ElementState::Pressed {
                    if let Some(key) = input.virtual_keycode {
                        match key {
                            glutin::event::VirtualKeyCode::C => camera_rotating = !camera_rotating,
                            glutin::event::VirtualKeyCode::L => light_rotating = !light_rotating,
                            _ => {}
                        }
                    }
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        // Rotate the light around the center of the scene
        let light_loc = {
            let x = 3.0 * light_t.cos();
            let z = 3.0 * light_t.sin();
            [x as f32, 5.0, z as f32]
        };

        // Render the scene from the light's point of view into depth buffer
        // ===============================================================================
        {
            // Orthographic projection used to demostrate a far-away light source
			let w = 4.0;
            let depth_projection_matrix: cgmath::Matrix4<f32> = cgmath::ortho(-w, w, -w, w, -10.0, 20.0);
            let view_center: cgmath::Point3<f32> = cgmath::Point3::new(0.0, 0.0, 0.0);
            let view_up: cgmath::Vector3<f32> = cgmath::Vector3::new(0.0, 1.0, 0.0);
            let depth_view_matrix = cgmath::Matrix4::look_at(light_loc.into(), view_center, view_up);

            let mut draw_params: glium::draw_parameters::DrawParameters = Default::default();
            draw_params.depth = glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLessOrEqual,
                write: true,
                ..Default::default()
            };
            draw_params.backface_culling = glium::BackfaceCullingMode::CullClockwise;

            // Write depth to shadow map texture
            let mut target = glium::framebuffer::SimpleFrameBuffer::depth_only(&display, &shadow_texture).unwrap();
            target.clear_color(1.0, 1.0, 1.0, 1.0);
            target.clear_depth(1.0);

            // Draw each model
            for md in &mut model_data {
                let depth_mvp = depth_projection_matrix * depth_view_matrix * md.model_matrix;
                md.depth_mvp = depth_mvp;

                let uniforms = uniform! {
                    depth_mvp: Into::<[[f32; 4]; 4]>::into(md.depth_mvp),
                };

                target.draw(
                    &model_vertex_buffer,
                    &model_index_buffer,
                    &shadow_map_shaders,
                    &uniforms,
                    &draw_params,
                ).unwrap();
            }
        }

        // Render the scene from the camera's point of view
        // ===============================================================================
        let screen_ratio = (win_size.width / win_size.height) as f32;
        let perspective_matrix: cgmath::Matrix4<f32> = cgmath::perspective(cgmath::Deg(45.0), screen_ratio, 0.0001, 100.0);
        let camera_x = 3.0 * camera_t.cos();
        let camera_z = 3.0 * camera_t.sin();
        let view_eye: cgmath::Point3<f32> = cgmath::Point3::new(camera_x as f32, 2.0, camera_z as f32);
        let view_center: cgmath::Point3<f32> = cgmath::Point3::new(0.0, 0.0, 0.0);
        let view_up: cgmath::Vector3<f32> = cgmath::Vector3::new(0.0, 1.0, 0.0);
        let view_matrix: cgmath::Matrix4<f32> = cgmath::Matrix4::look_at(view_eye, view_center, view_up);

        let bias_matrix: cgmath::Matrix4<f32> = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.5, 0.5, 0.5, 1.0],
        ].into();

        let mut draw_params: glium::draw_parameters::DrawParameters = Default::default();
        draw_params.depth = glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLessOrEqual,
            write: true,
            ..Default::default()
        };
        draw_params.backface_culling = glium::BackfaceCullingMode::CullCounterClockwise;
        draw_params.blend = glium::Blend::alpha_blending();

        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 0.0), 1.0);

        // Draw each model
        for md in &model_data {
            let mvp = perspective_matrix * view_matrix * md.model_matrix;
            let depth_bias_mvp = bias_matrix * md.depth_mvp;

            let uniforms = uniform! {
                light_loc: light_loc,
                perspective_matrix: Into::<[[f32; 4]; 4]>::into(perspective_matrix),
                view_matrix: Into::<[[f32; 4]; 4]>::into(view_matrix),
                model_matrix: Into::<[[f32; 4]; 4]>::into(md.model_matrix),
                model_color: md.color,

                mvp: Into::<[[f32;4];4]>::into(mvp),
                depth_bias_mvp: Into::<[[f32;4];4]>::into(depth_bias_mvp),
                shadow_map: glium::uniforms::Sampler::new(&shadow_texture)
					.magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
					.minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
                    .depth_texture_comparison(Some(glium::uniforms::DepthTextureComparison::LessOrEqual)),
            };

            target.draw(
                &model_vertex_buffer,
                &model_index_buffer,
                &render_shaders,
                &uniforms,
                &draw_params,
            ).unwrap();
        }

        {
            let uniforms = uniform! {
                tex: glium::uniforms::Sampler::new(&shadow_texture)
                    .magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
                    .minify_filter(glium::uniforms::MinifySamplerFilter::Nearest)
            };
            target.clear_depth(1.0);
            target
                .draw(
                    &debug_vertex_buffer,
                    &debug_index_buffer,
                    &debug_shadow_map_shaders,
                    &uniforms,
                    &Default::default(),
                )
                .unwrap();
        }

        target.finish().unwrap();
        let image: glium::texture::RawImage2d<u8> = display.read_front_buffer().unwrap();
        //let image = image::ImageBuffer::from_raw(image.width, image.height, image.data.into_owned()).unwrap();
        let raw_slice = &image.data[..];
        render(raw_slice, image.width, image.height);
    });
}

fn create_box(display: &glium::Display) -> (glium::VertexBuffer<Vertex>, glium::IndexBuffer<u16>) {
    let box_vertex_buffer = glium::VertexBuffer::new(display, &[
        // Max X
        Vertex { position: [ 0.5,-0.5,-0.5, 1.0], normal: [ 1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [ 0.5,-0.5, 0.5, 1.0], normal: [ 1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [ 0.5, 0.5, 0.5, 1.0], normal: [ 1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [ 0.5, 0.5,-0.5, 1.0], normal: [ 1.0, 0.0, 0.0, 0.0] },
        // Min X
        Vertex { position: [-0.5,-0.5,-0.5, 1.0], normal: [-1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [-0.5, 0.5,-0.5, 1.0], normal: [-1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5, 1.0], normal: [-1.0, 0.0, 0.0, 0.0] },
        Vertex { position: [-0.5,-0.5, 0.5, 1.0], normal: [-1.0, 0.0, 0.0, 0.0] },
        // Max Y
        Vertex { position: [-0.5, 0.5,-0.5, 1.0], normal: [ 0.0, 1.0, 0.0, 0.0] },
        Vertex { position: [ 0.5, 0.5,-0.5, 1.0], normal: [ 0.0, 1.0, 0.0, 0.0] },
        Vertex { position: [ 0.5, 0.5, 0.5, 1.0], normal: [ 0.0, 1.0, 0.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5, 1.0], normal: [ 0.0, 1.0, 0.0, 0.0] },
        // Min Y
        Vertex { position: [-0.5,-0.5,-0.5, 1.0], normal: [ 0.0,-1.0, 0.0, 0.0] },
        Vertex { position: [-0.5,-0.5, 0.5, 1.0], normal: [ 0.0,-1.0, 0.0, 0.0] },
        Vertex { position: [ 0.5,-0.5, 0.5, 1.0], normal: [ 0.0,-1.0, 0.0, 0.0] },
        Vertex { position: [ 0.5,-0.5,-0.5, 1.0], normal: [ 0.0,-1.0, 0.0, 0.0] },
        // Max Z
        Vertex { position: [-0.5,-0.5, 0.5, 1.0], normal: [ 0.0, 0.0, 1.0, 0.0] },
        Vertex { position: [-0.5, 0.5, 0.5, 1.0], normal: [ 0.0, 0.0, 1.0, 0.0] },
        Vertex { position: [ 0.5, 0.5, 0.5, 1.0], normal: [ 0.0, 0.0, 1.0, 0.0] },
        Vertex { position: [ 0.5,-0.5, 0.5, 1.0], normal: [ 0.0, 0.0, 1.0, 0.0] },
        // Min Z
        Vertex { position: [-0.5,-0.5,-0.5, 1.0], normal: [ 0.0, 0.0,-1.0, 0.0] },
        Vertex { position: [ 0.5,-0.5,-0.5, 1.0], normal: [ 0.0, 0.0,-1.0, 0.0] },
        Vertex { position: [ 0.5, 0.5,-0.5, 1.0], normal: [ 0.0, 0.0,-1.0, 0.0] },
        Vertex { position: [-0.5, 0.5,-0.5, 1.0], normal: [ 0.0, 0.0,-1.0, 0.0] },
        ]).unwrap();

    let mut indexes = Vec::new();
    for face in 0..6u16 {
        indexes.push(4 * face + 0);
        indexes.push(4 * face + 1);
        indexes.push(4 * face + 2);
        indexes.push(4 * face + 0);
        indexes.push(4 * face + 2);
        indexes.push(4 * face + 3);
    }
    let box_index_buffer = glium::IndexBuffer::new(display, glium::index::PrimitiveType::TrianglesList, &indexes).unwrap();
    (box_vertex_buffer, box_index_buffer)
}

#[derive(Clone, Copy, Debug)]
struct Vertex {
    position: [f32; 4],
    normal: [f32; 4],
}
implement_vertex!(Vertex, position, normal);

#[derive(Clone, Debug)]
struct ModelData {
    model_matrix: cgmath::Matrix4<f32>,
    depth_mvp: cgmath::Matrix4<f32>,
    color: [f32; 4],
}
impl ModelData {
    pub fn color(c: [f32; 3]) -> Self {
        Self {
            model_matrix: cgmath::Matrix4::identity(),
            depth_mvp: cgmath::Matrix4::identity(),
            color: [c[0], c[1], c[2], 1.0],
        }
    }
    pub fn scale(mut self, s: f32) -> Self {
        self.model_matrix = self.model_matrix * cgmath::Matrix4::from_scale(s);
        self
    }
    pub fn translate(mut self, t: [f32; 3]) -> Self {
        self.model_matrix = self.model_matrix * cgmath::Matrix4::from_translation(t.into());
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct DebugVertex {
    position: [f32; 2],
	tex_coords: [f32; 2],
}
implement_vertex!(DebugVertex, position, tex_coords);
impl DebugVertex {
    pub fn new(position: [f32; 2], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            tex_coords,
        }
    }
}
