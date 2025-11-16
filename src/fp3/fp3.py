import ctypes
import glm
import moderngl
import numpy
import numpy as np
import pygame

from loadModelUsingAssimp_V3 import create3DAssimpObject
from math import cos, sin, sqrt
from pathlib import Path

# ctypes.windll.user32.SetProcessDPIAware()

# ==================================================================================================
# USEFUL CONSTANTS
# ==================================================================================================

UP = glm.vec3(0, 1, 0)
FLOOR_SCALE = 3  # Factor to multiply the model bound's radius by

# ==================================================================================================
# PYGAME SETUP
# ==================================================================================================

screen_width = 500
screen_height = 500

pygame.init()  # Initlizes its different modules. Display module is one of them.

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)
pygame.display.set_mode(
    (screen_width, screen_height),
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
)
pygame.display.set_caption(title="Project Substitute 3: Rafael Niebles")

gl = moderngl.get_context()

main_fb = (
    gl.detect_framebuffer()
)  # Get the framebuffer so we can perform off-screen rendering

# ==================================================================================================
# SHADERS SETUP
# ==================================================================================================

# NOTE: The same shader is used by both floor and model!

VERTEX_SHADER = """
#version 410 core

// =================================================================================================
// INPUT
// =================================================================================================

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// =================================================================================================
// OUTPUT
// =================================================================================================

out vec2 f_uv; // Same as in
out vec3 f_normal; // Normal in world coordinates (and normalized to 0-1)
out vec3 f_position; // Position in world coordinates

// =================================================================================================
// UNIFORM
// =================================================================================================

uniform mat4 model; // Converts from model -> world
uniform mat4 view; // Converts from world -> camera
uniform mat4 perspective; // Applies depth

// =================================================================================================
// MAIN
// =================================================================================================

void main(){
  // Model -> world
  vec4 world_pos = model * vec4(position, 1);
  f_position = world_pos.xyz;

  // UV stays same
  f_uv = uv;

  // Compute normal
  mat3 normal_matrix = mat3(transpose(inverse(model)));
  f_normal = normalize(normal_matrix * normal);

  // Pos from world -> camera; apply perspective  
  gl_Position = perspective * view * world_pos;
} 
"""

FRAGMENT_SHADER = """
#version 410 core

// =================================================================================================
// CONSTANTS
// =================================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =================================================================================================
// INPUT
// =================================================================================================

in vec2 f_uv; // UV
in vec3 f_normal; // Normalized and in world coords
in vec3 f_position; // In world coords

// =================================================================================================
// OUTPUT
// =================================================================================================

out vec4 out_color;

// =================================================================================================
// UNIFORM
// =================================================================================================

uniform float shininess;
uniform vec3 eye_position;
uniform vec3 k_diffuse; // Diffuse coefficient; how much incoming light should scatter? 
uniform vec3 light;
uniform sampler2D map;

uniform bool color_flag; // Render with color?

/* 2 PASS RELATED */

uniform int pcf; // Percentage closer filtering
uniform bool bias_flag; // Use shadow bias
uniform sampler2D shadow_map; // Sampler for shadow map
uniform mat4 light_cam_view; // Matrices for light cam
uniform mat4 light_cam_persp;

// =================================================================================================
// AUX FUNCTIONS
// =================================================================================================

float computeVisibilityFactor(){
    // Convert fragment pos to light space
    vec4 light_space_pos = light_cam_persp * light_cam_view * vec4(f_position, 1.0);

    // Divide by perspective for normalized device coords (NDC) which are coords ranging from -1 to 1
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w; 

    // Convert range from [-1, 1] to [0, 1] (no negatives)
    // This is so that we can use these coords for texture sampling 
    proj_coords = proj_coords * 0.5 + 0.5;

    // Check if fragment goes off bounds
    if (proj_coords.z > 1.0 || 
        proj_coords.x < 0.0 || proj_coords.x > 1.0 || 
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0; // Anything outside the shadowmap is fully lit
    }

    // Get depth of current fragment from light's perspective
    float current_depth = proj_coords.z;

    // Add bias
    // Bias pulls fragment slightly toward light before depth comparison
    // This avoids self-shadowing
    float bias = bias_flag ? 0.005 : 0.0;

    // PCF calculation
    if (pcf > 0){
        float shadow = 0;
        vec2 texel_size = 1.0 / textureSize(shadow_map, 0);

        // Sample surrounding texels based on PCF value
        for (int x = -pcf; x <= pcf; ++x){
            for (int y = -pcf; y <= pcf; ++y){
                float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size).r;
                shadow += (current_depth - bias) > pcf_depth ? 0.0 : 1.0;
            }
        }

        // Average samples to get fractional visibility
        int samples = (2 * pcf + 1) * (2 * pcf + 1);
    
        return shadow / float(samples);
    } else {
        // Simple shadow test with no PCF
        float closest_depth = texture(shadow_map, proj_coords.xy).r;
        return (current_depth - bias) > closest_depth ? 0.0 : 1.0;
    }
}

vec3 computeColor(){
  vec3 light_vector = normalize(light.xyz - f_position); // Represents direction to the light source; assumes point light 
  vec3 material_color = texture(map, f_uv).rgb; // Sample the texture
  vec3 normal = normalize(f_normal); // Renormalize for safety

  // Initialize color to black
  vec3 color = vec3(0);

  float w = dot(normal, UP); // Quantifies the direction of the normal relative to the global up 

  // Compute ambient color
  vec3 ambient_color = 0.25 * (w * SKY_COLOR + (1 - w) * GROUND_COLOR) * material_color;

  // Get fractional light visibility
  float fractional_light_vis = computeVisibilityFactor();

  // If the light's intensity is greater than 0 
  float n_dot_l = dot(normal, light_vector); // Quantifies direction of light relative to normal; encodes the light's intensity
  if (n_dot_l > 0){
    vec3 diffuse_color = material_color * n_dot_l; // Base color with diffuse reflection applied to it
    vec3 eye_vector = normalize(eye_position - f_position); // Points from surface vertex to camera
    vec3 halfway = normalize(light_vector + eye_vector); // Halfway between light and camera
    
    // Initialize specular to 0; we'll determine it in a sec
    vec3 specular_color = vec3(0);
    
    if (shininess > 0){
      specular_color = vec3(1) * pow(dot(normal, halfway), shininess);
    }

    // Compute final color
    color = fractional_light_vis * (k_diffuse * diffuse_color + specular_color);
  }

  // Apply ambient color
  color += ambient_color;

  return color;
}

// =================================================================================================
// MAIN
// =================================================================================================

void main(){
    if (color_flag){
        out_color = vec4(computeColor(), 1);
    } else {
        out_color = vec4(1, 1, 1, 1);
    }
}
"""


shader_program = gl.program(
    vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
)


def query_program(program):
    """
    Displays all uniform variables and their current values in a program.
    """

    for uniform_name in program:
        uniform_value = program[uniform_name]
        print(uniform_name, type(uniform_value), uniform_value)


# ==================================================================================================
# MODEL SETUP
# ==================================================================================================

MODEL_FILEPATH = Path("./mario_obj/scene.gltf")

# Load
model = create3DAssimpObject(
    MODEL_FILEPATH.as_posix(), verbose=False, textureFlag=True, normalFlag=True
)

# Renderable + sampler
model.createRenderableAndSampler(shader_program)

# Get bounds & derived
model_bounds = model.bound
min_corner = model_bounds.boundingBox[0]
max_corner = glm.vec3(
    model_bounds.boundingBox[1].x, min_corner.y, model_bounds.boundingBox[1].z
)
ctr = (min_corner + max_corner) / 2


def render_model():
    """
    Renders the model.
    """

    model.render()


# ==================================================================================================
# FLOOR TEXTURE SETUP
# ==================================================================================================

FLOOR_TEXTURE_PATH = Path("./tile-squares-texture.jpg")

floor_texture_img = pygame.image.load(FLOOR_TEXTURE_PATH.as_posix())
floor_texture_data = pygame.image.tobytes(
    floor_texture_img, "RGB", True
)  # Flip to match OpenGL coords
floor_texture = gl.texture(
    floor_texture_img.get_size(), data=floor_texture_data, components=3
)
floor_texture.build_mipmaps()

floor_sampler = gl.sampler(
    texture=floor_texture,
    filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR),
    repeat_x=True,
    repeat_y=True,
)

# ==================================================================================================
# FLOOR GEOMETRY SETUP
# ==================================================================================================

floor_point = ctr
floor_normal = UP  # Floor plane

floor_side_length = FLOOR_SCALE * model_bounds.radius
floor_midlength = floor_side_length / 2

# fmt: off
floor_vertices = np.array([
  ctr.x - floor_midlength, ctr.y, ctr.z - floor_midlength, 0, 1, 0, 0, 0,
  ctr.x + floor_midlength, ctr.y, ctr.z - floor_midlength, 0, 1, 0, 1, 0,
  ctr.x + floor_midlength, ctr.y, ctr.z + floor_midlength, 0, 1, 0, 1, 1,
  ctr.x - floor_midlength, ctr.y, ctr.z + floor_midlength, 0, 1, 0, 0, 1
]).astype(np.float32)
# fmt: on

floor_vbo = gl.buffer(floor_vertices)

# fmt: off
floor_indices = numpy.array([
  0, 1, 2, 
  2, 3, 0
]).astype(np.int32)
# fmt: on

floor_ibo = gl.buffer(floor_indices)

# ==================================================================================================
# FLOOR RENDERING SETUP
# ==================================================================================================

floor_renderer = gl.vertex_array(
    shader_program,
    [(floor_vbo, "3f 3f 2f", "position", "normal", "uv")],
    floor_ibo,
    index_element_size=4,
)


def render_floor():
    """
    Renders the floor.
    """

    floor_sampler.use(0)

    shader_program["model"].write(
        glm.mat4(1)
    )  # We don't need to transform; we set up the geometry of floor to be in world space already

    floor_renderer.render()


# ==================================================================================================
# SCENE RENDERING SETUP
# ==================================================================================================


def render_scene(view, perspective, light_cam_view, light_cam_perspective, light, eye):
    """
    Renders the scene.
    """

    shader_program["view"].write(view)
    shader_program["perspective"].write(perspective)

    shader_program["light_cam_view"].write(light_cam_view)
    shader_program["light_cam_persp"].write(light_cam_perspective)

    shader_program["light"].write(light)

    shader_program["eye_position"].write(eye)

    # Shadow map sampler
    shadow_map_sampler.use(1)
    shader_program["shadow_map"].value = 1

    # 2 pass specific params
    shader_program["bias_flag"].value = False
    shader_program["pcf"] = 0

    # SHADOW PASS ----------------------------------------------------------------------------------

    shadow_fb.use()

    shader_program["color_flag"].value = False

    render_model()
    render_floor()

    # MAIN PASS ------------------------------------------------------------------------------------

    main_fb.use()

    shader_program["color_flag"].value = True

    render_model()

    shader_program["shininess"].value = 0  # Floor should not be shiny

    render_floor()


def show_shadow_map():
    """
    Displays the shadowmap in a separate viewport.
    """

    shadowmap_viewport_size = screen_width / 4

    gl.viewport = (
        screen_width - shadowmap_viewport_size,
        screen_height - shadowmap_viewport_size,
        shadowmap_viewport_size,
        shadowmap_viewport_size,
    )

    gl.clear(color=(0.5, 0.5, 0.5), viewport=gl.viewport)

    # TODO: Make render call to show shadowmap on this new viewport

    gl.viewport = 0, 0, screen_width, screen_height


# ==================================================================================================
# SHADOW MAPPING SETUP
# ==================================================================================================


DEPTH_TEXTURE_SIZE = (2048, 2048)

# Create depth texture
depth_texture = gl.depth_texture(DEPTH_TEXTURE_SIZE)

# Make a framebuffer for shadows
shadow_fb = gl.framebuffer(depth_attachment=depth_texture)

# Make a depth texture sampler
shadow_map_sampler = gl.sampler(
    texture=depth_texture,
    filter=(gl.NEAREST, gl.NEAREST),  # No filtering for depth
)

# ==================================================================================================
# CAMERA SETUP
# ==================================================================================================

# General

aspect = screen_width / screen_height

# Lighting

light_displacement = 4 * model_bounds.radius * glm.rotateZ(UP, glm.radians(45))

# Main camera

main_cam_displacement = 4 * model_bounds.radius * glm.rotateX(UP, glm.radians(85))

main_cam_target = glm.vec3(model_bounds.center)

main_cam_near = model_bounds.radius
main_cam_far = 20 * model_bounds.radius

main_cam_fov = glm.radians(30)

main_cam_persp_matrix = glm.perspective(
    main_cam_fov, aspect, main_cam_near, main_cam_far
)

# Light camera

light_cam_near = main_cam_near
light_cam_far = 10 * model_bounds.radius

light_cam_fov = glm.radians(60)

light_cam_persp_matrix = glm.perspective(
    light_cam_fov, aspect, light_cam_near, light_cam_far
)

# ==================================================================================================
# RENDER LOOP
# ==================================================================================================

TARGET_FPS = 60

clock = pygame.time.Clock()

world_rotation = 0
light_angle = 0

pcf = 0

gl.enable(gl.DEPTH_TEST)
gl.depth_func = "<="

debug_mode = False
use_bias = True
is_paused = True
is_running = True

while is_running:

    # ----------------------------------------------------------------------------------------------
    # EVENTS
    # ----------------------------------------------------------------------------------------------

    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                is_running = False
            case pygame.KEYDOWN:
                match event.key:
                    case pygame.K_ESCAPE:
                        is_running = False
                    case pygame.K_p:
                        is_paused = not is_paused
                    case pygame.K_d:
                        debug_mode = not debug_mode
                    case pygame.K_b:
                        use_bias = not use_bias
                    case pygame.K_LEFT:
                        light_angle -= 5
                    case pygame.K_RIGHT:
                        light_angle += 5
            case pygame.WINDOWRESIZED:
                screen_width = event.x
                screen_height = event.y

                # Recompute perspective on resize
                main_cam_persp_matrix = glm.perspective(
                    main_cam_fov,
                    screen_width / screen_height,
                    main_cam_near,
                    main_cam_far,
                )

                light_cam_persp_matrix = glm.perspective(
                    light_cam_fov,
                    screen_width / screen_height,
                    light_cam_near,
                    light_cam_far,
                )

    # ----------------------------------------------------------------------------------------------
    # UPDATE
    # ----------------------------------------------------------------------------------------------

    # Apply global rotations (light + camera rotation) to the displacements
    final_light_disp = glm.rotateY(light_displacement, glm.radians(light_angle))

    final_main_cam_disp = glm.rotateY(
        main_cam_displacement, glm.radians(world_rotation)
    )

    # Get final points
    light_point = main_cam_target + final_light_disp

    main_cam_point = main_cam_target + final_main_cam_disp

    # Compute view matrices for cameras
    main_cam_view_matrix = glm.lookAt(main_cam_point, main_cam_target, UP)
    light_cam_view_matrix = glm.lookAt(light_point, main_cam_target, UP)

    dt = clock.tick(TARGET_FPS)
    if not is_paused:
        world_rotation += 1
        if world_rotation >= 360:
            world_rotation = 0

    # ----------------------------------------------------------------------------------------------
    # RENDER
    # ----------------------------------------------------------------------------------------------

    gl.clear(color=(0.2, 0.2, 0))

    render_scene(
        main_cam_view_matrix,
        main_cam_persp_matrix,
        light_cam_view_matrix,
        light_cam_persp_matrix,
        light_point,
        main_cam_point,
    )

    # TODO: Replace render scene call with 2 passes, one that creates shadow map, and
    # another that uses the shadow map. Each pass calls render_scene with appropriate uniforms.

    if debug_mode:
        show_shadow_map()

    pygame.display.flip()

pygame.quit()
