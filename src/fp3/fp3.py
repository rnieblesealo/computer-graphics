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

# =========================================================================================================
# USEFUL CONSTANTS
# =========================================================================================================

UP = glm.vec3(0, 1, 0)
FLOOR_SCALE = 3  # Factor to multiply the model bound's radius by

# =========================================================================================================
# PYGAME SETUP
# =========================================================================================================

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

pygame.init()  # Initlizes its different modules. Display module is one of them.

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)
pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
)
pygame.display.set_caption(title="Project Substitute 3: Rafael Niebles")

gl = moderngl.get_context()

frame_buffer = (
    gl.detect_framebuffer()
)  # Get the framebuffer so we can perform off-screen rendering

# =========================================================================================================
# READ MODEL
# =========================================================================================================

MODEL_FILEPATH = Path("./chair_table_class/scene.gltf")

model = create3DAssimpObject(
    MODEL_FILEPATH.as_posix(), verbose=False, textureFlag=True, normalFlag=True
)

model_bounds = model.bound

# =========================================================================================================
# CREATE FLOOR GEOMETRY
# =========================================================================================================

min_corner = model_bounds.boundingBox[0]
max_corner = glm.vec3(
    model_bounds.boundingBox[1].x, min_corner.y, model_bounds.boundingBox[1].z
)
ctr = (min_corner + max_corner) / 2

plane_point = ctr
plane_normal = UP  # Floor plane

floor_quad_side_length = FLOOR_SCALE * model_bounds.radius
floor_quad_mid_side_length = floor_quad_side_length / 2

# fmt: off
floor_quad_vertices = np.array([
  ctr.x - floor_quad_mid_side_length, ctr.y, ctr.z - floor_quad_mid_side_length, 0, 1, 0, 0, 0,
  ctr.x + floor_quad_mid_side_length, ctr.y, ctr.z - floor_quad_mid_side_length, 0, 1, 0, 1, 0,
  ctr.x + floor_quad_mid_side_length, ctr.y, ctr.z + floor_quad_mid_side_length, 0, 1, 0, 1, 1,
  ctr.x - floor_quad_mid_side_length, ctr.y, ctr.z + floor_quad_mid_side_length, 0, 1, 0, 0, 1
]).astype(np.float32)
# fmt: on

floor_quad_vbo = gl.buffer(floor_quad_vertices)

# fmt: off
floor_quad_indices = numpy.array([
  0, 1, 2, 
  2, 3, 0
]).astype(np.float32)
# fmt: on

floor_quad_ibo = gl.buffer(floor_quad_indices)

# =========================================================================================================
# CREATE FLOOR TEXTURE SAMPLER
# =========================================================================================================

WOOD_TEXTURE_PATH = Path("./tile-squares-texture.jpg")

wood_texture_img = pygame.image.load(WOOD_TEXTURE_PATH.as_posix())
wood_texture_data = pygame.image.tobytes(
    wood_texture_img, "RGB", True
)  # Flip to match OpenGL coords
wood_texture = gl.texture(
    wood_texture_img.get_size(), data=wood_texture_data, components=3
)
wood_texture.build_mipmaps()

floor_texture_sampler = gl.sampler(
    texture=wood_texture,
    filter=(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR),
    repeat_x=True,
    repeat_y=True,
)

# =========================================================================================================
# SHADERS
# =========================================================================================================

MODEL_VERTEX_SHADER = """
#version 460 core

// =========================================================================================================
// INPUT
// =========================================================================================================

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// =========================================================================================================
// OUTPUT
// =========================================================================================================

out vec2 f_uv; // Same as in
out vec3 f_normal; // Normal in world coordinates (and normalized to 0-1)
out vec3 f_position; // Position in world coordinates

// =========================================================================================================
// UNIFORM
// =========================================================================================================

uniform mat4 model; // Converts from model -> world
uniform mat4 view; // Converts from world -> camera
uniform mat4 perspective; // Applies depth

// =========================================================================================================
// MAIN
// =========================================================================================================

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

MODEL_FRAGMENT_SHADER = """
#version 460 core

// =========================================================================================================
// CONSTANTS
// =========================================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =========================================================================================================
// INPUT
// =========================================================================================================

in vec2 f_uv; // UV
in vec3 f_normal; // Normalized and in world coords
in vec3 f_position; // In world coords

// =========================================================================================================
// OUTPUT
// =========================================================================================================

out vec4 out_color;

// =========================================================================================================
// UNIFORM
// =========================================================================================================

uniform int pcf; // Percentage closer filtering
uniform bool bias_flag; // Use shadow bias
uniform float shininess;
uniform vec3 eye_position;
uniform vec3 k_diffuse; // Diffuse coefficient; how much incoming light should scatter? 
uniform vec3 light;
uniform sampler2D map;

// =========================================================================================================
// AUX FUNCTIONS
// =========================================================================================================

float computeVisibilityFactor(){
  return 1.0;
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

// =========================================================================================================
// MAIN
// =========================================================================================================

void main(){
  out_color = vec4(computeColor(), 1);
}
"""

model_program = gl.program(
    vertex_shader=MODEL_VERTEX_SHADER, fragment_shader=MODEL_FRAGMENT_SHADER
)
