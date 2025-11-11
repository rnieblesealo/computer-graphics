import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V3 import create3DAssimpObject
from OpenGL.GL import *

# ===============================================================================================================
# PYGAME SETUP
# ===============================================================================================================

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

pygame.init()

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)  # Enable multisampling
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)  # 16 samples
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
)
pygame.display.set_caption(title="Final Project 02: Rafael Niebles")

# ===============================================================================================================
# MODERNGL SETUP
# ===============================================================================================================

ctx = moderngl.get_context()

# ===============================================================================================================
# SHADERS
# ===============================================================================================================

MODEL_VERTEX_SHADER = """
#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec2 f_uv; // Same as in_uv
out vec3 f_normal; // in_normal in world coords 
out vec3 f_position;  // in_position in world coords

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform mat4 model; // Transforms model -> world space
uniform mat4 view; // Transforms world space -> camera space
uniform mat4 perspective; // Transforms 3D coords -> 2D screen coords, creating depth effect

// =========================================================================================
// MAIN 
// =========================================================================================

void main(){
  f_uv = in_uv;

  // Convert model verts to world space 
  vec4 final_pos = model * vec4(in_position, 1);
  f_position = final_pos.xyz;

  // Compute normal matrix
  // This matrix correctly applies any model transformations to its normals
  mat3 normal_matrix = mat3(transpose(inverse(model)));

  // Apply the normal matrix to the model to match its world space transform
  f_normal = normalize(normal_matrix * in_normal);

  // Convert to world space and apply perspective
  gl_Position = perspective * view * final_pos;
}
"""

MODEL_FRAGMENT_SHADER = """
#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

in vec2 f_uv;
in vec3 f_normal; // World space normals
in vec3 f_position; // World space position

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec4 out_color;

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform sampler2D map; // Texture sampler
uniform float shininess; // Glossiness
uniform vec3 eye_position; // Camera pos in world space
uniform vec3 k_diffuse; // Diffuse reflection factor
uniform vec4 light; // Light vector (xyz) and type (w)

// =========================================================================================
// CONSTANT
// =========================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =========================================================================================
// FUNCTIONS
// =========================================================================================

vec3 computeColor(){
  // w = Encodes type of light 
  // If w > 0, we have a directional light; we can use xyz to determine direction 
  // If w <= 0, we have a point light; xyz represent its position
  // The light's visual properties are defined elsewhere 

  // Initialize light vector; assume directional
  vec3 light_vector = normalize(light.xyz);

  // If point light, get direction to light
  if (light.w > 0){
    light_vector = normalize(light.xyz - f_position);
  }

  // Get dot product of light dir. and normal 
  vec3 normal = normalize(f_normal); // Re-normalize for safety 
  float n_dot_l = dot(normal, light_vector);

  // Sample the texture
  vec3 material_color = texture(map, f_uv).rgb;

  // Compute dot product to evaluate surface orientation 
  float w = dot(normal, UP);

  // Compute ambient color
  vec3 ambient_color = 0.25 * (w * SKY_COLOR + (1 - w) * GROUND_COLOR) * material_color;

  // Initialize final color 
  vec3 color = vec3(0);

  if (n_dot_l > 0){
    // Compute diffusely reflected color
    vec3 diffusely_reflected_color = material_color * n_dot_l;

    // Compute halfway vector 
    vec3 eye_direction = normalize(eye_position - f_position); // Direction from vertex to camera
    vec3 halfway = normalize(light_vector + eye_direction);

    // Compute specularly reflected color 
    vec3 specularly_reflected_color = vec3(0);
    if (shininess > 0){
      specularly_reflected_color = vec3(1) * pow(dot(normal, halfway), shininess);
    }

    // Apply diffuse and specular to final color
    color = k_diffuse * diffusely_reflected_color + specularly_reflected_color;
  }

  // Apply ambient color to final color 
  color += ambient_color;

  return color;
}

// =========================================================================================
// FUNCTIONS
// =========================================================================================

void main(){
    out_color = vec4(computeColor(), 1);
}

"""

model_program = ctx.program(
    vertex_shader=MODEL_VERTEX_SHADER, fragment_shader=MODEL_FRAGMENT_SHADER
)
