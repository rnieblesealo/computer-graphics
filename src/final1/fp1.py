import re
import pygame
import moderngl
import numpy
import glm
from loadModelUsingAssimp_V2 import create3DAssimpObject
from pathlib import Path

# ======================================================================
# PYGAME SETUP
# ======================================================================

pygame.init()
pygame.display.gl_set_attribute(
    pygame.GL_MULTISAMPLEBUFFERS, 1
)  # Request multisample buffer, can only be 0 or 1

"""
- Multisampling = Pixels are treated as a large area instead of as a point
  - Pixel is large area divided into regions 
  - A sample is taken from each region
  - The samples are averaged to give the larger pixel its final value
- Think of multisampling as taking a pencil drawing and rubbing your finger a bit; it smooths out the edges!
  - This is why the end result is reduced aliasing :) 
- The regions from a pixel can also belong to another pixel 
"""

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)

"""
- The regions are NOT necessarily perfect subdivisions; this varies by technique, it may be random!
- The sample count = The amount of regions
- A sample count of 16 = 16 different regions, 16 different samples 
"""

pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
)

pygame.display.set_caption(title="Project Substitute 1: Rafael Niebles")

# ======================================================================
# MODERNGL SETUP
# ======================================================================

ctx = moderngl.get_context()

# ======================================================================
# SHADER CODE
# ======================================================================

VERTEX_SHADER = """
#version 460 core

// ====================================================================================================
// INPUT VARIABLES 
// ====================================================================================================

layout(location = 0) in vec3 in_position; // Positions from vertex data 
layout(location = 1) in vec3 in_normal;   // Normals from vertex data
layout(location = 2) in vec2 in_uv;       // Texture coordinates from vertex data

// ====================================================================================================
// OUTPUT VARIABLES
// ====================================================================================================

out vec2 f_uv;        // THE SAME as in_uv
out vec3 f_normal;    // Normal in world coordinates
out vec3 f_position;  // Position in world coordinates

// layout location = Helps tell shader code the order in which we sent in stuff

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

uniform mat4 model;         // Transforms model vertices from model space to world space
uniform mat4 view;          // Transforms general coords from world space to camera space
uniform mat4 perspective;   // Converts 3D coordinates to 2D screen coords
// uniform mat3 normalMatrix;  // Converts normals from model space to camera space; for lighting purposes

// ====================================================================================================
// MAIN
// ====================================================================================================

void main(){
  // = DISCARDS ======

  f_normal = vec3(0);

  // =================

  f_uv = in_uv;

  vec4 P = model * vec4(in_position, 1);  // Convert model from its space to world space
  f_position = P.xyz;                     // Set position in world coordinates
  gl_Position = perspective * view * P;   // Apply perspective and view to the converted model coords; this is what will be rendered

  mat3 normalMatrix = mat3(transpose(inverse(model)));  // Inverse transpose of model transformation (???)
  f_normal = normalize(normalMatrix * in_normal);       // Multiply to get normals (???)
}
"""

FRAGMENT_SHADER = """
#version 460 core

// ====================================================================================================
// INPUT VARIABLES 
// ====================================================================================================

  in vec3 f_normal;    // Model normals in world coords
  in vec3 f_position;  // Model position in world coords
  in vec2 f_uv;        // UVs as they are

// ====================================================================================================
// OUTPUT VARIABLES 
// ====================================================================================================

out vec4 out_color;

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

uniform sampler2D map;      // The texture map we sample from with UV
uniform vec3 light;         // The point at which the light is 
uniform vec3 eye_position;  // The point at which the camera is
uniform bool metal;         // Is this a metal? 

// ====================================================================================================
// CONSTANTS
// ====================================================================================================

const float shininess = 5; // Reflection factor

// ====================================================================================================
// FUNCTIONS
// ====================================================================================================

/* Calculates color at a given vertex */
vec3 computeColor(){
  vec3 color = vec3(0);

  // Normalization for all here to keep things consistent!

  vec3 N = normalize(f_normal);
  vec3 V = normalize(eye_position - f_position); // Distance between camera and vertex, normalized
  vec3 L = normalize(light); 
  vec3 H = normalize(L + V); // Where light and view direction meet 

  // Sample the texture
  vec3 material_color = texture(map, f_uv).rgb;

  // Calculate color with specular highlights factoring in material properties 
  if (dot(N, L) > 0){
    // For metallic surface
    if (metal){
      color = material_color * pow(dot(N, H), shininess);
    // For non-metallic surface 
    } else {
      color = material_color * dot(N, L);
    }
  }

  // The dot product check ensures only surfaces facing the light source are lit

  return color;
}

// ====================================================================================================
// MAIN
// ====================================================================================================

void main(){
  out_color = vec4(computeColor(), 1);
}
"""

# ======================================================================
# SHADER PROGRAM SETUP
# ======================================================================

shader_program = ctx.program(
  vertex_shader=VERTEX_SHADER,
  fragment_shader=FRAGMENT_SHADER
)

# ======================================================================
# MODEL/OBJECT SETUP 
# ======================================================================

MODEL_FILEPATH = Path("./the_utah_teapot/scene.gltf") 
MODEL = create3DAssimpObject(MODEL_FILEPATH.as_posix(), verbose=False, textureFlag=True, normalFlag=True)

FORMAT = "3f 3f 2f"
FORMAT_VARIABLES = ["in_position", "in_normal", "in_uv"]

MODEL_RENDERABLES = MODEL.getRenderables(ctx, shader_program, FORMAT, FORMAT_VARIABLES)
SCENE = MODEL.scene

MODEL_BOUNDS = MODEL.bound

"""
The scenegraph is a data structure used to organize & manage spatial representation of a 3D scene
"""

# ======================================================================
# RENDERING FUNCTIONS
# ======================================================================

def recursiveRender(node, M):
  """
  Renders a scenegraph node and also renders its children recursively

  A scenegraph node is akin to an unity GameObject; it may have different properties
  (Transform, color, etc.)

  What a node and a scenegraph even is/behaves like depends on our own implementation

  node = The scenegraph node
  M = Accumulates transforms as we recur down the scene graph node's children
  """

  # nodeTransform represents local transform of the current node relative to its parent
  # If no parent, is relative to world
  nodeTransform = glm.transpose(glm.mat4(node.transformation))
  currentTransform = M * nodeTransform

  # Do we have anything to render?
  if node.num_meshes > 0:
    # If so, render each mesh
    for index in node.mesh_indices:
      MODEL_RENDERABLES[index]._program["model"].write(currentTransform)
      MODEL_RENDERABLES[index].render()

  # Recur
  for node in node.children:
    recursiveRender(node, currentTransform)

 
def render():
  """
  Does our rendering :)
  """

  recursiveRender(SCENE.root_node, M=glm.mat4(1)) # Initialize the transformation to an ID matrix

# ======================================================================
# TEXTURE SETUP 
# ======================================================================

GOLD_IMAGE_FILEPATH = Path("./gold.jpg")
GOLD_TEXTURE_IMAGE = pygame.image.load(GOLD_IMAGE_FILEPATH.as_posix())
GOLD_TEXTURE_DATA = pygame.image.tobytes(GOLD_TEXTURE_IMAGE, "RGB", True) # Convert to bytes as RGB format
GOLD_TEXTURE = ctx.texture(GOLD_TEXTURE_IMAGE.get_size(), data=GOLD_TEXTURE_DATA, components=3) # Create OpenGL texture object
GOLD_TEXTURE_SAMPLER = ctx.sampler(texture=GOLD_TEXTURE)   

# ======================================================================
# SKYBOX SETUP 
# ======================================================================

SB_POSITIONS = numpy.array([
  [-1, 1],
  [1, 1],
  [1, -1],
  [-1, -1],
]).astype(numpy.float32) # Presumably screen positions; defines a quad

# Turn into flat geometry 
SB_GEOMETRY = SB_POSITIONS.flatten()

# Get the indices
SB_INDICES = numpy.array([
  0, 1, 2,
  2, 3, 0
]).astype(numpy.int32)

SB_VERTEX_SHADER = """
#version 460 core

// Just make a quad; nothing special

in vec2 in_position;

void main(){
  gl_Position = vec4(in_position, 1, 1);
}
"""

SB_FRAGMENT_SHADER = """
#version 460 core

in vec3 f_position;

uniform vec3 eye_position;

out vec4 out_color;

void main(){
  out_color = vec4(0.5, 0.5, 0.5, 1);
}
"""

SB_PROGRAM = ctx.program(
  vertex_shader=SB_VERTEX_SHADER,
  fragment_shader=SB_FRAGMENT_SHADER
)

SB_VBO = ctx.buffer(SB_GEOMETRY)
SB_VBO_FORMAT = "2f"
SB_VBO_VARIABLES = "in_position"
SB_INDEX_BUFFER = ctx.buffer(SB_INDICES)

# Renderable skybox VAO
SB_VAO = ctx.vertex_array(
  SB_PROGRAM,
  [(SB_VBO, SB_VBO_FORMAT, SB_VBO_VARIABLES)],
  index_buffer=SB_INDEX_BUFFER, 
  index_element_size=4
)

# ======================================================================
# SKYBOX SETUP 
# ======================================================================

DISPLACEMENT_VECTOR = 2 * MODEL_BOUNDS.radius * glm.rotate(glm.vec3(0, 1, 0), glm.radians(85), glm.vec3(1, 0, 0))

LIGHT_DISPLACEMENT_VECTOR = 2 * MODEL_BOUNDS.radius * glm.rotate(glm.vec3(0, 1, 0), glm.radians(45), glm.vec3(1, 0, 0))

TARGET_POINT = glm.vec3(MODEL_BOUNDS.center)

UP = glm.vec3(0, 1, 0)

# View volume

FOV = glm.radians(45)
ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT 
NEAR_PLANE = MODEL_BOUNDS.radius
FAR_PLANE = 3 * MODEL_BOUNDS.radius
PERSPECTIVE = glm.perspective(FOV, ASPECT, NEAR_PLANE, FAR_PLANE)