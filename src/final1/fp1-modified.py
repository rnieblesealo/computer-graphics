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
) 

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)

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

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

uniform mat4 model;         // Transforms model vertices from model space to world space
uniform mat4 view;          // Transforms general coords from world space to camera space
uniform mat4 perspective;   // Converts 3D coordinates to 2D screen coords

// ====================================================================================================
// MAIN
// ====================================================================================================

void main(){
  // = DISCARDS ======
  // f_normal = vec3(0); // This was a bug, it zeroed out all normals. It has been removed.
  // =================

  f_uv = in_uv;

  vec4 P = model * vec4(in_position, 1);  // Convert model from its space to world space
  f_position = P.xyz;                     // Set position in world coordinates
  gl_Position = perspective * view * P;   // Apply perspective and view to the converted model coords; this is what will be rendered

  mat3 normal_matrix = mat3(transpose(inverse(model)));   
  f_normal = normalize(normal_matrix * in_normal);         
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

uniform sampler2D map;            // The texture map we sample from with UV
uniform vec3 eye_position;        // The point at which the camera is
uniform bool metal;               // Is this a metal? 

uniform samplerCube environment_map; // The 6-sided cubemap for reflections

// ====================================================================================================
// CONSTANTS
// ====================================================================================================

// Constants for hemispherical lighting as requested
const vec3 skyColor = vec3(0.718, 0.741, 0.753);
const vec3 groundColor = vec3(0.322, 0.4, 0.11);
const vec3 skyDirection = vec3(0, 1, 0);

// ====================================================================================================
// FUNCTIONS
// ====================================================================================================

/* Calculates color at a given vertex */
vec3 computeColor(){
  vec3 color = vec3(0);
  vec3 N = normalize(f_normal);
  vec3 V = normalize(eye_position - f_position); // View direction from fragment to camera

  // Sample the material texture (e.g., "gold.jpg")
  vec3 material_color = texture(map, f_uv).rgb;

  // Lighting logic
  if (metal){
    // METALLIC: Use environment map reflection
    // Calculate the reflection vector
    vec3 R = reflect(-V, N); // R = reflect(Incident, Normal)
    
    // Sample the cubemap with the reflection vector
    vec3 reflection_color = texture(environment_map, R).rgb;
    
    // Tint the reflection with the material's color
    color = material_color * reflection_color;
    
  } else {
    // NON-METALLIC (DIFFUSE): Use hemispherical lighting
    // Calculate the interpolation factor based on the normal's direction
    float hemisphere_factor = (dot(N, skyDirection) + 1.0) * 0.5;
    
    // Linearly interpolate between ground and sky color
    vec3 hemisphere_light = mix(groundColor, skyColor, hemisphere_factor);
    
    // Apply the computed ambient light to the material's color
    color = material_color * hemisphere_light;
  }

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

TEAPOT_SHADER_PROGRAM = ctx.program(
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

MODEL_RENDERABLES = MODEL.getRenderables(ctx, TEAPOT_SHADER_PROGRAM, FORMAT, FORMAT_VARIABLES)
SCENE = MODEL.scene

MODEL_BOUNDS = MODEL.bound

# ======================================================================
# RENDERING FUNCTIONS
# ======================================================================

def recursiveRender(node, M):
  """
  Renders a scenegraph node and also renders its children recursively
  """
  nodeTransform = glm.transpose(glm.mat4(node.transformation))
  currentTransform = M * nodeTransform

  if node.num_meshes > 0:
    for index in node.mesh_indices:
      # Use .write(matrix.to_bytes()) for consistency
      MODEL_RENDERABLES[index]._program["model"].write(currentTransform.to_bytes())
      MODEL_RENDERABLES[index].render()

  for node in node.children:
    recursiveRender(node, currentTransform)

 
def render():
  """
  Does our rendering :)
  """
  recursiveRender(SCENE.root_node, M=glm.mat4(1)) 

# ======================================================================
# TEXTURE SETUP 
# ======================================================================

GOLD_IMAGE_FILEPATH = Path("./gold.jpg")
GOLD_TEXTURE_IMAGE = pygame.image.load(GOLD_IMAGE_FILEPATH.as_posix())
GOLD_TEXTURE_DATA = pygame.image.tobytes(GOLD_TEXTURE_IMAGE, "RGB", True) 
GOLD_TEXTURE = ctx.texture(GOLD_TEXTURE_IMAGE.get_size(), data=GOLD_TEXTURE_DATA, components=3) 
GOLD_TEXTURE_SAMPLER = ctx.sampler(texture=GOLD_TEXTURE)   

# Function to load the 6 cubemap faces
def load_cubemap(ctx):
    """
    Loads 6 images into a ModernGL cube texture, manually correcting Y-axis orientation.
    """
    # The order is: +X (Right), -X (Left), +Y (Up), -Y (Down), +Z (Front), -Z (Back)
    cubemap_paths = [
        Path('./Footballfield/posx.jpg'), # 0: Right
        Path('./Footballfield/negx.jpg'), # 1: Left
        Path('./Footballfield/posy.jpg'), # 2: Up (Sky)
        Path('./Footballfield/negy.jpg'), # 3: Down (Ground)
        Path('./Footballfield/posz.jpg'), # 4: Front
        Path('./Footballfield/negz.jpg')  # 5: Back
    ]
    
    # Check if paths exist
    if not all(p.exists() for p in cubemap_paths):
        print("="*50)
        print("WARNING: CUBEMAP IMAGES NOT FOUND.")
        print("Please check the folder './Footballfield' for your 6 cubemap images:")
        print("posx.jpg, negx.jpg, posy.jpg, negy.jpg, posz.jpg, negz.jpg")
        print("Using a dummy placeholder texture.")
        print("="*50)
        dummy_data = [(255, 0, 255) * 64 * 64] * 6 
        return ctx.texture_cube((64, 64), 3, data=b''.join(dummy_data))

    print("Loading cubemap images...")
    images = [pygame.image.load(path.as_posix()) for path in cubemap_paths]
    
    # --- FIX for upside-down skybox ---
    # The +Y (index 2) and -Y (index 3) faces often need to be vertically flipped 
    # when using the standard OpenGL/ModernGL cubemap order and coordinate system.
    # We use pygame.transform.flip(image, x_flip, y_flip)
    
    # MODIFIED: Flip the Up (+Y) and Down (-Y) faces vertically
    images[2] = pygame.transform.flip(images[2], False, True) # posy
    images[3] = pygame.transform.flip(images[3], False, True) # negy

    # End FIX ---

    # Convert all images (now correctly oriented) to a single bytes object
    list_of_bytes = [pygame.image.tobytes(img, "RGB", True) for img in images]
    image_data_combined = b''.join(list_of_bytes) 
    
    size = images[0].get_size()
    
    print(f"Cubemap loaded successfully with size {size}.")
    return ctx.texture_cube(size, 3, data=image_data_combined)

# Load the cubemap and create its sampler
ENVIRONMENT_CUBEMAP = load_cubemap(ctx)
ENVIRONMENT_SAMPLER = ctx.sampler(texture=ENVIRONMENT_CUBEMAP, repeat_x=False, repeat_y=False)   

# ======================================================================
# SKYBOX SETUP 
# ======================================================================

SB_POSITIONS = numpy.array([
  [-1, 1],
  [1, 1],
  [1, -1],
  [-1, -1],
]).astype(numpy.float32) 

SB_GEOMETRY = SB_POSITIONS.flatten()

SB_INDICES = numpy.array([
  0, 1, 2,
  2, 3, 0
]).astype(numpy.int32)

# Skybox vertex shader now un-projects to find world-space direction
SB_VERTEX_SHADER = """
#version 460 core

in vec2 in_position; // Fullscreen quad coords (-1 to 1)

// Inverse matrices to un-project
uniform mat4 inv_view_matrix;
uniform mat4 inv_perspective_matrix;

out vec3 f_direction; // World-space direction to sample cubemap

void main(){
  // Un-project screen-space coordinates to a world-space direction
  // We are at the far clip plane, z=1
  vec4 clip_space_pos = vec4(in_position.xy, 1.0, 1.0);
  
  // To view space
  vec4 view_space_pos = inv_perspective_matrix * clip_space_pos;
  
  // To world space direction (removing translation from view matrix)
  // We only use mat3 to discard the translation part of the view matrix
  f_direction = mat3(inv_view_matrix) * (view_space_pos.xyz / view_space_pos.w);

  // Output fullscreen quad position, at the far plane (z=1)
  // This ensures it's "behind" all other objects
  gl_Position = clip_space_pos;
}
"""

# Skybox fragment shader now samples the cubemap
SB_FRAGMENT_SHADER = """
#version 460 core

in vec3 f_direction; // World-space direction from VS

uniform samplerCube environment_map; // Cubemap texture

out vec4 out_color;

void main(){
  // Sample the cubemap with the direction vector
  // We normalize here to be safe
  out_color = texture(environment_map, normalize(f_direction));
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

SB_VAO = ctx.vertex_array(
  SB_PROGRAM,
  [(SB_VBO, SB_VBO_FORMAT, SB_VBO_VARIABLES)],
  index_buffer=SB_INDEX_BUFFER, 
  index_element_size=4
)

# ======================================================================
# CAMERA/LIGHT SETUP
# ======================================================================

UP = glm.vec3(0, 1, 0)
RIGHT = glm.vec3(1, 0, 0)

displacement_vector = 2 * MODEL_BOUNDS.radius * glm.rotate(UP, glm.radians(85), RIGHT)
light_displacement_vector = 2 * MODEL_BOUNDS.radius * glm.rotate(UP, glm.radians(45), RIGHT)
target_point = glm.vec3(MODEL_BOUNDS.center)

FOV = glm.radians(45)
ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT 
NEAR_PLANE = MODEL_BOUNDS.radius
FAR_PLANE = 3 * MODEL_BOUNDS.radius

perspective_matrix = glm.perspective(FOV, ASPECT, NEAR_PLANE, FAR_PLANE)

# ======================================================================
# MAIN RUN LOOP
# ======================================================================

TARGET_FPS = 60

is_running = True
clock = pygame.time.Clock()

teapot_rotation = 0
light_angle = 0 

is_metal = True 
is_paused = True
use_skybox = True 

# Use Z buffer
ctx.depth_func = "<=" 
ctx.enable(ctx.DEPTH_TEST)

while is_running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      is_running = False
    elif event.type == pygame.KEYDOWN:
      if event.key == pygame.K_m:
        is_metal = not is_metal
        print(f"Set to metal: {is_metal}")
      if event.key == pygame.K_p:
        is_paused = not is_paused
    elif event.type == pygame.WINDOWRESIZED:
      new_width = event.x
      new_height = event.y
      ASPECT = new_width / new_height 
      perspective_matrix = glm.perspective(FOV, ASPECT, NEAR_PLANE, FAR_PLANE) 
      ctx.viewport = (0, 0, new_width, new_height)

  # --- Update -------------------------------------------------------------------------------------

  new_displacement_vector = glm.rotate(displacement_vector, glm.radians(teapot_rotation), UP) 
  eye_position = target_point + new_displacement_vector
  view_matrix = glm.lookAt(eye_position, target_point, UP)

  # --- Render -------------------------------------------------------------------------------------

  ctx.clear(color=(0, 0, 0))

  # --- Skybox rendering -------------------------------------------------------------------------
  if use_skybox:
    ctx.depth_func = "<=" 
    curr_program = SB_PROGRAM
    
    # Explicitly convert glm.mat4 to bytes using .to_bytes()
    curr_program["inv_view_matrix"].write(glm.inverse(view_matrix).to_bytes())
    curr_program["inv_perspective_matrix"].write(glm.inverse(perspective_matrix).to_bytes())
    
    curr_program["environment_map"] = 0 
    ENVIRONMENT_SAMPLER.use(0)
    
    SB_VAO.render()

  # --- Teapot rendering -------------------------------------------------------------------------
  
  ctx.depth_func = "<=" 
  
  curr_program = TEAPOT_SHADER_PROGRAM

  # Use .write(matrix.to_bytes()) for all matrix uniforms
  curr_program["view"].write(view_matrix.to_bytes())
  curr_program["perspective"].write(perspective_matrix.to_bytes())
  
  # Non-matrix uniforms can still use direct assignment or .write() on vec3/vec4
  curr_program["eye_position"].write(eye_position)
  
  # curr_program["light"].write(new_light_displacement_vector) # Still commented out

  curr_program["map"] = 0
  GOLD_TEXTURE_SAMPLER.use(0)

  curr_program["environment_map"] = 1 
  ENVIRONMENT_SAMPLER.use(1)

  curr_program["metal"] = is_metal

  render()


  pygame.display.flip()

  clock.tick(TARGET_FPS)

  if not is_paused:
    teapot_rotation += 1 
    if teapot_rotation > 360:
      teapot_rotation = 0

pygame.quit()
