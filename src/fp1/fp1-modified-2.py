import pygame
import moderngl
import numpy
import glm
from pygame.image import load
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

TEAPOT_VERTEX_SHADER = """
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
    // f_normal = vec3(0); // <--- FIX 1: This line was the bug, it has been removed.
    // =================

    f_uv = in_uv;

    vec4 P = model * vec4(in_position, 1);  // Convert model from its space to world space
    f_position = P.xyz;                     // Set position in world coordinates
    gl_Position = perspective * view * P;   // Apply perspective and view to the converted model coords; this is what will be rendered

    mat3 normal_matrix = mat3(transpose(inverse(model)));   // Inverse transpose of model transformation
    f_normal = normalize(normal_matrix * in_normal);        // Multiply to get normals
}
"""

TEAPOT_FRAGMENT_SHADER = """
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

// Skybox

uniform samplerCube skybox_map; // 6-sided skybox cubemap; UV, but samples a cubemap 

// ====================================================================================================
// CONSTANTS
// ====================================================================================================

const float shininess = 5; // Reflection factor

// Skybox

const vec3 sky_color = vec3(0.718, 0.741, 0.753);       // Given by prof.
const vec3 ground_color = vec3(0.322, 0.400, 0.110);    // Also given by prof.
const vec3 sky_direction = vec3(0, 1, 0);               // This is the same as up

// ====================================================================================================
// FUNCTIONS
// ====================================================================================================

/* Calculates color at a given vertex */
vec3 computeColor(){
    vec3 color = vec3(0);

    // Normalization for all here to keep things consistent!

    vec3 N = normalize(f_normal);
    vec3 V = normalize(eye_position - f_position); // Distance between camera and vertex, normalized

    // Sample the texture for surface
    vec3 material_color = texture(map, f_uv).rgb;

    if (metal){
        // Metallic surfaces should reflect

        // Calculate light reflection vector
        vec3 reflection_vector = reflect(-V, N);

        // Sample the cubemap using the reflection vector
        vec3 reflection_color = texture(skybox_map, reflection_vector).rgb;

        // Apply 
        color = material_color * reflection_color; 
    } else {
        // Non-metallic surfaces should use hemispherical lighting
        // Hemispherical/diffuse lighting = simulates light coming from ALL directions, not just a specific one

        float diffuse_factor = (dot(N, sky_direction) + 1.0) * 0.5;
        vec3 diffuse_color = mix(ground_color, sky_color, diffuse_factor);

        // Apply
        color = material_color * diffuse_color;
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
    vertex_shader=TEAPOT_VERTEX_SHADER, fragment_shader=TEAPOT_FRAGMENT_SHADER
)

# ======================================================================
# MODEL/OBJECT SETUP
# ======================================================================

MODEL_FILEPATH = Path("./the_utah_teapot/scene.gltf")
MODEL = create3DAssimpObject(
    MODEL_FILEPATH.as_posix(), verbose=False, textureFlag=True, normalFlag=True
)

FORMAT = "3f 3f 2f"
FORMAT_VARIABLES = ["in_position", "in_normal", "in_uv"]

MODEL_RENDERABLES = MODEL.getRenderables(
    ctx, TEAPOT_SHADER_PROGRAM, FORMAT, FORMAT_VARIABLES
)
SCENE = MODEL.scene

MODEL_BOUNDS = MODEL.bound

# ======================================================================
# AUXILIARY FUNCTIONS
# ======================================================================


def loadCubemapTextures(ctx: moderngl.Context):
    """
    Loads the 6-sided cubemap textures
    """

    cubemap_paths = [
        Path("./Footballfield/posx.jpg"), # MODIFIED: Re-ordered to match standard
        Path("./Footballfield/negx.jpg"),
        Path("./Footballfield/posy.jpg"),
        Path("./Footballfield/negy.jpg"),
        Path("./Footballfield/posz.jpg"),
        Path("./Footballfield/negz.jpg"),
    ]

    print("Loading cubemap images...")
    cubemap_images = [pygame.image.load(path.as_posix()) for path in cubemap_paths]

    # --- FIX for upside-down skybox ---
    # We must flip the +Y (up) and -Y (down) images vertically
    cubemap_images[2] = pygame.transform.flip(cubemap_images[2], False, True)  # posy
    cubemap_images[3] = pygame.transform.flip(cubemap_images[3], False, True)  # negy
    # --- End FIX ---
    
    # Convert to bytes
    cubemap_image_data = [
        pygame.image.tobytes(image, "RGB", True) for image in cubemap_images
    ]

    # Concatenate individual list entries into big byte block
    cubemap_image_data_combined = b"".join(cubemap_image_data)

    cubemap_image_size = cubemap_images[0].get_size()
    
    print(f"Cubemap loaded successfully with size {cubemap_image_size}.")

    RGB_CHANNEL_COUNT = 3

    return ctx.texture_cube(
        cubemap_image_size, RGB_CHANNEL_COUNT, data=cubemap_image_data_combined
    )


def recursiveRender(node, M):
    """
    Renders a scenegraph node and also renders its children recursively
    """
    nodeTransform = glm.transpose(glm.mat4(node.transformation))
    currentTransform = M * nodeTransform

    if node.num_meshes > 0:
        for index in node.mesh_indices:
            # --- FIX 2: Added .to_bytes() ---
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
GOLD_TEXTURE_DATA = pygame.image.tobytes(
    GOLD_TEXTURE_IMAGE, "RGB", True
)
GOLD_TEXTURE = ctx.texture(
    GOLD_TEXTURE_IMAGE.get_size(), data=GOLD_TEXTURE_DATA, components=3
)
GOLD_TEXTURE_SAMPLER = ctx.sampler(texture=GOLD_TEXTURE)

# ======================================================================
# SKYBOX SETUP
# ======================================================================

SB_POSITIONS = numpy.array(
    [
        [-1, 1],
        [1, 1],
        [1, -1],
        [-1, -1],
    ]
).astype(
    numpy.float32
)

SB_GEOMETRY = SB_POSITIONS.flatten()
SB_INDICES = numpy.array([0, 1, 2, 2, 3, 0]).astype(numpy.int32)

SB_VERTEX_SHADER = """
#version 460 core
in vec2 in_position;
out vec3 f_direction;
uniform mat4 inv_view_matrix;
uniform mat4 inv_perspective_matrix;

void main(){
    vec4 clip_space_pos = vec4(in_position.xy, 1, 1);
    vec4 view_space_pos = inv_perspective_matrix * clip_space_pos;
    f_direction = mat3(inv_view_matrix) * (view_space_pos.xyz / view_space_pos.w);
    gl_Position = clip_space_pos; 
}
"""

SB_FRAGMENT_SHADER = """
#version 460 core
in vec3 f_direction;
out vec4 out_color;
uniform samplerCube skybox_map;

void main(){
    out_color = texture(skybox_map, normalize(f_direction));
}
"""

SB_PROGRAM = ctx.program(
    vertex_shader=SB_VERTEX_SHADER, fragment_shader=SB_FRAGMENT_SHADER
)

SB_VBO = ctx.buffer(SB_GEOMETRY)
SB_VBO_FORMAT = "2f"
SB_VBO_VARIABLES = "in_position"
SB_INDEX_BUFFER = ctx.buffer(SB_INDICES)

SB_VAO = ctx.vertex_array(
    SB_PROGRAM,
    [(SB_VBO, SB_VBO_FORMAT, SB_VBO_VARIABLES)],
    index_buffer=SB_INDEX_BUFFER,
    index_element_size=4,
)

SB_CUBEMAP_TEXTURE = loadCubemapTextures(ctx)
SB_CUBEMAP_SAMPLER = ctx.sampler(texture=SB_CUBEMAP_TEXTURE)

# ======================================================================
# RENDERING SETUP
# ======================================================================

UP = glm.vec3(0, 1, 0)
RIGHT = glm.vec3(1, 0, 0)

displacement_vector = 2 * MODEL_BOUNDS.radius * glm.rotate(UP, glm.radians(85), RIGHT)
light_displacement_vector = (
    2 * MODEL_BOUNDS.radius * glm.rotate(UP, glm.radians(45), RIGHT)
)
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

# Set to False by default, as in your file.
# The teapot will now appear in diffuse mode.
is_metal = False 
is_paused = True
use_skybox = True

ctx.depth_func = "<="
ctx.enable(ctx.DEPTH_TEST)

while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        elif event.type == pygame.KEYDOWN:
            # You can add your controls here
            pass
        elif event.type == pygame.WINDOWRESIZED:
            new_width = event.x
            new_height = event.y
            ASPECT = new_width / new_height # Recalculate aspect ratio
            perspective_matrix = glm.perspective(FOV, ASPECT, NEAR_PLANE, FAR_PLANE)
            ctx.viewport = (0, 0, new_width, new_height) # Update viewport

    # --- Update -------------------------------------------------------------------------------------

    new_displacement_vector = glm.rotate(
        displacement_vector, glm.radians(teapot_rotation), UP
    )
    new_light_displacement_vector = glm.rotate(
        light_displacement_vector, glm.radians(light_angle), UP
    )
    eye_position = target_point + new_displacement_vector
    view_matrix = glm.lookAt(eye_position, target_point, UP)

    # --- Render -------------------------------------------------------------------------------------

    ctx.clear(color=(0, 0, 0))

    # Skybox rendering
    if use_skybox:
        curr_program = SB_PROGRAM

        curr_program["inv_view_matrix"].write(glm.inverse(view_matrix).to_bytes())
        curr_program["inv_perspective_matrix"].write(
            glm.inverse(perspective_matrix).to_bytes()
        )

        curr_program["skybox_map"] = 0
        SB_CUBEMAP_SAMPLER.use(0)

        SB_VAO.render()

    # Teapot rendering
    curr_program = TEAPOT_SHADER_PROGRAM

    # --- FIX 3 & 4: Added .to_bytes() ---
    curr_program["view"].write(view_matrix.to_bytes())
    curr_program["perspective"].write(perspective_matrix.to_bytes())
    
    curr_program["eye_position"].write(eye_position)
    # curr_program["light"].write(new_light_displacement_vector)

    curr_program["map"] = 0
    GOLD_TEXTURE_SAMPLER.use(0)
    
    # --- NOTE: This is the NEXT fix ---
    # If you set is_metal = True, the teapot will be black
    # until you add these two lines:
    #
    # curr_program["skybox_map"] = 1
    # SB_CUBEMAP_SAMPLER.use(1)
    #
    
    curr_program["metal"] = is_metal

    render()

    pygame.display.flip()

    clock.tick(TARGET_FPS)

    if not is_paused:
        teapot_rotation += 1
        if teapot_rotation > 360:
            teapot_rotation = 0

pygame.quit()
