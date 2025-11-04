import pygame
import moderngl
import numpy
import glm
from PIL import Image
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

// layout location = Helps tell shader code the order in which we sent in stuff

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

uniform mat4 model;         // Transforms model vertices from model space to world space
uniform mat4 view;          // Transforms general coords from world space to camera space
uniform mat4 perspective;   // Converts 3D coordinates to 2D screen coords
// uniform mat3 normal_matrix;  // Converts normals from model space to camera space; for lighting purposes

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

    mat3 normal_matrix = mat3(transpose(inverse(model)));   // Inverse transpose of model transformation (???)
    f_normal = normalize(normal_matrix * in_normal);        // Multiply to get normals (???)
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

    // Doesn't seem like we use these again?
    /*
    vec3 L = normalize(light); 
    vec3 H = normalize(L + V); // Where light and view direction meet 
    */

    // Default teapot texture to be white
    vec3 material_color = vec3(1, 1, 1);

    if (metal){
        // Metallic surfaces should reflect

        // Sample the texture for surface
        material_color = texture(map, f_uv).rgb;

        // Calculate light reflection vector
        vec3 reflection_vector = reflect(-V, N);

        // Sample the cubemap using the reflection vector
        vec3 reflection_color = texture(skybox_map, reflection_vector).rgb;

        // Apply 
        color = material_color * reflection_color; 
    } else {
        // Non-metallic surfaces should use hemispherical lighting
        // Hemispherical/diffuse lighting = simulates light coming from ALL directions, not just a specific one

        // I HAVE NO IDEA WHAT THIS DOES
        // I think it's interpolating between both ground and sky colors to blend them though!

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

"""
The scenegraph is a data structure used to organize & manage spatial representation of a 3D scene
"""

# ======================================================================
# AUXILIARY FUNCTIONS
# ======================================================================


def loadCubemapTextures(ctx: moderngl.Context):
    """
    Loads the 6-sided cubemap textures
    """

    cubemap_paths = [
        Path("./Footballfield/negx.jpg"),  # Left
        Path("./Footballfield/posx.jpg"),  # Right
        Path("./Footballfield/posy.jpg"),  # Up
        Path("./Footballfield/negy.jpg"),  # Down
        Path("./Footballfield/posz.jpg"),  # Front
        Path("./Footballfield/negz.jpg"),  # Back
    ]

    # Nothing else uses existence checking
    # For consistency I won't do that here either

    # Load all
    cubemap_images = [Image.open(path.as_posix()) for path in cubemap_paths]

    # Make coords match MGL
    cubemap_images = [
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in cubemap_images
    ]

    # Transform to array form

    # Convert to bytes
    cubemap_image_data = [numpy.array(image) for image in cubemap_images]

    # Concatenate individual list entries into big byte block
    cubemap_image_data = b"".join(cubemap_image_data)

    # NOTE: This assumes all images are same size! They should be; is a cubemap...
    cubemap_image_size = cubemap_images[0].size

    # Return the OpenGL skybox object
    RGB_CHANNEL_COUNT = 3

    return ctx.texture_cube(
        cubemap_image_size, RGB_CHANNEL_COUNT, data=cubemap_image_data
    )


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
    node_transform = glm.transpose(glm.mat4(node.transformation))
    current_transform = M * node_transform

    # Do we have anything to render?
    if node.num_meshes > 0:
        # If so, render each mesh
        for index in node.mesh_indices:
            MODEL_RENDERABLES[index]._program["model"].write(
                current_transform.to_bytes()
            )
            MODEL_RENDERABLES[index].render()

    # Recur
    for node in node.children:
        recursiveRender(node, current_transform)


def render():
    """
    Does our rendering :)
    """

    # Initialize the transformation to an ID matrix
    recursiveRender(SCENE.root_node, M=glm.mat4(1))


# ======================================================================
# TEXTURE SETUP
# ======================================================================


GOLD_IMAGE_FILEPATH = Path("./gold.jpg")
GOLD_TEXTURE_IMAGE = pygame.image.load(GOLD_IMAGE_FILEPATH.as_posix())
GOLD_TEXTURE_DATA = pygame.image.tobytes(
    GOLD_TEXTURE_IMAGE, "RGB", True
)  # Convert to bytes as RGB format
GOLD_TEXTURE = ctx.texture(
    GOLD_TEXTURE_IMAGE.get_size(), data=GOLD_TEXTURE_DATA, components=3
)  # Create OpenGL texture object
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
)  # Presumably screen positions; defines a quad

# Turn into flat geometry
SB_GEOMETRY = SB_POSITIONS.flatten()

# Get the indices
SB_INDICES = numpy.array([0, 1, 2, 2, 3, 0]).astype(numpy.int32)

# Set up shaders

SB_VERTEX_SHADER = """
#version 460 core

// ====================================================================================================
// INPUT VARIABLES 
// ====================================================================================================

in vec2 in_position;

// ====================================================================================================
// OUTPUT VARIABLES
// ====================================================================================================

out vec3 f_direction; // World space cube sampling direction (?)

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

uniform mat4 inv_view_matrix;
uniform mat4 inv_perspective_matrix;

// ====================================================================================================
// MAIN 
// ====================================================================================================

void main(){
    vec4 clip_space_pos = vec4(in_position.xy, 1, 1);
    vec4 view_space_pos = inv_perspective_matrix * clip_space_pos;

    // Compute sampling direction (???)
    f_direction = mat3(inv_view_matrix) * (view_space_pos.xyz / view_space_pos.w);

    gl_Position = clip_space_pos; 
}
"""

SB_FRAGMENT_SHADER = """
#version 460 core

// ====================================================================================================
// INPUT VARIABLES
// ====================================================================================================

in vec3 f_direction;

// ====================================================================================================
// OUTPUT VARIABLES
// ====================================================================================================

out vec4 out_color;

// ====================================================================================================
// UNIFORM VARIABLES
// ====================================================================================================

// uniform vec3 eye_position;

uniform samplerCube skybox_map;

// ====================================================================================================
// MAIN
// ====================================================================================================

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

# Renderable skybox VAO
SB_VAO = ctx.vertex_array(
    SB_PROGRAM,
    [(SB_VBO, SB_VBO_FORMAT, SB_VBO_VARIABLES)],
    index_buffer=SB_INDEX_BUFFER,
    index_element_size=4,
)

# Setup cubemap texture + sampler
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

# View volume

FOV = glm.radians(45)
NEAR_PLANE = MODEL_BOUNDS.radius
FAR_PLANE = 3 * MODEL_BOUNDS.radius

curr_aspect = SCREEN_WIDTH / SCREEN_HEIGHT
perspective_matrix = glm.perspective(FOV, curr_aspect, NEAR_PLANE, FAR_PLANE)

# ======================================================================
# MAIN RUN LOOP
# ======================================================================

TARGET_FPS = 60

is_running = True
clock = pygame.time.Clock()

teapot_rotation = 0
light_angle = 0

is_metal = False 
is_paused = True
use_skybox = True

# Use Z buffer
ctx.depth_func = "<="  # This has got something to do with Z buffer
ctx.enable(ctx.DEPTH_TEST)

while is_running:
    for event in pygame.event.get():
        # Quit event on hit window X
        if event.type == pygame.QUIT:
            is_running = False
        elif event.type == pygame.KEYDOWN:
            # Ugh, implement controls later...
            pass
        elif event.type == pygame.WINDOWRESIZED:
            # Recalculate perspective matrix on window resize
            new_width = event.x
            new_height = event.y

            curr_aspect = new_width / new_height

            perspective_matrix = glm.perspective(
                FOV, curr_aspect, NEAR_PLANE, FAR_PLANE
            )

    # --- Update -------------------------------------------------------------------------------------

    new_displacement_vector = glm.rotate(
        displacement_vector, glm.radians(teapot_rotation), UP
    )

    # new_light_displacement_vector = glm.rotate(
    #     light_displacement_vector, glm.radians(light_angle), UP
    # )

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

    curr_program["view"].write(view_matrix.to_bytes())
    curr_program["perspective"].write(perspective_matrix.to_bytes())
    curr_program["eye_position"].write(eye_position)
    # curr_program["light"].write(new_light_displacement_vector)

    curr_program["map"] = 0
    GOLD_TEXTURE_SAMPLER.use(0)

    curr_program["skybox_map"] = 1
    SB_CUBEMAP_SAMPLER.use(1)

    curr_program["metal"] = is_metal

    render()

    # Post render stuff
    pygame.display.flip()

    clock.tick(TARGET_FPS)

    # Rotate the teapot
    if not is_paused:
        teapot_rotation += 1
        if teapot_rotation > 360:
            teapot_rotation = 0

pygame.quit()
