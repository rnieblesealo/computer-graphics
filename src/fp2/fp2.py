import glm
import moderngl
import numpy
import pygame

from OpenGL.GL import *
from loadModelUsingAssimp_V3 import create3DAssimpObject
from pathlib import Path

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
pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)

pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
)
pygame.display.set_caption(title="Project Substitute 2: Rafael Niebles")

# ===============================================================================================================
# MODERNGL SETUP
# ===============================================================================================================

ctx = moderngl.get_context()

# ===============================================================================================================
# MODEL SETUP
# ===============================================================================================================

MODEL_FILEPATH = Path("./mario_obj/scene.gltf")

MODEL_VERTEX_SHADER = """
#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec2 f_uv; // Same as uv
out vec3 f_normal; // normal in world coords 
out vec3 f_position;  // position in world coords

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform mat4 model; // Transforms model -> world space
uniform mat4 view; // Transforms world space -> camera space
uniform mat4 perspective; // Transforms 3D coords -> 2D screen coords, creating depth effect
uniform vec4 light; // The light vector (xyz) and type (w)
uniform bool is_shadow_pass;
uniform vec3 plane_point;
uniform vec3 plane_normal;

// =========================================================================================
// MAIN 
// =========================================================================================

void main(){
    vec4 world_pos_4 = model * vec4(position, 1.0);
    vec3 world_pos_3 = world_pos_4.xyz;

    f_position = world_pos_3;
    f_uv = uv;
    mat3 normal_matrix = mat3(transpose(inverse(model)));
    f_normal = normalize(normal_matrix*normal);

    if (!is_shadow_pass) {
        gl_Position = perspective*view*world_pos_4;
    } else {
        vec3 vector_to_light;

        if (light.w > 0.0) { // Point light
            vector_to_light = light.xyz - world_pos_3;
        } else { // Directional light
            vector_to_light = -light.xyz;
        }

        float light_plane_dot = dot(vector_to_light, plane_normal);
        if (abs(light_plane_dot) < 1e-5) {
            gl_Position = perspective * view * world_pos_4;
        } else {
            float t = dot(plane_point - world_pos_3, plane_normal) / light_plane_dot;
            vec3 shadow_pos = world_pos_3 + t * vector_to_light + 0.005 * plane_normal;
            gl_Position = perspective * view * vec4(shadow_pos, 1.0);
        }
    }
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
uniform bool is_shadow_pass;

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
// MAIN
// =========================================================================================

void main(){
    if (is_shadow_pass) {
        out_color = vec4(0.1, 0.1, 0.1, 0.5); 
    } else {
        out_color = vec4(computeColor(), 1.0);
    }
}

"""

# Compile program
model_program = ctx.program(
    vertex_shader=MODEL_VERTEX_SHADER, fragment_shader=MODEL_FRAGMENT_SHADER
)

# Create 3D object
model = create3DAssimpObject(
    MODEL_FILEPATH.as_posix(), verbose=False, textureFlag=True, normalFlag=True
)

# Get texture sampler
model.createRenderableAndSampler(model_program)

# Get bounds
model_bound = model.bound


def render_model(view, perspective, light, eye, is_shadow=False):
    """
    Writes uniforms to the model and then renders it.
    """

    # Write uniforms to model program
    model_program["view"].write(view)
    model_program["perspective"].write(perspective)
    model_program["light"].write(light)
    model_program["eye_position"].write(eye)

    model_program["is_shadow_pass"].value = is_shadow
    if is_shadow:
        # Pass plane parameters needed for shadow projection
        model_program["plane_point"].write(plane_point)
        model_program["plane_normal"].write(plane_normal)

    # Render using loader object function
    model.render()


# ===============================================================================================================
# FLOOR SETUP
# ===============================================================================================================

FLOOR_VERTEX_SHADER = """
#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

in vec3 position;
in vec2 uv;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec2 f_uv; // Same as uv
out vec3 f_position; // Same as position
out vec3 f_normal; // Normalized normals

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform mat4 view;
uniform mat4 perspective;
uniform vec3 normal;

// =========================================================================================
// MAIN
// =========================================================================================

void main(){
  f_position = position;
  f_uv = uv;

  // Normalize normals
  f_normal = normalize(normal);

  // For final pos, apply perspective and convert to final pos 
  gl_Position = perspective * view * vec4(position, 1);
}
"""

FLOOR_FRAGMENT_SHADER = """
#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

in vec2 f_uv;
in vec3 f_position;
in vec3 f_normal;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec4 out_color;

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform vec4 light;
uniform sampler2D map;

// =========================================================================================
// CONSTANT
// =========================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =========================================================================================
// MAIN
// =========================================================================================

void main(){
  vec3 light_vector = normalize(light.xyz); // Assume directional
  
  // Handle point
  if (light.w > 0){
    light_vector = normalize(light.xyz - f_position);
  }

  vec3 normal = normalize(f_normal); // For safety, renormalize

  // Compute surface orient. coefficient 
  float w = dot(normal, UP);

  // Sample the texture
  vec3 material_color = texture(map, f_uv).rgb;
  
  // Compute ambient color
  vec3 ambient_color = 0.1 * (w * SKY_COLOR + (1 - w) * GROUND_COLOR) * material_color;

  // Combine into final color
  vec3 color = ambient_color + material_color * clamp(dot(normal, light_vector), 0, 1);

  out_color = vec4(color, 1);
}
"""

# Compile floor program
floor_program = ctx.program(
    vertex_shader=FLOOR_VERTEX_SHADER, fragment_shader=FLOOR_FRAGMENT_SHADER
)


# Get model bounds, center
m_min = model_bound.boundingBox[0]  # Lowest x, y, z coords of model box (min corner)
m_max = glm.vec3(
    model_bound.boundingBox[1].x, m_min.y, model_bound.boundingBox[1].z
)  # Highest x, y, z (max corner)
m_ctr = (m_min + m_max) / 2  # Center of model bounds

plane_point = m_ctr
plane_normal = glm.vec3(0, 1, 0)  # This is floor plane, so up is its norm

# Define floor plane dimensions
floor_quad_side = 3 * model_bound.radius  # Floor size will be 3 times model's radius
floor_quad_midlength = floor_quad_side / 2  # Floor square edge's midlength

# VBO vertices; they define the 4 corners of the square floor plane
floor_quad_vertices = numpy.array(
    [
        m_ctr.x - floor_quad_midlength,
        m_ctr.y,
        m_ctr.z - floor_quad_midlength,
        0,
        0,
        m_ctr.x + floor_quad_midlength,
        m_ctr.y,
        m_ctr.z - floor_quad_midlength,
        1,
        0,
        m_ctr.x + floor_quad_midlength,
        m_ctr.y,
        m_ctr.z + floor_quad_midlength,
        1,
        1,
        m_ctr.x - floor_quad_midlength,
        m_ctr.y,
        m_ctr.z + floor_quad_midlength,
        0,
        1,
    ]
).astype(numpy.float32)

floor_quad_vbo = ctx.buffer(floor_quad_vertices)

# Make index buffer to define tris
floor_quad_index = numpy.array([0, 1, 2, 2, 3, 0]).astype(numpy.int32)
floor_quad_index_buffer = ctx.buffer(floor_quad_index)

# Create VAO
floor_quad_vao_format = "3f 2f"
floor_quad_vao = ctx.vertex_array(
    floor_program,
    [(floor_quad_vbo, floor_quad_vao_format, "position", "uv")],
    floor_quad_index_buffer,
    index_element_size=4,
)

# Make floor texture
FLOOR_TEXTURE_PATH = Path("./tile-squares-texture.jpg")

floor_texture_img = pygame.image.load(FLOOR_TEXTURE_PATH.as_posix())
floor_texture_data = pygame.image.tobytes(floor_texture_img, "RGB", True)
floor_texture = ctx.texture(
    floor_texture_img.get_size(), data=floor_texture_data, components=3
)

# Make mipmaps
floor_texture.build_mipmaps()

# Create sampler
floor_texture_sampler = ctx.sampler(
    texture=floor_texture,
    filter=(ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR),
    repeat_x=True,
    repeat_y=True,
)


def render_floor(view, perspective, light):
    """
    Writes uniforms to floor shader and then renders it.
    """

    floor_program["view"].write(view)
    floor_program["perspective"].write(perspective)
    floor_program["light"].write(light)
    floor_program["normal"].write(plane_normal)

    floor_texture_sampler.use(0)  # Bind sampler to slot 0
    floor_program["map"] = 0  # Tell shader this

    # Render floor!
    floor_quad_vao.render()


# ===============================================================================================================
# CAMERA PARAMETER SETUP
# ===============================================================================================================

UP = glm.vec3(0, 1, 0)

light_displacement = (
    4 * model_bound.radius * glm.rotateZ(UP, glm.radians(45))
)  # Displaces the light by applying a rotation

eye_displacement = (
    4 * model_bound.radius * glm.rotateX(UP, glm.radians(85))
)  # Displaces the camera by applying a rotation

eye_target_point = glm.vec3(model_bound.center)  # Where camera will point to

fov = glm.radians(30)  # x degree FOV in radians

near_plane = model_bound.radius
far_plane = 20 * model_bound.radius

aspect = SCREEN_WIDTH / SCREEN_HEIGHT
perspective_matrix = glm.perspective(fov, aspect, near_plane, far_plane)

# ===============================================================================================================
# RENDERING
# ===============================================================================================================

TARGET_FPS = 60

clock = pygame.time.Clock()

orbit_rotation = 0
light_angle = 0

is_running = True
is_paused = True
use_blend = False
use_point_light = False
draw_shadow = True

ctx.enable(ctx.DEPTH_TEST)  # Enable Z-buffer

glClearColor(0.0, 0.0, 0.0, 1.0)

model_program["plane_point"].write(plane_point)
model_program["plane_normal"].write(plane_normal)
model_program["is_shadow_pass"].value = False

while is_running:
    for event in pygame.event.get():
        match event.type:
            case pygame.QUIT:
                is_running = False
            case pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    is_paused = not is_paused
                elif event.key == pygame.K_s:
                    draw_shadow = not draw_shadow
                    print(f"Draw Shadow: {draw_shadow}")
                elif event.key == pygame.K_b:
                    use_blend = not use_blend
                    print(f"Use Blend: {use_blend}")
                elif event.key == pygame.K_l:
                    use_point_light = not use_point_light
                    print(f"Use Point Light: {use_point_light}")
                elif event.key == pygame.K_LEFT:
                    light_angle -= 5
                elif event.key == pygame.K_RIGHT:
                    light_angle += 5
            case pygame.WINDOWRESIZED:
                # Recompute aspect and perspective based on aspect ratio change
                new_width = event.x
                new_height = event.y

                ctx.viewport = (0, 0, new_width, new_height)  # Set viewport
                aspect = new_width / new_height

                perspective_matrix = glm.perspective(fov, aspect, near_plane, far_plane)

    # ---------------------------------------------------------------------------------------------------------
    # UPDATES
    # ---------------------------------------------------------------------------------------------------------

    # Update displacements based on updated orbit rotation and light angle
    new_light_displacement = glm.rotate(
        light_displacement, glm.radians(light_angle), UP
    )

    new_eye_displacement = glm.rotate(
        eye_displacement, glm.radians(orbit_rotation), UP
    )  # Use orbit_rotation for camera

    # Define light depending on type
    if use_point_light:
        light = glm.vec4(eye_target_point + new_light_displacement, 1)
    else:
        light = glm.vec4(new_light_displacement, 0)

    # Set camera pos
    eye_point = eye_target_point + new_eye_displacement

    # Compute view matrix (world -> camera space)
    view_matrix = glm.lookAt(eye_point, eye_target_point, UP)

    # Tick
    dt = clock.tick(TARGET_FPS)
    if not is_paused:
        orbit_rotation += 1 * (dt / 16.6)  # Frame-rate independent rotation
        if orbit_rotation > 360:
            orbit_rotation = 0

    # ---------------------------------------------------------------------------------------------------------
    # RENDERING
    # ---------------------------------------------------------------------------------------------------------

    ctx.clear(color=(0, 0, 0))

    if use_blend or use_point_light:
        glEnable(GL_STENCIL_TEST)
        glStencilMask(0xFF)
        glClearStencil(0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
    else:
        glDisable(GL_STENCIL_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glDisable(GL_BLEND)

    render_model(view_matrix, perspective_matrix, light, eye_point)
    render_floor(view_matrix, perspective_matrix, light)

    # Set required OpenGL opts before model rendering with shadow
    if draw_shadow:
        if use_blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_STENCIL_TEST)
            glStencilFunc(GL_EQUAL, 1, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_ZERO)
        elif use_point_light:
            glDisable(GL_BLEND)
            glEnable(GL_STENCIL_TEST)
            glStencilFunc(GL_EQUAL, 1, 0xFF)
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
        else:
            glDisable(GL_BLEND)
            glDisable(GL_STENCIL_TEST)

        # Write shadow uniforms to model and render it in a shadow pass
        model_program["view"].write(view_matrix)
        model_program["perspective"].write(perspective_matrix)
        model_program["eye_position"].write(eye_point)
        model_program["light"].write(light)
        model_program["is_shadow_pass"].value = True
        model.render()
        model_program["is_shadow_pass"].value = False

    pygame.display.flip()

pygame.quit()
