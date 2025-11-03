import glm
import moderngl
import numpy
import pygame
import random

from loadModelUsingAssimp_V3 import create3DAssimpObject
from pathlib import Path

# ---------------------------------------------------------------------
# CONFIG / ASSETS
# ---------------------------------------------------------------------

FLOOR_TEX = Path("./floor-wood.jpg")
MODEL_PATH = Path("./mario_obj/scene.gltf")
TEXTURE_PATH = Path("./mario_obj/textures/")

# ---------------------------------------------------------------------
# WINDOW SETUP
# ---------------------------------------------------------------------

WIDTH = 1280
HEIGHT = 800

pygame.init()

pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
)

pygame.display.set_mode(
    (WIDTH, HEIGHT), flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
)

pygame.display.set_caption("Assignment 10: Rafael Niebles")

# ---------------------------------------------------------------------
# MODERNGL SETUP
# ---------------------------------------------------------------------

gl = moderngl.get_context()
gl.viewport = (0, 0, WIDTH, HEIGHT)
gl.enable(gl.DEPTH_TEST)

# ---------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------

modelObj = create3DAssimpObject(str(MODEL_PATH))

# ---------------------------------------------------------------------
# FLOOR SETUP
# ---------------------------------------------------------------------

size = 25.0

floor_positions = numpy.array(
    [
        [-size / 2, 0.0, -size / 2],
        [size / 2, 0.0, -size / 2],
        [size / 2, 0.0, size / 2],
        [size / 2, 0.0, size / 2],
        [-size / 2, 0.0, size / 2],
        [-size / 2, 0.0, -size / 2],
    ],
    dtype="float32",
)

floor_uvs = numpy.array(
    [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 4.0],
        [4.0, 4.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ],
    dtype="float32",
)

floor_geometry = (
    numpy.concatenate((floor_positions, floor_uvs),
                      axis=1).flatten().astype("float32")
)

# ---------------------------------------------------------------------
# SHADER SETUP
# ---------------------------------------------------------------------

FLOOR_SHADER_VERTEX = """
#version 330 core
layout (location=0) in vec3 position;
layout (location=1) in vec2 uv;

uniform mat4 view, perspective, M;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;

void main() {
    f_uv = uv;
    vec4 P = M * vec4(position, 1.0);
    f_position = P.xyz;
    gl_Position = perspective * view * P;
    f_normal = vec3(0.0, 1.0, 0.0);  // Upwards normal for XZ floor
}
"""

FLOOR_SHADER_FRAGMENT = """
#version 330 core
in vec2 f_uv;
in vec3 f_normal;
in vec3 f_position;

uniform sampler2D map;
uniform vec3 light;

out vec4 out_color;

void main() {
    vec3 L = normalize(light);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = max(dot(N, L), 0.0);
    vec3 ambient = 0.1 * materialColor;
    vec3 color = ambient + materialColor * NdotL;
    out_color = vec4(color, 1.0);
}
"""

MARIO_SHADER_VERTEX = """
#version 330 core
layout (location=0) in vec3 position;
layout (location=1) in vec3 normal;
layout (location=2) in vec2 uv;

uniform mat4 model;
uniform mat4 view, perspective;

out vec2 f_uv;
out vec3 f_normal;
out vec3 f_position;

void main() {
    f_uv = uv;
    vec4 P = model * vec4(position, 1.0);
    f_position = P.xyz;
    gl_Position = perspective * view * P;

    mat3 normalMatrix = mat3(transpose(inverse(model)));
    f_normal = normalize(normalMatrix * normal);
}
"""

MARIO_SHADER_FRAGMENT = """
#version 330 core
in vec2 f_uv;
in vec3 f_normal;
in vec3 f_position;

uniform sampler2D map;
uniform vec3 light;
uniform float shininess;
uniform vec3 eye_position;
uniform vec3 k_diffuse;

out vec4 out_color;

void main() {
    vec3 L = normalize(light);
    vec3 materialColor = texture(map, f_uv).rgb;
    vec3 N = normalize(f_normal);
    float NdotL = max(dot(N, L), 0.0);
    vec3 ambient = 0.1 * materialColor;
    vec3 color = ambient;

    if (NdotL > 0.0){
        vec3 diffuse = materialColor * NdotL;
        vec3 V = normalize(eye_position - f_position);
        vec3 H = normalize(L + V);
        vec3 specular = (shininess > 0.0) ? vec3(pow(max(dot(N, H), 0.0), shininess)) : vec3(0.0);
        color += k_diffuse * diffuse + specular;
    }
    out_color = vec4(color, 1.0);
}
"""

# ---------------------------------------------------------------------
# CREATE SHADER PROGRAMS
# ---------------------------------------------------------------------

model_program = gl.program(
    vertex_shader=MARIO_SHADER_VERTEX, fragment_shader=MARIO_SHADER_FRAGMENT
)

modelObj.createRenderableAndSampler(model_program)

floor_program = gl.program(
    vertex_shader=FLOOR_SHADER_VERTEX, fragment_shader=FLOOR_SHADER_FRAGMENT
)

# ---------------------------------------------------------------------
# FLOOR VAO AND TEXTURING
# ---------------------------------------------------------------------

floor_renderable = gl.vertex_array(
    floor_program, [(gl.buffer(floor_geometry), "3f 2f", "position", "uv")]
)

tex_img = pygame.image.load(str(FLOOR_TEX))
tex_data = pygame.image.tobytes(tex_img, "RGB", True)
floor_tex = gl.texture(tex_img.get_size(), components=3, data=tex_data)
floor_tex.build_mipmaps()
floor_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
floor_tex.repeat_x = True
floor_tex.repeat_y = True
floor_sampler = gl.sampler(texture=floor_tex)

# ---------------------------------------------------------------------
# TRANSFORMING MODEL, PLACING GRID
# ---------------------------------------------------------------------

GRID_N = 10  # 10x10 grid
FLOOR_SIZE = 25.0
GRID_DELTA = FLOOR_SIZE / GRID_N
OBJ_SIZE = 0.75 * GRID_DELTA  # model should fill 75% of a cell
JITTER_AMPLITUDE = 0.35 * GRID_DELTA  # random offset inside cell

# Compute Mario model bounds
min_corner, max_corner = modelObj.bound.boundingBox

base_center = glm.vec3(
    (min_corner.x + max_corner.x) / 2.0,
    min_corner.y,
    (min_corner.z + max_corner.z) / 2.0,
)

base_w = max_corner.x - min_corner.x
base_h = max_corner.z - min_corner.z
base_max = max(base_w, base_h)
scale_factor = (OBJ_SIZE / base_max) if base_max > 1e-6 else 1.0

# Base transforms
T_base = glm.translate(glm.mat4(1.0), -base_center)
S_fit = glm.scale(glm.mat4(1.0), glm.vec3(scale_factor))

# Jittered grid centers
startX = -FLOOR_SIZE / 2.0 + GRID_DELTA * 0.5
startZ = -FLOOR_SIZE / 2.0 + GRID_DELTA * 0.5

rng = random.Random(42)  # deterministic jitter pattern
instance_transforms = []

for gx in range(GRID_N):
    for gz in range(GRID_N):
        cx = startX + gx * GRID_DELTA + \
            rng.uniform(-JITTER_AMPLITUDE, JITTER_AMPLITUDE)
        cz = startZ + gz * GRID_DELTA + \
            rng.uniform(-JITTER_AMPLITUDE, JITTER_AMPLITUDE)

        # Random rotation around Y axis
        yaw_deg = rng.uniform(-30.0, 30.0)
        R_yaw = glm.rotate(glm.mat4(1.0), glm.radians(
            yaw_deg), glm.vec3(0.0, 1.0, 0.0))

        # Translation to jittered cell center
        T_cell = glm.translate(glm.mat4(1.0), glm.vec3(cx, 0.0, cz))

        # Compose transform (translate * rotate * scale * base align)
        M = T_cell * R_yaw * S_fit * T_base
        instance_transforms.append(M)


# ---------------------------------------------------------------------
# CAMERA AND LIGHTING SETUP
# ---------------------------------------------------------------------

fov = glm.radians(50.0)
near, far = 0.1, 200.0
perspectiveMat = glm.perspective(fov, WIDTH / HEIGHT, near, far)

cam_angle = 0.0
cam_radius = 22.0
cam_height = 18.0
light_angle = 90.0

clock = pygame.time.Clock()
running = True
pause = False

# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pause = not pause
        elif event.type == pygame.WINDOWRESIZED:
            WIDTH, HEIGHT = event.x, event.y
            gl.viewport = (0, 0, WIDTH, HEIGHT)
            perspectiveMat = glm.perspective(fov, WIDTH / HEIGHT, near, far)

    # Camera orbit
    if not pause:
        cam_angle = (cam_angle + 20.0 * (1 / 60.0)) % 360.0
    rotY = glm.rotate(glm.mat4(1.0), glm.radians(cam_angle), glm.vec3(0, 1, 0))
    eye = glm.vec3(rotY * glm.vec4(cam_radius, cam_height, 0.0, 1.0))
    viewMat = glm.lookAt(eye, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    # Light
    rotY_light = glm.rotate(glm.mat4(1.0), glm.radians(
        light_angle), glm.vec3(0, 1, 0))
    light_dir = (
        glm.normalize(
            glm.vec3(rotY_light * glm.vec4(-0.3, -6.0, -0.8, 0.0))) * 2.0
    )

    # Clear
    gl.clear(0.05, 0.05, 0.07, 1.0, 1.0)

    LIGHT_BOOST = 100

    # Draw floor
    floor_program["view"].write(viewMat)
    floor_program["perspective"].write(perspectiveMat)
    floor_program["M"].write(glm.mat4(1.0))
    floor_program["light"].write(light_dir + LIGHT_BOOST)
    floor_sampler.use(0)
    floor_program["map"] = 0
    floor_renderable.render()

    # --- Draw all Mario instances ---
    mp = modelObj.program

    mp["view"].write(viewMat)
    mp["perspective"].write(perspectiveMat)
    mp["light"].write(light_dir + LIGHT_BOOST)
    mp["shininess"].value = 32.0
    mp["k_diffuse"].write(glm.vec3(1.0, 1.0, 1.0))
    mp["eye_position"].write(eye)

    for M in instance_transforms:
        modelObj.render(M)

    pygame.display.flip()
    clock.tick(60)

pygame.display.quit()
