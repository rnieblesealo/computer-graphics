import pygame
import moderngl
import math
import numpy as np
from PIL import Image

# === Pygame Setup ============================================================

# Start pygame
pygame.init()

# Start clock
clock = pygame.time.Clock()

# Set window title
pygame.display.set_caption("Assignment 4: Rafael Niebles")

# Enable forward compat; request context
# https://stackoverflow.com/questions/76151435/creating-a-context-utilizing-moderngl-for-pygame
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Initialize window
screen_width = 500
screen_height = 500
display = pygame.display.set_mode((screen_width, screen_height),
                                  pygame.DOUBLEBUF |
                                  pygame.OPENGL |
                                  pygame.RESIZABLE)  # Make resizable

# === ModernGL Setup ==========================================================

ctx = moderngl.create_context()

# Load diamond vertex shader
with open("glsl/assignment04-diamond-vertex.glsl", "r") as shaderfile:
    diamond_vertex_shader_code = shaderfile.read()

# Load diamond fragment shader
with open("glsl/assignment04-diamond-fragment.glsl", "r") as shaderfile:
    diamond_fragment_shader_code = shaderfile.read()

# Load line vertex shader
with open("glsl/assignment04-line-vertex.glsl", "r") as shaderfile:
    line_vertex_shader_code = shaderfile.read()

# Load line fragment shader
with open("glsl/assignment04-line-fragment.glsl", "r") as shaderfile:
    line_fragment_shader_code = shaderfile.read()

# Load image fragment shader
with open("glsl/assignment05/image-fragment.glsl", "r") as shaderfile:
    image_fragment_shader_code = shaderfile.read()

# Load image vertex shader
with open("glsl/assignment05/image-vertex.glsl", "r") as shaderfile:
    image_vertex_shader_code = shaderfile.read()

# === Image Rendering Setup ===================================================

# Setup fullscreen quad with 2 tris

# Setup verts
cf_verts = np.array([
    0, 0,  # Bottom left
    0, 1,  # Top left
    1, 0,  # Bottom right
    1, 1   # Top right
], dtype=np.float32)

# VBO setup
cf_vbo = ctx.buffer(cf_verts.tobytes())

# VAO setup
cf_program = ctx.program(
    vertex_shader=image_vertex_shader_code,
    fragment_shader=image_fragment_shader_code
)
cf_vao = ctx.vertex_array(cf_program, cf_vbo, "vert")

# Load image
cf_img = Image.open(
    "./assets/clockface.png").convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)

# Create texture for it
cf_img_channels = 3  # RGB
cf_texture = ctx.texture(cf_img.size, cf_img_channels, cf_img.tobytes())

# === Diamond Setup ===========================================================

# Load up verts
diamond_vertices = np.array([
    -0.8, 0, 0,
    0, 0.8, 0,
    0.8, 0, 0,  # Upper triangle
    -0.8, 0, 0,
    0, -0.8, 0,
    0.8, 0, 0,  # Lower triangle
], dtype="f4")  # Precision 4 float

# Make VBO
diamond_vbo = ctx.buffer(diamond_vertices.tobytes())

# Make shader program
diamond_program = ctx.program(
    vertex_shader=diamond_vertex_shader_code,
    fragment_shader=diamond_fragment_shader_code
)

# Make VAO
diamond_vao = ctx.vertex_array(
    diamond_program,
    [(diamond_vbo, "3f", "position")]
)

# === Line Setup ==============================================================

# Pretty much the same stuff as the diamond but for the line

line_vertices = np.array([
    0, 0, 0,  # Start
    1, 1, 0,  # Finish
], dtype="f4")

line_vbo = ctx.buffer(line_vertices.tobytes())

line_program = ctx.program(
    vertex_shader=line_vertex_shader_code,
    fragment_shader=line_fragment_shader_code
)

line_vao = ctx.vertex_array(
    line_program,
    [(line_vbo, "3f", "position")]
)

# === Main Setup ==============================================================

ANGLE_INCREMENT_PER_SECOND = 6
ANGLE_START_OFFSET = 90  # Start at 12 o'clock
DIAMOND_SCALE = 0.1
FPS = 60
ORBIT_DISTANCE = 0.8
CLEAR_COLOR = (124 / 255, 135 / 255, 3 / 255, 0)

angle = 0
curr_width = screen_width
curr_height = screen_height

# Initialize constant uniform variables
diamond_program["scale"].value = DIAMOND_SCALE
line_program["distance"].value = ORBIT_DISTANCE

# === Main Loop ===============================================================

# WARNING: Pygame calls will be ignored in favor of ModernGL calls

running = True
while running:

    # ==== Update ===================================

    for event in pygame.event.get():
        # Quit mechanism
        if event.type == pygame.QUIT:
            running = False
        # HACK: display.get_size() doesn't report back correct screen size
        # This does though!
        elif event.type == pygame.WINDOWRESIZED:
            curr_width = event.x
            curr_height = event.y

    # Compute angle
    dt = clock.tick(FPS) / 1000
    angle += ANGLE_INCREMENT_PER_SECOND * dt

    # Aspect-corrective
    aspect_ratio = curr_width / curr_height

    m_scaling = np.array([
        [1/aspect_ratio, 0],
        [0, 1]
    ], order="F")

    # Rotation
    a = math.radians(angle)
    m_rotation = np.array([
        [math.cos(a), -math.sin(a)],
        [math.sin(a), math.cos(a)]
    ], order="F")  # NOTE: F = Fortran order; column-major memory layout

    # Aspect correction + rotation
    m = m_rotation @ m_scaling  # NOTE: @ = Matrix mult. operator
    m = tuple(m.flatten())

    # Pass transformation matrix
    diamond_program["m"].value = m
    line_program["m"].value = m

    # ==== Render ===================================

    # Clear display
    ctx.clear(color=CLEAR_COLOR)

    # Render clock face bg
    cf_texture.use(0)  # TODO: What does the 0 stand for?
    cf_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    # Diamond at center
    diamond_program["distance"].value = 0
    diamond_vao.render(moderngl.TRIANGLES)

    # Diamond that orbits
    diamond_program["distance"].value = ORBIT_DISTANCE
    diamond_vao.render(moderngl.TRIANGLES)

    # Line
    line_vao.render(moderngl.LINES, vertices=2)

    # Switch to back buffer
    pygame.display.flip()

    # ===============================================

pygame.quit()
