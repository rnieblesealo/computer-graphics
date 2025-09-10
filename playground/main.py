import pygame
import moderngl
import numpy as np

# Start pygame
pygame.init()

# Start clock
clock = pygame.time.Clock()

# Set window title
pygame.display.set_caption("ModernGL Window")

# We will use OpenGL rendering to Pygame window
# Mac requires requesting OpenGL context + enabling forward compat (Metal)
# https://stackoverflow.com/questions/76151435/creating-a-context-utilizing-moderngl-for-pygame
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Initialize display with double buffer and OpenGL rendering mode
screen_width = 500
screen_height = 500
display = pygame.display.set_mode((screen_width, screen_height),
                                  pygame.DOUBLEBUF | pygame.OPENGL)

# === SHADER STUFF ==================================================

# Make ModernGL context
ctx = moderngl.create_context()

# Create VBO
diamond = np.array([
    0, 0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Upper triangle
    0, -0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Lower triangle
], dtype="f4")  # 4-place precision float

vbo = ctx.buffer(diamond.tobytes())

# Get shader code
with open("diamond-vertex.glsl", "r") as file:
    diamond_vertex_shader_code = file.read()

with open("diamond-fragment.glsl", "r") as file:
    diamond_fragment_shader_code = file.read()


# Compile into program
program = ctx.program(
    vertex_shader=diamond_vertex_shader_code,
    fragment_shader=diamond_fragment_shader_code
)

# Create VAO
float_precision = "3f"
vertex_position_varname = "position"
vao = ctx.vertex_array(
    program,
    [(vbo, float_precision, vertex_position_varname)]
)

# Set uniform variables...

# Ready!

# === MAIN LOOP =====================================================

# WARNING: We made it so OpenGL bypasses Pygame draw calls
# Anything we draw with Pygame will be ignored

target_fps = 60

angle = 0
angle_increment_per_second = 12

running = True
while running:
    # Quit mechanism
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear background
    ctx.clear(color=(20 / 255, 20 / 255, 20 / 255, 1.0))

    # Increment angle
    dt = clock.tick(target_fps) / 1000
    angle += dt * angle_increment_per_second
    program["angle"].value = angle

    # Issue a render call; this sets the VAO active, etc
    vao.render(moderngl.TRIANGLES)

    # Swap buffer
    pygame.display.flip()
pygame.quit()
