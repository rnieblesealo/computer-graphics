#!/home/rniebles/Developer/GRAPHICS/.cgenv/bin/python3

import pygame
import moderngl
import numpy
import math

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
                                  pygame.DOUBLEBUF | pygame.OPENGL)

# === ModernGL Setup ==========================================================

mgl_ctx = moderngl.create_context()

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

# === Diamond =================================================================

# Load up verts
diamond_vertices = numpy.array([
    0, 0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Upper triangle
    0, -0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Lower triangle
], dtype="f4")  # Precision 4 float

# Make VBO
diamond_vbo = mgl_ctx.buffer(diamond_vertices.tobytes())

# Make shader program
diamond_program = mgl_ctx.program(
    vertex_shader=diamond_vertex_shader_code,
    fragment_shader=diamond_fragment_shader_code
)

# Make VAO
diamond_vao = mgl_ctx.vertex_array(
    diamond_program,
    [(diamond_vbo, "3f", "position")]
)

# Scale the diamond
scale_value = 0.1
diamond_program["scale"].value = scale_value

# Initialize positioning-related uniform variables
angle = 0
angle_increment_per_second = 6

# === Line ====================================================================

# Pretty much the same stuff as the diamond but for the line

line_vertices = numpy.array([
    0, 0, 0,  # Start
    0, 0.5, 0,  # Finish
], dtype="f4")

line_vbo = mgl_ctx.buffer(line_vertices.tobytes())

line_program = mgl_ctx.program(
    vertex_shader=line_vertex_shader_code,
    fragment_shader=line_fragment_shader_code
)

line_vao = mgl_ctx.vertex_array(
    line_program,
    [(line_vbo, "3f", "position")]
)

# === Main Loop ===============================================================

# WARNING: Pygame calls will be ignored in favor of ModernGL calls

# Framerate
fps = 60

# Displacement of line and diamond
displacement = 0.5

running = True
while running:
    # Quit mechanism
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear display
    mgl_ctx.clear(color=(15 / 255, 15 / 255, 15 / 255, 0))

    # Draw diamond without displacement
    diamond_program["dx"].value = 0
    diamond_program["dy"].value = 0

    diamond_vao.render(moderngl.TRIANGLES)

    # Draw diamond with displacement and angling
    dt = clock.tick(fps) / 1000
    angle += angle_increment_per_second * dt
    diamond_program["angle"].value = angle

    diamond_program["dx"].value = displacement
    diamond_program["dy"].value = displacement

    diamond_vao.render(moderngl.TRIANGLES)

    # Adjust position of end vertex (placed at center of orbiting rect)
    line_vertices[3] = displacement * math.cos(math.radians(angle))
    line_vertices[4] = displacement * math.sin(math.radians(angle))

    line_vbo.write(line_vertices)

    # Render line
    line_vao.render(moderngl.LINES, vertices=2)

    # Switch to back buffer
    pygame.display.flip()

pygame.quit()
