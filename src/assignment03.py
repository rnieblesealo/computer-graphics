import pygame
import moderngl
import numpy

# Start pygame
pygame.init()

# Set window title
pygame.display.set_caption("Assignment 2: Rafael Niebles")

# We will use OpenGL rendering to Pygame window
# Doing this on Mac requires requesting an OpenGL context and setting forward compatible flag
# https://stackoverflow.com/questions/76151435/creating-a-context-utilizing-moderngl-for-pygame
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(
    pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Initialize display with double buffer and OpenGL rendering
screen_width = 500
screen_height = 500
display = pygame.display.set_mode((screen_width, screen_height),
                                  pygame.DOUBLEBUF | pygame.OPENGL)

# Init clock
clock = pygame.time.Clock()

# Make ModernGL context
mgl_ctx = moderngl.create_context()

# Define vertices
# Assignment specifies:
# - [x] 2 triangles, upper and lower to draw diamond
# - [x] 0.8 center offset
# - [x] 0 z-position
vertices = numpy.array([
    0, 0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Upper triangle
    0, -0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Lower triangle
], dtype="f4")  # Each float takes up 4 bytes; float 4; f4! This allows us to adjust precision

# Create vertex buffer from vertex data
# A vertex buffer stores vertices
# Good practice to convert to bytes since this is how OpenGL expects them
vbo = mgl_ctx.buffer(vertices.tobytes())

# Vertex shader does things to vertex data!
# This is setting all vertex positions to exactly match the input
vertex_shader_code = """
#version 330 core

in vec3 position; // Input position
out vec3 v_color; // Output color to fragment shader

uniform float scale;
uniform float angle;
uniform float dx;
uniform float dy;

void main() {
    float angleInRadians = radians(angle);
    float dxAngled = dx * cos(angleInRadians);
    float dyAngled = dy * sin(angleInRadians);

    vec3 updatedPosition = (position * scale) + vec3(dxAngled, dyAngled, 0);

    // Apply vertex position; in this case we leave as it was on input
    gl_Position = vec4(updatedPosition, 1.0);

    // Set fragment color to yellow, as specified in assignment
    // The fragment shader will receive this value
    v_color = vec3(1.0, 1.0, 0.0);
}
"""

# This shader applies the fragment color
fragment_shader_code = """
#version 330 core

in vec3 v_color; // Input color from vertex shader
out vec4 f_color; // Final pixel color

void main() {
    f_color = vec4(v_color, 1.0); // Apply final pixel color!
}
"""

program = mgl_ctx.program(
    vertex_shader=vertex_shader_code,
    fragment_shader=fragment_shader_code
)

# Create vertex array object
# It encapsulates state needed to send vertex data to graphics pipeline
vao = mgl_ctx.vertex_array(
    program,
    [(vbo, "3f", "position")]  # Specify VBO, format of vertex data (3 floats per vertex) and name of vertex data array (we use "position") and name of vertex data array (we use "position") and name of vertex data array (we use "position") and name of vertex data array (we used name "position")
)

# Set the uniform variable for scaling
scale_value = 0.1
program["scale"].value = scale_value

# Main loop!

# BG clear color
# Needs to be between 0-1 to match OpenGL expectation
# The 0-255 RGB values were colorpicked from assignment sample image
clear_color_r_norm = 118/255
clear_color_g_norm = 115/255
clear_color_b_norm = 24/255
clear_color_a_norm = 1

fps = 60
angle = 0
angle_increment_per_second = 6

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # OpenGL bypasses Pygame at this point, so we must specify clear color to the ModernGL context
    mgl_ctx.clear(color=(clear_color_r_norm, clear_color_g_norm,
                  clear_color_b_norm, clear_color_a_norm))

    # Draw without displacement
    program["dx"].value = 0
    program["dy"].value = 0

    vao.render(moderngl.TRIANGLES)

    # Draw with displacement and angling
    dt = clock.tick(fps) / 1000
    angle += angle_increment_per_second * dt
    program["angle"].value = angle

    program["dx"].value = 0.5
    program["dy"].value = 0.5

    vao.render(moderngl.TRIANGLES)

    # Switch to back buffer
    pygame.display.flip()

pygame.quit()
