import pygame
import moderngl
import numpy
import math

# Start pygame
pygame.init()

# Start clock
clock = pygame.time.Clock()

# Set window title
pygame.display.set_caption("Assignment 3: Rafael Niebles")

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

# Make ModernGL context
mgl_ctx = moderngl.create_context()

# Diamond =====================================================================

diamond_vertices = numpy.array([
    0, 0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Upper triangle
    0, -0.8, 0,
    -0.8, 0, 0,
    0.8, 0, 0,  # Lower triangle
], dtype="f4")  # Each float takes up 4 bytes; float 4; f4! This allows precision adjust

diamond_vbo = mgl_ctx.buffer(diamond_vertices.tobytes())

diamond_vertex_shader_code = """
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

diamond_fragment_shader_code = """
#version 330 core

in vec3 v_color; // Input color from vertex shader
out vec4 f_color; // Final pixel color

void main() {
    f_color = vec4(v_color, 1.0); // Apply final pixel color!
}
"""

diamond_program = mgl_ctx.program(
    vertex_shader=diamond_vertex_shader_code,
    fragment_shader=diamond_fragment_shader_code
)

diamond_vao = mgl_ctx.vertex_array(
    diamond_program,
    # VBO, format of vertex data (3 floats per vertex), desired name of vertex arr.
    [(diamond_vbo, "3f", "position")]
)

# Set the uniform variable for scaling
scale_value = 0.1
diamond_program["scale"].value = scale_value

# Positioning angle for other
angle = 0
angle_increment_per_second = 6

# Line ========================================================================

line_vertices = numpy.array([
    0, 0, 0,  # Start
    0, 0.5, 0,  # Finish
], dtype="f4")

line_vbo = mgl_ctx.buffer(line_vertices.tobytes())

line_vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 position;

void main() {
    gl_Position = vec4(position, 1.0);
}
"""

line_fragment_shader_code = """
#version 330 core
out vec4 fragColor;

void main() {
    fragColor = vec4(0.0, 0.0, 0.0, 0.0);
}
"""

line_program = mgl_ctx.program(
    vertex_shader=line_vertex_shader_code,
    fragment_shader=line_fragment_shader_code
)

line_vao = mgl_ctx.vertex_array(
    line_program,
    [(line_vbo, "3f", "position")]
)

# Main Loop ===================================================================

# Framerate
fps = 60

# BG color
clear_color_r_norm = 115/255
clear_color_g_norm = 115/255
clear_color_b_norm = 115/255
clear_color_a_norm = 1

# Displacement of line and diamond
displacement = 0.5

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # OpenGL bypasses Pygame at this point; specify clear color to MGL ourselves
    mgl_ctx.clear(color=(clear_color_r_norm, clear_color_g_norm,
                  clear_color_b_norm, clear_color_a_norm))

    # Draw without displacement
    diamond_program["dx"].value = 0
    diamond_program["dy"].value = 0

    diamond_vao.render(moderngl.TRIANGLES)

    # Draw with displacement and angling
    dt = clock.tick(fps) / 1000
    angle -= angle_increment_per_second * dt
    diamond_program["angle"].value = angle

    diamond_program["dx"].value = displacement
    diamond_program["dy"].value = displacement

    diamond_vao.render(moderngl.TRIANGLES)

    # Adjust position of end vertex (placed at center of orbiting rect)
    line_vertices[3] = displacement * math.cos(math.radians(angle))
    line_vertices[4] = displacement * math.sin(math.radians(angle))

    # Write new vertex pos
    line_vbo.write(line_vertices)

    # Render line
    line_vao.render(moderngl.LINES, vertices=2)

    # Switch to back buffer
    pygame.display.flip()

pygame.quit()
