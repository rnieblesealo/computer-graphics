import pygame
import moderngl
import numpy

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

# === Line ====================================================================

# Pretty much the same stuff as the diamond but for the line

line_vertices = numpy.array([
    0, 0, 0,  # Start
    1, 1, 0,  # Finish
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

# === Main Setup ==============================================================

ANGLE_INCREMENT_PER_SECOND = 6
ANGLE_START_OFFSET = 90  # Start at 12 o'clock
DIAMOND_SCALE = 0.1
FPS = 60
ORBIT_DISTANCE = 0.5

angle = 0
curr_width = screen_width
curr_height = screen_height

# Initialize unchanging uniform variables
diamond_program["angleOffset"].value = ANGLE_START_OFFSET
diamond_program["scale"].value = DIAMOND_SCALE

line_program["angleOffset"].value = ANGLE_START_OFFSET
line_program["dx"].value = ORBIT_DISTANCE
line_program["dy"].value = ORBIT_DISTANCE

# === Main Loop ===============================================================

# WARNING: Pygame calls will be ignored in favor of ModernGL calls

running = True
while running:
    for event in pygame.event.get():
        # Quit mechanism
        if event.type == pygame.QUIT:
            running = False
        # HACK: display.get_size() doesn't report back correct screen size
        # This does though!
        elif event.type == pygame.VIDEORESIZE:
            curr_width = event.w
            curr_height = event.h

    # Calculate corrective scaling factor; new / old
    x_correction_factor = screen_width / curr_width
    y_correction_factor = screen_height / curr_height

    # Calculate angle
    dt = clock.tick(FPS) / 1000
    angle -= ANGLE_INCREMENT_PER_SECOND * dt  # Subtraction = clockwise

    # Clear display
    mgl_ctx.clear(color=(124 / 255, 135 / 255, 3 / 255, 0))

    # Pass it to the program(s)
    diamond_program["xCorrectionFactor"].value = x_correction_factor
    diamond_program["yCorrectionFactor"].value = y_correction_factor

    line_program["xCorrectionFactor"].value = x_correction_factor
    line_program["yCorrectionFactor"].value = y_correction_factor

    # Apply same angle to both diamonds and the line
    line_program["angle"].value = angle
    diamond_program["angle"].value = angle

    # Diamond at center
    diamond_program["dx"].value = 0
    diamond_program["dy"].value = 0

    diamond_vao.render(moderngl.TRIANGLES)

    # Diamond that orbits
    diamond_program["dx"].value = ORBIT_DISTANCE
    diamond_program["dy"].value = ORBIT_DISTANCE

    diamond_vao.render(moderngl.TRIANGLES)

    # Render line
    line_vao.render(moderngl.LINES, vertices=2)

    # Switch to back buffer
    pygame.display.flip()
