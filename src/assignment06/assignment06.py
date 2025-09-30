# TODO: What is moderngl.DEPTH_TEST? A: Enables Z-buffer
# TODO: What is "in_texcoord_0"? Is it our UVs?
# TODO: I don't understand perspective + view matrices
# TODO: What is a model matrix?
# TODO: How is vertex shader interpolated?
# TODO: What is texture sampler?
# TODO: Is creating program = compiling shaders?
# TODO: What is bounding info?

import glm
import math
import moderngl
import pygame


# Import the provided helper function from your file
from LoadObject import getObjectData
from PIL import Image


class GraphicsEngine:
    def __init__(self, width=800, height=600):
        # Initialize Pygame
        pygame.init()

        # Enable forward compat; request context
        # https://stackoverflow.com/questions/76151435/creating-a-context-utilizing-moderngl-for-pygame
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        # We can set member vars like this
        self.WIDTH, self.HEIGHT = width, height

        pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                                pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Assignment 6: Rotating Camera")

        # Create a ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)  # Important for 3D rendering
        self.ctx.clear_color = (0.0, 0.0, 0.0)  # Black background

        # Load the shaders
        with open("./glsl/vertex.glsl") as file:
            self.vertex_shader = file.read()

        with open("./glsl/fragment.glsl") as file:
            self.fragment_shader = file.read()

        # Create the shader program from the GLSL source code
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )

        # Load vertex data + bounding info (this returns a list)
        try:
            shape_data, self.bounds = getObjectData(
                "./assets/teapot.obj", texture=True)

            # Load the texture image using Pillow
            texture_image = Image.open('./assets/brick.jpg').convert('RGB')
            self.texture = self.ctx.texture(
                texture_image.size, 3, texture_image.tobytes())

        except FileNotFoundError as e:
            print(f"Error: Could not find a required file: {e.filename}")
            pygame.quit()
            quit()

        # Create a VBO and send the flattened vertex data to the GPU
        vbo = self.ctx.buffer(shape_data)

        # Create a VAO to describe how the VBO data is structured for the shader
        self.vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '3f 2f', 'in_position', 'in_texcoord_0')]
        )

        # WARNING: --------------- BEGIN BLACK MAGIC ------------------------

        # The camera will always look at the center of the object
        self.lookAtPoint = glm.vec3(self.bounds.center)
        self.up_vector = glm.vec3(0, 1, 0)

        # Initial Camera Position Calculation (as per assignment specs):
        # On YZ plane, +60 degrees from Y-axis, 2*radius distance from center
        angle_rad = glm.radians(60)
        distance = 2 * self.bounds.radius

        # Calculate initial position relative to the origin
        initial_cam_pos_relative = glm.vec3(
            0,  # On YZ plane, so X is 0
            distance * math.cos(angle_rad),  # Y component
            distance * math.sin(angle_rad)  # Z component
        )
        # Add the object's center to get the final world position
        self.initial_cam_pos = initial_cam_pos_relative + self.lookAtPoint

        # Create the Perspective Matrix
        fov = 45.0
        aspect_ratio = self.WIDTH / self.HEIGHT
        near_plane = self.bounds.radius
        far_plane = 3 * self.bounds.radius

        perspective_matrix = glm.perspective(
            glm.radians(fov), aspect_ratio, near_plane, far_plane)

        # WARNING: --------------- END BLACK MAGIC --------------------------

        # Get references to the uniform variables in the shader program
        self.view_uniform = self.program['view']
        self.perspective_uniform = self.program['perspective']
        self.texture_uniform = self.program['u_texture']

        # Set the uniforms that will not change during the render loop
        self.perspective_uniform.write(perspective_matrix)
        self.texture_uniform.value = 0  # Tell the shader to use texture unit 0

    def run(self):
        running = True
        while running:
            # Handle Pygame events (like closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get current time in seconds for smooth animation
            time = pygame.time.get_ticks() / 1000.0

            # WARNING: --------------- BEGIN BLACK MAGIC ------------------------

            # The camera will orbit around the world's Y-axis
            rotation_angle = time * 0.5  # You can change 0.5 to alter rotation speed

            # Create a rotation matrix for the Y-axis
            rotation_matrix = glm.rotate(
                glm.mat4(1.0), rotation_angle, glm.vec3(0, 1, 0))

            # To rotate around the teapot's center, we translate the camera's initial
            # position to the origin, apply the rotation, and then translate it back.
            cam_pos_relative_to_center = self.initial_cam_pos - self.lookAtPoint
            rotated_cam_pos = glm.vec3(
                rotation_matrix * glm.vec4(cam_pos_relative_to_center, 1.0))
            current_eye_point = rotated_cam_pos + self.lookAtPoint

            # Recalculate the view matrix for every frame using the new camera position
            view_matrix = glm.lookAt(
                current_eye_point, self.lookAtPoint, self.up_vector)

            # WARNING: --------------- END BLACK MAGIC --------------------------

            # Update the 'view' uniform in the shader with the new matrix
            self.view_uniform.write(view_matrix)

            # --- Render the Scene ---
            # Clear the color and depth buffers from the previous frame
            self.ctx.clear()

            # Bind the texture to texture unit 0
            self.texture.use(0)

            # Render the VAO (which represents the teapot)
            self.vao.render()

            # Swap the front and back buffers to display the rendered frame
            pygame.display.flip()

        # Cleanup resources
        pygame.quit()


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()
