#version 330

// Input vertex attrs. from VBO
in vec3 in_position;
in vec2 in_texcoord_0;

// Uniform matrices passed from Python script
uniform mat4 view;
uniform mat4 perspective;

// Output to frag. shader
out vec2 v_texcoord;

void main() {
    // Transform the vertex position to screen coordinates.
    // Note: No model matrix is needed as the teapot is static and the camera moves.
    gl_Position = perspective * view * vec4(in_position, 1.0);

    // Pass the texture coordinate to the next shader stage
    v_texcoord = in_texcoord_0;
}
