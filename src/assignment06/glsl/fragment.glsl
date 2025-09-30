#version 330

// Input from vertex shader (interpolated)
in vec2 v_texcoord;

// Uniform for the texture sampler
uniform sampler2D u_texture;

// Final output color for the pixel
out vec4 f_color;

void main() {
    // Look up the color from the texture using the coordinate
    f_color = texture(u_texture, v_texcoord);
}
