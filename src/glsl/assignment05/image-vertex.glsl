// Clockface image rendering vertex shader

#version 330

in vec2 in_vert;
in vec2 in_text;
out vec2 text_coord;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    text_coord = in_text;
}
