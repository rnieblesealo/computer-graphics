// Clockface image rendering vertex shader

#version 330

in vec2 vert;
out vec2 text; // Texture coords

void main() {
    gl_Position = vec4(vert * 2.0 - 1.0, 0.0, 1.0);
    text = vert;
}
