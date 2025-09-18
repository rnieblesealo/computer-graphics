// Clockface image rendering vertex shader

#version 330

uniform sampler2D texture;

in vec2 text_coord;
out vec4 fragColor;

void main() {
    fragColor = texture(texture, text_coord);
}
