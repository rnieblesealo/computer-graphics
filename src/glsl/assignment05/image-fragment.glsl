// Clockface image rendering fragment shader

#version 330

uniform sampler2D Texture;

in vec2 text;
out vec4 color;

void main() {
    color = texture(Texture, text);
}
