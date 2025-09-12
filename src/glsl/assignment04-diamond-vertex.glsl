// Assignment 4 Diamond Vertex Shader

#version 330 core

in vec2 position;

uniform float scale;
uniform float distance;
uniform mat2 m;

void main() {
    vec2 d = distance * vec2(0, 1);
    vec2 p = position * scale + d;

    // Apply correction, etc. here
    gl_Position = vec4(m * p, 0, 1);
}
