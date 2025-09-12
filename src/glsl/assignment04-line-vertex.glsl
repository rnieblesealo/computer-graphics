// Assignment 4 Line Vertex Shader

#version 330 core

in vec2 position;

uniform float distance;
uniform mat2 m;

void main() {
    vec2 p = position;

    // Operate on the end vertex only
    if (gl_VertexID > 0) {
        vec2 d = distance * vec2(0, 1);
        p = d;
    }

    gl_Position = vec4(m * p, 0, 1);
}
