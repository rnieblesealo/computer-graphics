#version 330 core

in vec2 position; // Input position

// Use these values... (Cont'd here!)
uniform float scale;
uniform float distance;
uniform mat2 m;

void main() {
    vec2 displacement = vec2(distance, distance);
    vec2 p = m * position * scale + displacement;

    // Apply correction, etc. here
    gl_Position = vec4(p, 0, 1);
}
