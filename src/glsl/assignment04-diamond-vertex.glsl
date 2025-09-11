#version 330 core

in vec3 position; // Input position

// Use these values... (Cont'd here!)
uniform float scale;
uniform float distance;
uniform mat2 m;

// These are my own but I presume I need them here
uniform float angle;
uniform float angleOffset; // Start offset applied to angle

void main() {
    vec3 displacement = vec3(cos(distance), sin(distance), 0);
    vec3 p = position * scale + displacement;

    // Make mat2 into mat3
    mat3 m3 = mat3(m, vec3(0.0, 0.0, 1.0));

    // Apply correction, etc. here
    gl_Position = vec4(m3 * p, 0, 1);
}
