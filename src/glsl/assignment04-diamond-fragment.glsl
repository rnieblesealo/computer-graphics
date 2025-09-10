#version 330 core

in vec3 v_color; // Input color from vertex shader
out vec4 f_color; // Final pixel color

void main() {
    f_color = vec4(v_color, 1.0); // Apply final pixel color!
}
