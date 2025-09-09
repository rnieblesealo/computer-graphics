// Vertex 
#version 330 core

in vec3 position; // Input position
out vec3 v_color; // Output color to fragment shader

uniform float scale;

void main() {
    gl_Position = vec4(position * scale, 1.0); // Apply vertex position; in this case we leave as it was on input
    v_color = vec3(1.0, 1.0, 0.0); // Set fragment color to yellow, as specified in assignment; the fragment shader will receive this value
}

// Fragment
#version 330 core

in vec3 v_color; // Input color from vertex shader
out vec4 f_color; // Final pixel color

void main() {
    f_color = vec4(v_color, 1.0); // Apply final pixel color!
}
