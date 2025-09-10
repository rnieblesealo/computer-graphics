#version 330 core

layout(location = 0) in vec3 position;

uniform float xCorrectionFactor;
uniform float yCorrectionFactor;

void main() {
    // Apply corrective scaling
    vec3 updatedPosition = vec3(
            position.x * xCorrectionFactor,
            position.y * yCorrectionFactor,
            0);

    gl_Position = vec4(updatedPosition, 1.0);
}
