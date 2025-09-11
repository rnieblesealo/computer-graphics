#version 330 core

in vec3 position; // Input position

uniform float angle;
uniform float angleOffset; // Start offset applied to angle
uniform float scale;
uniform vec2 correction;
uniform vec2 displacement;

void main() {
    float angleInRadians = radians(angleOffset + angle);
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    // Rotation about screen center
    mat3 rotation = mat3(
            cosA, 0, 0,
            0, sinA, 0,
            0, 0, 1);

    // Rotation about diamond center
    mat3 localRotation = mat3(
            cosA, sinA, 0,
            sinA, -cosA, 0,
            0, 0, 1);

    vec3 updatedPosition = ((localRotation * position * scale) + (vec3(displacement, 0) * rotation)) * vec3(correction, 0);

    // Apply computed position
    gl_Position = vec4(updatedPosition, 1);
}
