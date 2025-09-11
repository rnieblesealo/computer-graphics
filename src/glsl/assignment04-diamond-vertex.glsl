#version 330 core

in vec3 position; // Input position

uniform float angle;
uniform float angleOffset; // Start offset applied to angle
uniform float scale;
uniform vec3 correction;
uniform vec3 displacement;

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

    vec3 updatedPosition = ((localRotation * position * scale) + (displacement * rotation)) * correction;

    // Apply computed position
    gl_Position = vec4(updatedPosition, 1);
}
