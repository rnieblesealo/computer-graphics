#version 330 core

in vec3 position; // Input position

uniform float scale;
uniform float angle;
uniform float angleOffset; // Start offset applied to angle
uniform float dx;
uniform float dy;
uniform float xCorrectionFactor;
uniform float yCorrectionFactor;
uniform bool rotateClockwise;

void main() {
    float angleInRadians = radians(angleOffset + angle);
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    vec3 displacement = vec3(dx, dy, 0);

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

    mat3 aspectCorrection = mat3(
            xCorrectionFactor, 0, 0,
            0, yCorrectionFactor, 0,
            0, 0, 1);

    vec3 updatedPosition = ((localRotation * position * scale) + (displacement * rotation)) * aspectCorrection;

    // Apply computed position
    gl_Position = vec4(updatedPosition, 1);
}
