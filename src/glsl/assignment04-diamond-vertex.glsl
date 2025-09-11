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
    // Calculate angle in radians
    // NOTE: It starts at an offset
    float angleInRadians = radians(angleOffset + angle);

    // Calculate trig ratios
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    // Compute displacement & orbit
    mat3 orbitRotation = mat3(
            cosA, 0, 0,
            0, sinA, 0,
            0, 0, 1);

    vec3 displacementAndOrbit = vec3(dx, dy, 0) * orbitRotation;

    // Compute rotation about self
    mat3 selfRotation = mat3(
            cosA, sinA, 0,
            sinA, -cosA, 0,
            0, 0, 1);

    // Calculate updated position
    vec3 updatedPosition = (selfRotation * position * scale) + displacementAndOrbit;

    // Apply aspect correction
    updatedPosition = vec3(
            updatedPosition.x * xCorrectionFactor,
            updatedPosition.y * yCorrectionFactor,
            0);

    // Apply computed position
    gl_Position = vec4(updatedPosition, 1);
}
