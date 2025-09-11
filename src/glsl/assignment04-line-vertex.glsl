#version 330 core

layout(location = 0) in vec3 position;

uniform float xCorrectionFactor;
uniform float yCorrectionFactor;
uniform float dx;
uniform float dy;
uniform float angle;
uniform float angleOffset;

void main() {
    float angleInRadians = radians(angleOffset + angle);
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    vec3 displacement = vec3(dx, dy, 0);

    // Rotation (creates an orbit with displacement)
    mat3 rotation = mat3(
            cosA, 0, 0,
            0, sinA, 0,
            0, 0, 1);

    mat3 aspectCorrection = mat3(
            xCorrectionFactor, 0, 0,
            0, yCorrectionFactor, 0,
            0, 0, 1);

    vec3 updatedPosition = (displacement * rotation * position) * aspectCorrection;

    gl_Position = vec4(updatedPosition, 1);
}
