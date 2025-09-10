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
    float angleInRadians = radians(angleOffset + angle);

    // Compute rotation matrices
    float cosA = cos(angleInRadians); // NOTE: How offset is not applied!
    float sinA = sin(angleInRadians);

    mat3 clockwiseRotationMatrix = mat3(
            -cosA, -sinA, 0,
            sinA, -cosA, 0,
            0, 0, 1
        );

    // Apply orbit rotation
    float dxAngled = dx * cos(angleInRadians);
    float dyAngled = dy * sin(angleInRadians);

    // Apply rotation about self
    vec3 updatedPosition = clockwiseRotationMatrix * position;

    // Apply scale
    updatedPosition = (updatedPosition * scale) + vec3(dxAngled, dyAngled, 0);

    // Apply aspect correction
    updatedPosition = vec3(
            updatedPosition.x * xCorrectionFactor,
            updatedPosition.y * yCorrectionFactor,
            0);

    // Apply vertex position; in this case we leave as it was on input
    gl_Position = vec4(updatedPosition, 1);
}
