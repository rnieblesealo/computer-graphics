#version 330 core

in vec3 position; // Input position

uniform float scale;
uniform float angle;
uniform float angleOffset; // Start offset applied to angle
uniform float dx;
uniform float dy;
uniform float xCorrectionFactor;
uniform float yCorrectionFactor;

void main() {
    float angleInRadians = radians(angleOffset + angle);

    // Apply desired rotation to offset
    float dxAngled = dx * cos(angleInRadians);
    float dyAngled = dy * sin(angleInRadians);

    // Apply shape scale
    vec3 updatedPosition = (position * scale) + vec3(dxAngled, dyAngled, 0);

    // Apply aspect correction
    updatedPosition = vec3(
            updatedPosition.x * xCorrectionFactor,
            updatedPosition.y * yCorrectionFactor,
            0);

    // Apply vertex position; in this case we leave as it was on input
    gl_Position = vec4(updatedPosition, 1);
}
