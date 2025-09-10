#version 330 core

in vec3 position; // Input position
out vec3 v_color; // Output color to fragment shader

uniform float scale;
uniform float angle;
uniform float angleOffset; // Start offset applied to angle
uniform float dx;
uniform float dy;
uniform float correctionFactor;

void main() {
    float angleInRadians = radians(angleOffset + angle);

    float dxAngled = dx * cos(angleInRadians);
    float dyAngled = dy * sin(angleInRadians);

    vec3 updatedPosition = (position * scale) + vec3(dxAngled, dyAngled, 0);

    // Apply vertex position; in this case we leave as it was on input
    gl_Position = vec4(updatedPosition * correctionFactor, 1.0);

    // Set fragment color to yellow, as specified in assignment
    // The fragment shader will receive this value
    v_color = vec3(1.0, 1.0, 0.0);
}
