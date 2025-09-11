#version 330 core

layout(location = 0) in vec3 position;

uniform float angle;
uniform float angleOffset;
uniform vec2 correction;
uniform vec2 displacement;

void main() {
    float angleInRadians = radians(angleOffset + angle);
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    // Rotation (creates an orbit with displacement)
    mat3 rotation = mat3(
            cosA, 0, 0,
            0, sinA, 0,
            0, 0, 1);

    vec3 updatedPosition = (vec3(displacement, 0) * rotation * position) * vec3(correction, 0);

    gl_Position = vec4(updatedPosition, 1);
}
