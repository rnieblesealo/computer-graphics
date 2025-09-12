#version 330 core

layout(location = 0) in vec3 position;

uniform float angle;
uniform float angleOffset;
uniform float distance;
uniform vec3 correction;
uniform mat2 m;

void main() {
    mat2 a = m;
    float angleInRadians = radians(angleOffset + angle);
    float cosA = cos(angleInRadians);
    float sinA = sin(angleInRadians);

    // Rotation (creates an orbit with displacement)
    mat3 rotation = mat3(
            cosA, 0, 0,
            0, sinA, 0,
            0, 0, 1);

    vec3 updatedPosition = (distance * rotation * position) * correction;

    gl_Position = vec4(updatedPosition, 1);
}
