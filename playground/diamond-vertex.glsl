// This shader rotates the initial diamond shape by the given angle using a rotation matrix multiplication

#version 330 core

in vec3 position;

uniform float angle;

void main() {
    float angle = radians(angle);

    float cosA = cos(angle);
    float sinA = sin(angle);

    mat3 rotation = mat3(
            cosA, -sinA, 0,
            sinA, cosA, 0,
            0, 0, 1
        );

    vec3 finalPosition = position * rotation;

    gl_Position = vec4(finalPosition, 1.0);
}
