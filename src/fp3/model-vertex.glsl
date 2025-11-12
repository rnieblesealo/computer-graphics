#version 460 core

// =========================================================================================================
// INPUT
// =========================================================================================================

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;

// =========================================================================================================
// OUTPUT
// =========================================================================================================

out vec2 f_uv; // Same as in
out vec3 f_normal; // Normal in world coordinates (and normalized to 0-1)
out vec3 f_position; // Position in world coordinates

// =========================================================================================================
// UNIFORM
// =========================================================================================================

uniform mat4 model; // Converts from model -> world
uniform mat4 view; // Converts from world -> camera
uniform mat4 perspective; // Applies depth

// =========================================================================================================
// MAIN
// =========================================================================================================

void main(){
  // Model -> world
  vec4 world_pos = model * vec4(position, 1);
  f_position = world_pos.xyz;

  // UV stays same
  f_uv = uv;

  // Compute normal
  mat3 normal_matrix = mat3(transpose(inverse(model)));
  f_normal = normalize(normal_matrix * normal);

  // Pos from world -> camera; apply perspective  
  gl_Position = perspective * view * world_pos;
} 