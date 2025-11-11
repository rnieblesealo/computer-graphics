#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_uv;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec2 f_uv; // Same as in_uv
out vec3 f_normal; // in_normal in world coords 
out vec3 f_position;  // in_position in world coords

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform mat4 model; // Transforms model -> world space
uniform mat4 view; // Transforms world space -> camera space
uniform mat4 perspective; // Transforms 3D coords -> 2D screen coords, creating depth effect

// =========================================================================================
// MAIN 
// =========================================================================================

void main(){
  f_uv = in_uv;

  // Convert model verts to world space 
  vec4 final_pos = model * vec4(in_position, 1);
  f_position = final_pos.xyz;

  // Compute normal matrix
  // This matrix correctly applies any model transformations to its normals
  mat3 normal_matrix = mat3(transpose(inverse(model)));

  // Apply the normal matrix to the model to match its world space transform
  f_normal = normalize(normal_matrix * normal);

  // Convert to world space and apply perspective
  gl_Position = perspective * view * final_pos;
}