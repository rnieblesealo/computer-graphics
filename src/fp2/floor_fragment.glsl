#version 460 core

// =========================================================================================
// INPUT 
// =========================================================================================

in vec2 f_uv;
in vec3 f_position;
in vec3 f_normal;

// =========================================================================================
// OUTPUT
// =========================================================================================

out vec4 out_color;

// =========================================================================================
// UNIFORM
// =========================================================================================

uniform vec4 light;
uniform sampler2D map;

// =========================================================================================
// CONSTANT
// =========================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =========================================================================================
// MAIN
// =========================================================================================

void main(){
  vec3 light_vector = normalize(light.xyz); // Assume directional
  
  // Handle point
  if (light.w > 0){
    light_vector = normalize(light.xyz - f_position);
  }

  vec3 normal = normalize(f_normal); // For safety, renormalize

  // Compute surface orient. coefficient 
  float w = dot(normal, UP);

  // Sample the texture
  vec3 material_color = texture(map, f_uv).rgb;
  
  // Compute ambient color
  vec3 ambient_color = 0.1 * (w * SKY_COLOR + (1 - w) * GROUND_COLOR) * material_color;

  // Combine into final color
  vec3 color = ambient_color + material_color * clamp(dot(normal, light_vector), 0, 1);

  out_color = vec4(color, 1);
}