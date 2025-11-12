#version 460 core

// =========================================================================================================
// CONSTANTS
// =========================================================================================================

const vec3 UP = vec3(0, 1, 0);
const vec3 GROUND_COLOR = vec3(0.3215686274509804,0.4,0.10980392156862745);
const vec3 SKY_COLOR = vec3(0.7176470588235294, 0.7411764705882353, 0.7529411764705882);

// =========================================================================================================
// INPUT
// =========================================================================================================

in vec2 f_uv; // UV
in vec3 f_normal; // Normalized and in world coords
in vec3 f_position; // In world coords

// =========================================================================================================
// OUTPUT
// =========================================================================================================

out vec4 out_color;

// =========================================================================================================
// UNIFORM
// =========================================================================================================

uniform int pcf; // Percentage closer filtering
uniform bool bias_flag; // Use shadow bias
uniform float shininess;
uniform vec3 eye_position;
uniform vec3 k_diffuse; // Diffuse coefficient; how much incoming light should scatter? 
uniform vec3 light;
uniform sampler2D map;

// =========================================================================================================
// AUX FUNCTIONS
// =========================================================================================================

float computeVisibilityFactor(){
  return 1.0;
}

vec3 computeColor(){
  vec3 light_vector = normalize(light.xyz - f_position); // Represents direction to the light source; assumes point light 
  vec3 material_color = texture(map, f_uv).rgb; // Sample the texture
  vec3 normal = normalize(f_normal); // Renormalize for safety

  // Initialize color to black
  vec3 color = vec3(0);

  float w = dot(normal, UP); // Quantifies the direction of the normal relative to the global up 

  // Compute ambient color
  vec3 ambient_color = 0.25 * (w * SKY_COLOR + (1 - w) * GROUND_COLOR) * material_color;

  // Get fractional light visibility
  float fractional_light_vis = computeVisibilityFactor();

  // If the light's intensity is greater than 0 
  float n_dot_l = dot(normal, light_vector); // Quantifies direction of light relative to normal; encodes the light's intensity
  if (n_dot_l > 0){
    vec3 diffuse_color = material_color * n_dot_l; // Base color with diffuse reflection applied to it
    vec3 eye_vector = normalize(eye_position - f_position); // Points from surface vertex to camera
    vec3 halfway = normalize(light_vector + eye_vector); // Halfway between light and camera
    
    // Initialize specular to 0; we'll determine it in a sec
    vec3 specular_color = vec3(0);
    
    if (shininess > 0){
      specular_color = vec3(1) * pow(dot(normal, halfway), shininess);
    }

    // Compute final color
    color = fractional_light_vis * (k_diffuse * diffuse_color + specular_color);
  }

  // Apply ambient color
  color += ambient_color;

  return color;
}

// =========================================================================================================
// MAIN
// =========================================================================================================

void main(){
  out_color = vec4(computeColor(), 1);
}