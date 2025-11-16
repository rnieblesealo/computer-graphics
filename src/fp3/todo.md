# To-Do List: Render Shadows Using the Shadow Mapping Technique

## Pass 1: ShadowMap Creation
- [x] Set up camera at the light source (define eye point, target point, upVector).
- [x] Compute viewMatrix using light point as the eye.
- [x] Compute perspectiveMatrix (near = bound.radius, far = 10*bound.radius, FOV=60).
- [x] Update shader program to render scene with no color output.
- [ ] Create renderables for the model object and floor.
- [ ] Create an empty 2k x 2k depth texture.
- [ ] Attach depth texture to a Frame buffer.
- [ ] Create shadowMapSampler using the depth texture.
- [ ] Save default Frame Buffer value.
- [ ] Use created Framebuffer (only depth texture).
- [ ] Render scene (model object and floor renderables).
- [ ] Restore old Frame Buffer for subsequent rendering.

## Pass 2: Rendering with Shadow
- [ ] Attach shadowMapSampler to an unused texture unit.
- [ ] Update shader code with sampler2D uniform for shadowMapSampler.
- [ ] Add mat4 uniform variables for light camera view and perspective matrices.
- [ ] Update ComputeVisibilityFactor function in Fragment shader for shadow checking.
- [ ] Implement logic for returning float based on shadow presence (0 or 1 for no PCF; fractional for PCF).
- [ ] Use two uniform values: bias (boolean) for depth adjustment and pcf (int) for controlling PCF computation.

## Show Shadow Map
- [ ] Update ShowShadowMap method in template code.
- [ ] Create screen-aligned quad geometry with appropriate positions and texture coordinates.
- [ ] Write shader to map depth texture to quad.
- [ ] Create shadowmap renderer using shader program and quad geometry.
- [ ] Make render call in ShowShadowMap with required uniform settings.

## Grading Rubric
- [ ] Implement keyboard key "d" for Debug: show shadow map in smaller viewport.
- [ ] Implement PCF support using keyboard keys "upArrow" and "downArrow" for shadowmap access increment/decrement.
- [ ] Toggle bias use with keyboard key "b" for correct shadow rendering.
- [ ] Ensure correct shadow displayed and texture at top left corner for full grade.

