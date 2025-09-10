Materials authored by me throughout taking CAP4720 Computer Graphics.

# Notes

- **OpenGL Context:** Holds state related to rendering; **thread-specific**
- **Vertex Shader:** Operates and transforms vertex data
- **Fragment Shader:** Handles how vertices will be presented to screen
- **VBO:** Stores vertex data
- **VAO:** Encapsulates state surrounding vertex data (Is initialized with shader program + VBO)
- **Shader Program/`ctx.program()`:** Compiles shaders into a program
- **What happens when issuing a `vao.render()` call:**
1. VAO is bound to context
> i.e. Set as active
2. Its shader program is activated
3. GPU does render processing and display buffer writing 
4. VAO state cleared but remains active until is changed
- `gl_Position` is a `vec4` because OGL uses homogeneous coordinate system
> Essentially the 4th `w` coord allows messing around with position further

# Checklists

### Assignment 4

- **Change Shape Starting Position**
  - [x] Adjust the shape and line to start at the vertical location (12 O' clock position).
  - [x] Implement clockwise rotation to cover 360 degrees in one minute.

- **Enable Window Resizing**
  - [x] Make the display window resizable.
  - [x] Implement aspect ratio correction to prevent shape distortion during resizing.

- **Create and Apply Transformation Matrix**
  - [1/2] Create a 2x2 transformation matrix for rotation and aspect ratio handling.
  - [ ] Pass the transformation matrix as a uniform variable before the render call.
  - [ ] In the shader, set the uniform matrix value before the render call.
  - [ ] Apply the transformation matrix to the position attribute in the shader.

- **Ensure Correct Shape Rotation**
  - [x] Verify that the shape geometry rotates around its center and along the perimeter of the circular path.

> Generated with duck.ai

# Guides

### How to Fix libGL Missing

ModernGL looks for `libGL.so` and `libEGL.so` by these literals; in the system they may have a version extension, in PopOS `libGL.so.1` for example.

To fix, add symlinks to the versioned files with the names ModernGL expects.

Exact Steps:

1. `cd / && fdfind libgl.so`
2. `sudo -s` (this is a permission-restricted folder) and go to that dir
3. Make the symlinks:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so # libGL fix
ln -sf /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libEGL.so # libEGL fix
```

4. Run the env and the Python src file normally
> ModernGL might ask for more libraries to be symlinked; try this fix and it'll likely work
