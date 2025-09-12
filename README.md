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

#### Assignment 4

- Matrix multiplication order matters! *(Why...?)*
- Only one axis must be scaled to aspect-correct; use a non-uniform matrix
> Honestly, I need to do the math for all this step-by-step... I got cooked
 
# Checklists

Nothing here!

# Guides

### How to Fix `libGL` Missing

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
