Materials authored by me throughout taking CAP4720 Computer Graphics.

### How to Fix libGL Missing

ModernGL looks for `libGL.so` and `libEGL.so` by these literals; in the system they may have a version extension, inPopOS `libGL.so.1` for example.

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
