import glm
import moderngl
import numpy as np
import pygame

from PIL import Image
from loadModelUsingAssimp_V1 import getObjectDataList
from pathlib import Path

MODEL_PATH = "./assets/mario_obj/scene.gltf"
SCREENSHOT_NAME = "mario.jpg"


class SceneBound:
    """Lightweight bounds container (center, radius)."""

    def __init__(self, min_v: np.ndarray, max_v: np.ndarray):
        self.boundingBox = [min_v, max_v]

        c = (min_v + max_v) * 0.5

        self.center = glm.vec3(float(c[0]), float(c[1]), float(c[2]))

        r = max_v - min_v

        self.radius = float(glm.length(
            glm.vec3(float(r[0]), float(r[1]), float(r[2]))) * 0.5)


def compute_bounds_from_geom(geomDataList: list[np.ndarray]) -> SceneBound:
    """
    The loader returns per-mesh float32 arrays laid out as [x,y,z, u,v, x,y,z, u,v, ...].
    We recompute bounds directly from these to avoid any issues with the loader's globals.
    """

    min_v = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_v = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for geom in geomDataList:
        if geom.size == 0:
            continue
        # reshape to N x 5: columns 0..2 are position
        arr = geom.reshape(-1, 5)
        pos = arr[:, 0:3].astype(np.float64)
        min_v = np.minimum(min_v, pos.min(axis=0))
        max_v = np.maximum(max_v, pos.max(axis=0))

    return SceneBound(min_v, max_v)


def resolve_texture_path(base_dir: Path, tex_rel: str | None) -> Path | None:
    """
    Resolve a texture path relative to the GLTF directory.

    The loader may return values like:
      - "textures/submesh_0_baseColor.png"
      - "mario_obj/textures/submesh_0_baseColor.png"
      - just "submesh_0_baseColor.png"
    We try multiple options under base_dir.
    """

    if not tex_rel:
        return None

    # Clean up potential path issues like backslashes
    tex_rel = tex_rel.replace('\\', '/')

    candidates = []
    base_dir_name = base_dir.name

    # 1) As-is under base_dir
    candidates.append(base_dir / tex_rel)

    # 2) If tex_rel starts with the model folder name, strip it
    parts = Path(tex_rel).parts
    if len(parts) >= 2 and parts[0] == base_dir_name:
        candidates.append(base_dir / Path(*parts[1:]))

    # 3) Just the filename under base_dir/textures
    candidates.append(base_dir / "textures" / Path(tex_rel).name)

    # 4) Just the filename under base_dir
    candidates.append(base_dir / Path(tex_rel).name)

    for p in candidates:
        if p.exists() and p.is_file():
            return p

    print(f"[WARN] Could not resolve texture path from '{tex_rel}'.")

    return None


class Mesh:
    """Wraps a mesh's GPU buffers and its texture."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program,
                 vertex_data: np.ndarray, index_data: np.ndarray, texture_path: Path | None):
        self.ctx = ctx
        self.program = program

        # GPU buffers
        self.vbo = ctx.buffer(vertex_data.astype('f4').tobytes())
        self.ibo = ctx.buffer(index_data.astype('i4').tobytes())

        # VAO layout: vec3 position + vec2 uv
        self.vao = ctx.vertex_array(
            program,
            [(self.vbo, '3f 2f', 'in_position', 'in_texcoord_0')],
            index_buffer=self.ibo,
            index_element_size=4  # Use 32-bit indices
        )

        # Texture
        self.texture = None
        if texture_path is not None and texture_path.exists():
            try:
                img = Image.open(texture_path)
                # Flip vertically - GLTF textures often need this
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

                # Convert to RGB/RGBA for GL
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")
                comp = 4 if img.mode == "RGBA" else 3

                self.texture = ctx.texture(img.size, comp, img.tobytes())
                self.texture.build_mipmaps()
                self.texture.filter = (
                    moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                try:
                    self.texture.anisotropy = 8.0
                except Exception:
                    pass
                self.texture.repeat_x = True
                self.texture.repeat_y = True
            except Exception as e:
                print(f"[ERROR] Failed to load texture {texture_path}: {e}")

    def render(self):
        # The engine will now handle texture binding, this just renders the geometry
        self.vao.render()


class GraphicsEngine:
    def __init__(self, width=1280, height=720):
        pygame.init()

        # Request a forward-compatible Core profile 3.3 context
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        self.WIDTH, self.HEIGHT = width, height
        pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                                pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Assignment 7: Rafael Niebles")

        # ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Load model via the provided loader
        try:
            geomDataList, indexList, _bounds_from_loader, texNames, _scene = getObjectDataList(
                MODEL_PATH, verbose=False
            )
        except FileNotFoundError as e:
            print(f"[ERROR] Could not find: {e.filename}")
            pygame.quit()
            raise SystemExit

        # Recompute robust bounds from vertex data (ignores loader's global-state issues)
        self.bounds = compute_bounds_from_geom(geomDataList)

        # Shaders
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec2 in_texcoord_0;

                uniform mat4 u_model; // <<< ADDED
                uniform mat4 u_view;
                uniform mat4 u_proj;

                out vec2 v_uv;

                void main() {
                    v_uv = in_texcoord_0;
                    // Standard MVP transformation
                    gl_Position = u_proj * u_view * \
                        u_model * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;

                uniform sampler2D u_texture;
                uniform bool u_has_texture; // <<< ADDED

                out vec4 f_color;

                void main() {
                    if (u_has_texture) {
                        f_color = texture(u_texture, v_uv);
                        // Discard transparent fragments if they exist
                        if (f_color.a < 0.1) discard;
                    } else {
                        // Render a solid color if no texture is available
                        f_color = vec4(0.8, 0.8, 0.8, 1.0);
                    }
                }
            """
        )
        self.u_model = self.program["u_model"]
        self.u_view = self.program["u_view"]
        self.u_proj = self.program["u_proj"]
        self.program["u_texture"].value = 0  # texture unit 0
        self.u_has_texture = self.program["u_has_texture"]  # <<< ADDED

        # To rotate the model around its own center and not the world origin,
        # we compose the model matrix in three steps:
        # 1. Translate the model to the origin.
        # 2. Rotate it to stand upright.
        # 3. Translate it back to its original position.
        # Note: GLM matrix operations are applied in reverse order of calls.

        # Start with an identity matrix
        model_matrix = glm.mat4(1.0)

        # 3. Translate back to its original center (applied last)
        model_matrix = glm.translate(model_matrix, self.bounds.center)

        # 2. Rotate -90 degrees around the X-axis (applied second)
        model_matrix = glm.rotate(
            model_matrix, glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))

        # 1. Translate the model's center to the origin (applied first)
        model_matrix = glm.translate(model_matrix, -self.bounds.center)

        self.u_model.write(model_matrix.to_bytes())

        # Pack meshes and resolve textures relative to the GLTF folder
        base_dir = Path(MODEL_PATH).parent  # e.g., ./assets/mario_obj
        self.meshes: list[Mesh] = []
        for geom, idx, tex_rel in zip(geomDataList, indexList, texNames):
            if geom.size == 0 or idx.size == 0:
                continue
            tex_path = resolve_texture_path(base_dir, tex_rel)
            if tex_path is None and tex_rel:
                print(f"[WARN] Texture not found for mesh entry '{
                      tex_rel}'. Rendering with default color.")
            self.meshes.append(
                Mesh(self.ctx, self.program, geom, idx, tex_path))

        # ---------- Camera ----------

        self.lookAtPoint = glm.vec3(self.bounds.center)
        self.up = glm.vec3(0, 1, 0)

        # Start at 2*radius along +Z as required

        self.radius = self.bounds.radius if self.bounds.radius > 0 else 1.0

        d = 2.0 * self.radius

        self.initial_eye = self.lookAtPoint + glm.vec3(0.0, 0.0, d)

        # Projection sized from camera distance d (robust to odd scales)

        fov_deg = 45.0
        aspect = self.WIDTH / self.HEIGHT
        near_plane = 0.1 * self.radius
        far_plane = 10.0 * self.radius
        proj = glm.perspective(glm.radians(
            fov_deg), aspect, near_plane, far_plane)

        self.u_proj.write(proj.to_bytes())

    # ---------- Main loop ----------

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Orbit around Y axis
            t = pygame.time.get_ticks() / 2000.0  # Slowed down rotation a bit
            angle = t
            rel = self.initial_eye - self.lookAtPoint
            R = glm.rotate(glm.mat4(1.0), angle, glm.vec3(0, 1, 0))
            eye = glm.vec3(R * glm.vec4(rel, 1.0)) + self.lookAtPoint

            view = glm.lookAt(eye, self.lookAtPoint, self.up)
            self.u_view.write(view.to_bytes())

            self.ctx.clear(color=(126 / 255, 128 / 255, 28 / 255))

            for m in self.meshes:
                if m.texture:
                    self.u_has_texture.value = True
                    m.texture.use(location=0)
                else:
                    self.u_has_texture.value = False
                m.render()

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    app = GraphicsEngine()
    app.run()
