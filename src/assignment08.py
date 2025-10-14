import glm
import moderngl
import numpy as np
import pygame

from PIL import Image
from loadModelUsingAssimp_V1 import getObjectDataList
from pathlib import Path

MODEL_PATH = "./assets/mario_obj/scene.gltf"


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
        arr = geom.reshape(-1, 5)
        pos = arr[:, 0:3].astype(np.float64)
        min_v = np.minimum(min_v, pos.min(axis=0))
        max_v = np.maximum(max_v, pos.max(axis=0))
    return SceneBound(min_v, max_v)


def resolve_texture_path(base_dir: Path, tex_rel: str | None) -> Path | None:
    """
    Resolve a texture path relative to the GLTF directory (robust against variants).
    """
    if not tex_rel:
        return None
    tex_rel = tex_rel.replace('\\', '/')
    candidates = []
    base_dir_name = base_dir.name
    candidates.append(base_dir / tex_rel)
    parts = Path(tex_rel).parts
    if len(parts) >= 2 and parts[0] == base_dir_name:
        candidates.append(base_dir / Path(*parts[1:]))
    candidates.append(base_dir / "textures" / Path(tex_rel).name)
    candidates.append(base_dir / Path(tex_rel).name)
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    print(f"[WARN] Could not resolve texture path from '{tex_rel}'.")
    return None


def compute_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    positions: (N,3) float32
    indices: (M,) int32 (triangulated)
    Returns per-vertex normals (N,3) float32.
    """
    N = positions.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)

    tris = indices.reshape(-1, 3)
    p = positions.astype(np.float64)

    for i0, i1, i2 in tris:
        v0, v1, v2 = p[i0], p[i1], p[i2]
        e1 = v1 - v0
        e2 = v2 - v0
        fn = np.cross(e1, e2)  # area-weighted face normal
        normals[i0] += fn
        normals[i1] += fn
        normals[i2] += fn

    # Normalize (avoid div-by-zero)
    lens = np.linalg.norm(normals, axis=1)
    nz = lens > 1e-20
    normals[nz] /= lens[nz][:, None]
    normals[~nz] = np.array([0.0, 1.0, 0.0])  # default up if degenerate

    return normals.astype(np.float32)


class Mesh:
    """Wraps a mesh's GPU buffers and its texture."""

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program,
                 vertex_data: np.ndarray, index_data: np.ndarray, texture_path: Path | None):
        self.ctx = ctx
        self.program = program

        # --- Expand layout to include normals: [pos(3), normal(3), uv(2)] ---
        # The loader gives us [pos(3), uv(2)] interleaved; derive normals:
        arr = vertex_data.reshape(-1, 5).astype(np.float32)
        positions = arr[:, 0:3]
        uvs = arr[:, 3:5]
        indices = index_data.astype(np.int32)

        normals = compute_vertex_normals(positions, indices)

        interleaved = np.hstack([positions, normals, uvs]).astype(np.float32)

        # GPU buffers
        self.vbo = ctx.buffer(interleaved.tobytes())
        self.ibo = ctx.buffer(indices.astype('i4').tobytes())

        # VAO layout: vec3 position + vec3 normal + vec2 uv
        self.vao = ctx.vertex_array(
            program,
            [(self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_texcoord_0')],
            index_buffer=self.ibo,
            index_element_size=4  # 32-bit indices
        )

        # Texture
        self.texture = None
        if texture_path is not None and texture_path.exists():
            try:
                img = Image.open(texture_path)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
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
        self.vao.render()


class GraphicsEngine:
    def __init__(self, width=500, height=500):
        pygame.init()

        # GL 3.3 Core
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        self.WIDTH, self.HEIGHT = width, height
        pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                                pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(
            "Assignment 7 (Updated Lighting): Rafael Niebles")

        # ModernGL
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Load model
        try:
            geomDataList, indexList, _bounds_from_loader, texNames, _scene = getObjectDataList(
                MODEL_PATH, verbose=False
            )
        except FileNotFoundError as e:
            print(f"[ERROR] Could not find: {e.filename}")
            pygame.quit()
            raise SystemExit

        # Robust bounds
        self.bounds = compute_bounds_from_geom(geomDataList)

        # --- Shaders: now with normals + diffuse lighting ---
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_proj;

                // For transforming normals when non-uniform scale exists
                uniform mat3 u_normal_matrix;

                out vec2 v_uv;
                out vec3 v_normal; // in world space

                void main() {
                    v_uv = in_texcoord_0;
                    // Transform to world space
                    vec4 world_pos = u_model * vec4(in_position, 1.0);
                    // Normal matrix already derived from u_model
                    v_normal = normalize(u_normal_matrix * in_normal);

                    gl_Position = u_proj * u_view * world_pos;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in vec3 v_normal;

                uniform sampler2D u_texture;
                uniform bool u_has_texture;

                // Directional light coming FROM this direction.
                // (If you prefer "to light" instead, flip the sign when computing L.)
                uniform vec3 u_light_dir;

                out vec4 f_color;

                void main() {
                    vec3 N = normalize(v_normal);
                    vec3 L = normalize(-u_light_dir); // light points opposite its direction vector

                    float ndotl = max(dot(N, L), 0.0);

                    vec3 kd = vec3(0.8); // default gray
                    vec4 texc = vec4(kd, 1.0);
                    if (u_has_texture) {
                        texc = texture(u_texture, v_uv);
                        if (texc.a < 0.1) discard; // keep your alpha-discard
                        kd = texc.rgb;
                    }

                    vec3 diffuse = kd * ndotl;

                    // No ambient per assignment (pure diffuse). If you want a hint of ambient, add a small term.
                    f_color = vec4(diffuse, texc.a);
                }
            """
        )

        # Uniform handles
        self.u_model = self.program["u_model"]
        self.u_view = self.program["u_view"]
        self.u_proj = self.program["u_proj"]
        self.u_normal_matrix = self.program["u_normal_matrix"]
        self.program["u_texture"].value = 0
        self.u_has_texture = self.program["u_has_texture"]
        self.u_light_dir = self.program["u_light_dir"]

        # ----- Model transform (rotate upright around its center) -----
        model_matrix = glm.mat4(1.0)
        model_matrix = glm.translate(
            model_matrix, self.bounds.center)                               # step 3
        model_matrix = glm.rotate(
            # step 2
            model_matrix, glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))
        model_matrix = glm.translate(
            model_matrix, -self.bounds.center)                              # step 1
        self.u_model.write(model_matrix.to_bytes())

        # Normal matrix: mat3(transpose(inverse(model)))
        normal_mat = glm.transpose(glm.inverse(glm.mat3(model_matrix)))
        self.u_normal_matrix.write(normal_mat.to_bytes())

        # Pack meshes + resolve texture
        base_dir = Path(MODEL_PATH).parent  # <-- fixed typo
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

        self.radius = self.bounds.radius if self.bounds.radius > 0 else 1.0
        d = 2.0 * self.radius
        self.initial_eye = self.lookAtPoint + glm.vec3(0.0, 0.0, d)

        fov_deg = 45.0
        aspect = self.WIDTH / self.HEIGHT
        near_plane = 0.1 * self.radius
        far_plane = 10.0 * self.radius
        proj = glm.perspective(glm.radians(
            fov_deg), aspect, near_plane, far_plane)
        self.u_proj.write(proj.to_bytes())

        # Track camera displacement to drive light direction
        self.prev_eye = glm.vec3(self.initial_eye)
        self.curr_light_dir = glm.vec3(0.0, 0.0, -1.0)  # default

    # ---------- Main loop ----------
    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Orbit around Y axis
            t = pygame.time.get_ticks() / 2000.0
            angle = t
            rel = self.initial_eye - self.lookAtPoint
            R = glm.rotate(glm.mat4(1.0), angle, glm.vec3(0, 1, 0))
            eye = glm.vec3(R * glm.vec4(rel, 1.0)) + self.lookAtPoint

            # View matrix
            view = glm.lookAt(eye, self.lookAtPoint, self.up)
            self.u_view.write(view.to_bytes())

            # --- Light direction = camera displacement direction ---
            disp = eye - self.prev_eye
            if glm.length(disp) > 1e-6:
                # Directional light coming FROM disp direction.
                self.curr_light_dir = glm.normalize(disp)
                self.prev_eye = glm.vec3(eye)

            # Push uniform (as vec3)
            # Note: fragment uses L = normalize(-u_light_dir),
            # so u_light_dir is interpreted as the direction the light is pointing *from*.
            self.u_light_dir.write(bytes(np.array(
                [self.curr_light_dir.x, self.curr_light_dir.y, self.curr_light_dir.z], dtype=np.float32)))

            # Clear & draw
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
