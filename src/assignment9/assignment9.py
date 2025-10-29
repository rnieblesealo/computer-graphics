from PIL import Image
from loadModelUsingAssimp_V1 import getObjectDataList
from pathlib import Path
from pyglm import glm

import moderngl
import numpy as np
import pygame

MODEL_PATH = Path("./assets/mario_obj/scene.gltf")
TEXTURE_PATH = Path("./assets/mario_obj/textures/")


class SceneBounds:
    def __init__(self, minV: np.ndarray, maxV: np.ndarray):
        # Store bounds

        self.bounds = [minV, maxV]

        # Compute center
        c = (minV + maxV) * 1 / 2
        self.center = glm.vec3(float(c[0]), float(c[1]), float(c[2]))

        # Compute radius
        r = maxV - minV
        self.radius = float(
            glm.length(glm.vec3(float(r[0]), float(r[1]), float(r[2]))) * 1 / 2
        )


def getBoundsFromGeometry(geometryData: list[np.ndarray]) -> SceneBounds:
    # Initialize bounds
    minV = np.array([np.inf] * 3, dtype=np.float32)
    maxV = np.array([-np.inf] * 3, dtype=np.float32)

    # Go over all meshes/geometry
    for geom in geometryData:
        # Don't process empty meshes; we shouldn't receive these in the first place
        if geom.size == 0:
            continue

        # Convert to 2D array where vertices are rows
        geomVertices = geom.reshape(-1, 5)

        # Extract positions only (discard UVs)
        geomPositions = geomVertices[:, 0:3].astype(np.float32)

        """
        Example geom:
        [
            x, y, z, u, v,
            x, y, z, u, v,
            ...
        ]

        Example geomVertices:
        [
            [x, y, z, u, v],
            [x, y, z, u, v],
            [...]
        ]

        Example geomPositions:
        [
            [x, y, z], 
            [x, y, z],
            [...]
        ]
        """

        # Update min and max
        # Min will shrink, max will grow
        # The smallest x, y, z out of all meshes will be left for min (each can be from diff. vert of that mesh)
        # Same logic for max
        minV = np.minimum(minV, geomPositions.min(axis=0))
        maxV = np.maximum(maxV, geomPositions.max(axis=0))

    return SceneBounds(minV, maxV)


def computeVertexNormals(
    geomPositions: np.ndarray, geomIndices: np.ndarray
) -> np.ndarray:
    nVerts = geomPositions.shape[0]  # .shape = [R, C]; each vertex is a row

    # Initialize normals
    geomNormals = np.zeros(
        (nVerts, 3), dtype=np.float32
    )  # Always 3 cols because x, y, z (no UVs here)

    # Indices will be passed flat
    # Marshal them into 2D array with 3 cols (index 1, index 2, index 3...)
    geomTris = geomIndices.reshape(-1, 3)

    # Ensure positions are float
    geomPositions = geomPositions.astype(np.float64)

    """
    Example vertex data:
    [
        [x, y, z], # Vertex 0
        [x, y, z], # Vertex 1
        [x, y, z], # Vertex 2
        [x, y, z]  # Vertex 3
    ]

    Example triangle/index data:
    [
        [0, 1, 3], # Comprised of vertex 0, vertex 1, vertex 3
        [0, 2, 3]  # Same logic as above...
    ]
    """

    # Go over each tri (i.e. list of 3 indices)
    for index0, index1, index2 in geomTris:
        # Retrieve vertices comprising each tri
        vertex0, vertex1, vertex2 = (
            geomPositions[index0],
            geomPositions[index1],
            geomPositions[index2],
        )

        # Compute 2 edges and their cross product to get the normal (cross product requires 2 edges only, not 3)
        edgeA = vertex1 - vertex0
        edgeB = vertex2 - vertex0
        faceNormal = np.cross(edgeA, edgeB)

        # Assign that normal to each vertex index
        # TODO: Increment here is needed, why?
        geomNormals[index0] += faceNormal
        geomNormals[index1] += faceNormal
        geomNormals[index2] += faceNormal

    # Now we need to normalize our calculated face normal (divide by length)

    # Get face normal lengths
    geomNormalLengths = np.linalg.norm(geomNormals, axis=1)

    # Exclude very small lengths to avoid zero div errors
    SMALL_THRESHOLD = 1e-20
    nonZero = geomNormalLengths > SMALL_THRESHOLD

    geomNormals[nonZero] /= geomNormalLengths[nonZero][:, None]

    # Assign a default normal to those that were excluded
    DEFAULT_FACE_NORMAL = np.array([0, 1, 0], dtype=np.float32)
    geomNormals[~nonZero] = DEFAULT_FACE_NORMAL

    return geomNormals.astype(np.float32)


class Mesh:
    def __init__(
        self,
        ctx: moderngl.Context,
        program: moderngl.Program,
        vertexData: np.ndarray,
        indexData: np.ndarray,
        texturePath: Path,
    ):
        self.ctx = ctx
        self.program = program

        """
        Loader gives:
        [
            [x, y, z, u, v],
            ...
        ]

        We want to expand to include normals in middle: 
        [
            [x, y, z, nx, ny, nz, u, v],
            ...
        ]
        """

        geomVertices = vertexData.reshape(-1, 5).astype(np.float32)
        geomPositions = geomVertices[:, 0:3]
        geomUVs = geomVertices[:, 3:5]
        geomIndices = indexData.astype(np.int32)

        # Compute normals and interleave them for expanded form
        geomNormals = computeVertexNormals(geomPositions, geomIndices)
        geomInterleaved = np.hstack([geomPositions, geomNormals, geomUVs]).astype(
            np.float32
        )

        # Create GPU buffers
        self.vbo = ctx.buffer(geomInterleaved.tobytes())
        self.ibo = ctx.buffer(geomIndices.astype(np.int32).tobytes())

        """
        The IBO stores vertex indices and is used to ultimately draw triangles
        """

        # Create VAO
        # Layout: Position, Normals, UV, just like we did above
        self.vao = ctx.vertex_array(
            program,
            [(self.vbo, "3f 3f 2f", "in_position", "in_normal", "in_texcoord_0")],
            index_buffer=self.ibo,
            index_element_size=4,  # 32-bit indices
        )

        # Load & create texture

        self.texture = None
        if texturePath is not None and texturePath.exists():
            try:
                texImg = Image.open(texturePath)
                texImg = texImg.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )  # OpenGL textures are bottom-left!

                # Ensure texture is in RGBA (consistent format for all)
                if texImg.mode != "RGBA":
                    texImg = texImg.convert("RGBA")

                RGBA_COMPONENT_COUNT = 4

                # Create texture
                self.texture = ctx.texture(
                    texImg.size, RGBA_COMPONENT_COUNT, texImg.tobytes()
                )

                # Build scaled versions of texture for reduced aliasing
                self.texture.build_mipmaps()

                # Set texture sampling settings
                self.texture.filter = (
                    moderngl.LINEAR_MIPMAP_LINEAR,  # Trilinear, smoothest mipmap blend
                    moderngl.LINEAR,  # Smooth interpolation between pixels
                )

                # Set anisotropic filtering to 8; sharpens texture at steeper angles
                # Hardware must support this so we must prepare try catch in case it fails
                try:
                    self.texture.anisotropy = 8.0
                except:
                    pass

                # Make sure texture repeats if coordinates go over
                self.texture.repeat_x = True
                self.texture.repeat_y = True
            except:
                print("Unable to set up texture :/")

    def render(self):
        # Render this mesh
        self.vao.render()


class GraphicsEngine:
    def __init__(self, width=500, height=500) -> None:
        pygame.init()

        # Ensure forward compat with OpenGL for Mac to work
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        self.WIDTH = width
        self.HEIGHT = height

        # Start pygame window overriden by OpenGL
        pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Assignment 9: Rafael Niebles")

        # Initialize ModernGL
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        # Load model
        objectDataList = getObjectDataList(MODEL_PATH.as_posix(), verbose=True)
        if objectDataList is None:
            print("Loader failed :/")
            pygame.quit()
            exit()

        geomDataList, indexList, loaderBounds, textureNames, scene = objectDataList

        # Get bounds
        self.bounds = getBoundsFromGeometry(geomDataList)

        # Set default state
        self.cameraPaused = False
        self.isPointLight = True
        self.lightAngleY = glm.radians(45)

        # Create shader program
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                uniform mat4 u_model;
                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform mat3 u_normal_matrix;

                out vec2 v_uv;
                out vec3 v_normal;      // Normal in world space
                out vec3 v_world_pos;   // Fragment position in world space

                void main() {
                    v_uv = in_texcoord_0;

                    // Transform vertex position and normal to world space
                    vec4 world_pos_4 = u_model * vec4(in_position, 1.0);
                    v_world_pos = world_pos_4.xyz;
                    v_normal = normalize(u_normal_matrix * in_normal);

                    // Transform to clip space for rendering
                    gl_Position = u_proj * u_view * world_pos_4;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in vec3 v_normal;
                in vec3 v_world_pos;

                uniform sampler2D u_texture;
                uniform bool u_has_texture;

                // Light properties
                uniform vec4 u_light;      // .xyz = pos/dir, .w=1 for point, .w=0 for directional
                uniform vec3 u_eye_pos;    // Camera position for specular calculation

                // Material properties
                uniform vec3 u_diffuse_reflectance;
                uniform float u_shininess;

                out vec4 f_color;

                void main() {
                    // Normalize inputs
                    vec3 N = normalize(v_normal);
                    vec3 V = normalize(u_eye_pos - v_world_pos); // View vector

                    // Determine Light Vector (L) based on light type
                    vec3 L;
                    if (u_light.w == 0.0) { // Directional light
                        L = normalize(-u_light.xyz);
                    } else { // Point light
                        L = normalize(u_light.xyz - v_world_pos);
                    }

                    // Get base color from texture or use default material color
                    vec3 base_color = u_diffuse_reflectance;
                    vec4 tex_color = vec4(1.0);
                    if (u_has_texture) {
                        tex_color = texture(u_texture, v_uv);
                        if (tex_color.a < 0.1) discard;
                        base_color *= tex_color.rgb; // Modulate material color with texture
                    }

                    // Ambient component
                    vec3 ambient = vec3(0.1) * base_color;

                    // Diffuse component
                    float diffuse_intensity = max(dot(N, L), 0.0);
                    vec3 diffuse = diffuse_intensity * base_color;

                    // Specular component (Blinn-Phong)
                    vec3 H = normalize(L + V); // Halfway vector
                    float spec_intensity = pow(max(dot(N, H), 0.0), u_shininess);
                    vec3 specular = vec3(1.0, 1.0, 1.0) * spec_intensity; // White highlights

                    // Combine all
                    vec3 final_color = ambient + diffuse + specular;
                    f_color = vec4(final_color, tex_color.a);
                }
            """,
        )

        # Create handles for the uniforms so we can more easily modify them
        self.u_model = self.program["u_model"]
        self.u_view = self.program["u_view"]
        self.u_proj = self.program["u_proj"]
        self.u_normal_matrix = self.program["u_normal_matrix"]
        self.u_has_texture = self.program["u_has_texture"]
        self.u_light = self.program["u_light"]
        self.u_eye_pos = self.program["u_eye_pos"]
        self.u_diffuse_reflectance = self.program["u_diffuse_reflectance"]
        self.u_shininess = self.program["u_shininess"]

        # We will sample from texture at unit 0 (albedo)
        self.program["u_texture"].value = 0

        """
        Texture units are "layers" to our model's texture
        0 -> Diffuse/Albedo 
        1 -> Normal map
        2 -> Roughness

        You've seen these before

        Setting u_texture to 0 is us telling the texture sampling code to sample from tex. unit 0 (the albedo)
        We must bind the texture object to that same slot, obviously!
        """

        # Set default material properties; loader doesn't give them
        self.u_diffuse_reflectance.value = (1, 1, 1)
        self.u_shininess.value = 32

        # Model transformation setup
        # TODO: Explain what these transformations do
        modelMatrix = glm.mat4(1)
        modelMatrix = glm.translate(modelMatrix, self.bounds.center)
        modelMatrix = glm.rotate(modelMatrix, glm.radians(-90), glm.vec3(1, 0, 0))
        modelMatrix = glm.translate(modelMatrix, -self.bounds.center)

        # TODO: Can't we use .write to address .value? Isn't this the right way?
        self.u_model.write(modelMatrix.to_bytes())

        normalMatrix = glm.transpose(glm.inverse(glm.mat3(modelMatrix)))

        self.u_normal_matrix.write(normalMatrix.to_bytes())

        # ==== MESH SETUP =========================================================

        self.meshes: list[Mesh] = []

        textureFiles = sorted(TEXTURE_PATH.iterdir())

        for i, (geom, idx) in enumerate(zip(geomDataList, indexList)):
            # Skip empty files/textures
            if geom.size == 0 or idx.size == 0:
                continue

            # Retrieve the current texture
            texturePath = textureFiles[i]

            # Add it to meshes under the correct index and assigned to the correct geometry
            self.meshes.append(Mesh(self.ctx, self.program, geom, idx, texturePath))

        # ==== CAMERA SETUP =======================================================

        self.lookAtPoint = self.bounds.center
        self.up = glm.vec3(0.0, 1.0, 0.0)
        self.radius = self.bounds.radius if self.bounds.radius > 0 else 1

        cameraDistance = 2.0 * self.radius  # Camera will be 2 radii away

        # Camera position
        self.initialEye = self.lookAtPoint + glm.vec3(0, 0, cameraDistance)

        # ---- PROJECTION MATRIX --------------------------------------------------

        fov = 45
        aspect = self.WIDTH / self.HEIGHT
        nearPlane = 0.1 * self.radius
        farPlane = 10 * self.radius

        projectionMatrix = glm.perspective(
            glm.radians(fov), aspect, nearPlane, farPlane
        )

        self.u_proj.write(projectionMatrix.to_bytes())

    # ==== MAIN LOOP ==========================================================

    def run(self):
        clock = pygame.time.Clock()
        running = True

        cameraAngleY = 0
        cameraEye = self.initialEye

        LIGHT_ANGLE_DELTA = 5  # In degrees

        while running:
            for event in pygame.event.get():
                # Window closing
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    # Pausing/unpausing camera with space
                    if event.key == pygame.K_SPACE:
                        self.cameraPaused = not self.cameraPaused

                    # Changing light type with L
                    if event.key == pygame.K_l:
                        self.isPointLight = not self.isPointLight

                    # Moving light with arrow key left/right
                    if event.key == pygame.K_LEFT:
                        self.lightAngleY -= glm.radians(LIGHT_ANGLE_DELTA)
                    if event.key == pygame.K_RIGHT:
                        self.lightAngleY += glm.radians(LIGHT_ANGLE_DELTA)

            # ---- CAMERA ORBITING ----------------------------------------------------

            # TODO: Understand math

            if not self.cameraPaused:
                # Update camera's Y angle with respect to time , then transform it
                cameraAngleY = (pygame.time.get_ticks() / 1000) % (2 * np.pi)

                # Compute camera position
                cameraPosReal: glm.vec3 = self.initialEye - self.lookAtPoint

                # Compute the rotation we wish to give it
                cameraRotation = glm.rotate(
                    glm.mat4(1.0), cameraAngleY, glm.vec3(0.0, 1.0, 0.0)
                )

                # Compute final orbiting camera pos
                cameraEye = (
                    glm.vec3(
                        cameraRotation
                        * glm.vec4(cameraPosReal.x, cameraPosReal.y, cameraPosReal.z, 1)
                    )
                    + self.lookAtPoint
                )

                view = glm.lookAt(cameraEye, self.lookAtPoint, self.up)

                self.u_view.write(view.to_bytes())
                self.u_eye_pos.write(cameraEye.to_bytes())

            # ---- LIGHTING -----------------------------------------------------------

            # TODO: Understand math

            lightDistance = 3 * self.radius
            lightX = lightDistance * glm.sin(self.lightAngleY)
            lightZ = lightDistance * glm.cos(self.lightAngleY)
            lightPos = self.lookAtPoint + glm.vec3(lightX, self.bounds.center.y, lightZ)

            if self.isPointLight:
                # A point light has w = 1
                lightVec4 = glm.vec4(lightPos.x, lightPos.y, lightPos.z, 1.0)
            else:
                lightDir = glm.normalize(lightPos - self.lookAtPoint)
                lightVec4 = glm.vec4(lightDir.x, lightDir.y, lightDir.z, 0.0)

            self.u_light.write(lightVec4.to_bytes())

            # ---- RENDERING ----------------------------------------------------------

            CLEAR_COLOR = (126.0 / 255, 128.0 / 255, 28.00 / 255)

            self.ctx.clear(color=CLEAR_COLOR)

            for mesh in self.meshes:
                # Use albedo for mesh rendering
                if mesh.texture:
                    self.u_has_texture.value = True
                    mesh.texture.use(location=0)

                # If no texture for this mesh, specify
                else:
                    self.u_has_texture.value = False

                mesh.render()

            pygame.display.flip()

            TARGET_FPS = 60

            clock.tick(TARGET_FPS)

    pygame.quit()


if __name__ == "__main__":
    app = GraphicsEngine()
    app.run()
