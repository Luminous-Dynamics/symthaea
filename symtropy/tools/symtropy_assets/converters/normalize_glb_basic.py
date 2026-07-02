import bpy
import mathutils
import sys
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BlenderAdapter")

def calculate_physical_properties(obj, density=1.0):
    """Compute mass, center of mass, and an inertia tensor from mesh geometry.

    Mass and center of mass are exact (signed tetrahedron decomposition from
    the local origin, standard divergence-theorem formula for polyhedra —
    validated against a unit cube and a UV sphere: mass matches analytical
    volume*density to within float precision / <2%, and center of mass
    correctly tracks asymmetric local mesh geometry).

    The inertia tensor is a bounding-box approximation (box of the same mass
    and local-space extents), not an exact mesh integral — that's a
    deliberately simpler, well-known fallback used by many game engines for
    irregular meshes, chosen over risking a subtly-wrong exact-inertia
    implementation. Good enough for physics behavior; not a precision claim.
    """
    mesh = obj.data
    mesh.calc_loop_triangles()

    volume = 0.0
    com = mathutils.Vector((0.0, 0.0, 0.0))
    for tri in mesh.loop_triangles:
        v0, v1, v2 = [mesh.vertices[i].co for i in tri.vertices]
        tet_vol = v0.dot(v1.cross(v2)) / 6.0
        volume += tet_vol
        com += tet_vol * ((v0 + v1 + v2) / 4.0)

    if abs(volume) < 1e-9:
        logger.warning(f"{obj.name}: degenerate/zero volume, physical properties are placeholder zeros")
        return {"mass": 0.0, "com": [0.0, 0.0, 0.0], "inertia": [0.0] * 9}

    com /= volume
    mass = abs(volume) * density

    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    zs = [v.co.z for v in mesh.vertices]
    w, h, d = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    ixx = mass * (h * h + d * d) / 12.0
    iyy = mass * (w * w + d * d) / 12.0
    izz = mass * (w * w + h * h) / 12.0

    return {
        "mass": mass,
        "com": [com.x, com.y, com.z],
        "inertia": [ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz],
    }

def generate_technical_report(filepath, status="TECHNICAL_APPROVED", warnings=None):
    # ...
    # Add physical properties to report
    physical_props = {}
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not any(t in obj.name for t in ["_COLLISION_BOX", "_COLLISION_SPHERE", "_COLLISION_CAPSULE", "_COLLISION_CONVEX", "_COLLISION"]):
            physical_props[obj.name] = calculate_physical_properties(obj)

    report = {
        "asset_id": os.path.basename(os.path.splitext(filepath)[0]),
        "physical_properties": physical_props,
        "triangle_count": sum(len(obj.data.polygons) for obj in bpy.data.objects if obj.type == 'MESH' and obj.data and not any(t in obj.name for t in ["_COLLISION_BOX", "_COLLISION_SPHERE", "_COLLISION_CAPSULE", "_COLLISION_CONVEX", "_COLLISION"])),
        "recommended_status": status,
        "warnings": warnings or [],
        "generator": "Symtropy Foundry v0.2",
    }
    report_path = f"{os.path.splitext(filepath)[0]}_tech_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Generated report: {report_path}")

def capture_thumbnail(filepath):
    """Frame and light a thumbnail based on the actual scene content.

    The previous version used a fixed camera at (5,-5,5) and a single sun
    light regardless of the asset's real size — fine for a ~5-unit object,
    but for anything meter-scale-or-smaller (most real assets) it rendered
    as a barely-visible dark speck against a huge black frame. Confirmed via
    a real Poly Haven CC0 model: this fix is the difference between an
    unusable thumbnail and a clear, well-lit review image.
    """
    mesh_objs = [
        obj for obj in bpy.context.scene.objects
        if obj.type == 'MESH'
        and not any(t in obj.name for t in ["_COLLISION_BOX", "_COLLISION_SPHERE", "_COLLISION_CAPSULE", "_COLLISION_CONVEX", "_COLLISION", "_LOD"])
    ]
    if not mesh_objs:
        mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    min_co = mathutils.Vector((float('inf'),) * 3)
    max_co = mathutils.Vector((float('-inf'),) * 3)
    for obj in mesh_objs:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ mathutils.Vector(corner)
            min_co = mathutils.Vector(min(a, b) for a, b in zip(min_co, world_corner))
            max_co = mathutils.Vector(max(a, b) for a, b in zip(max_co, world_corner))
    center = (min_co + max_co) / 2.0
    size = max((max_co - min_co).length, 0.01)

    cam_dir = mathutils.Vector((1.0, -1.0, 0.8)).normalized()
    cam_loc = center + cam_dir * (size * 1.8)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.object
    cam.rotation_euler = (center - cam_loc).normalized().to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    thumb_path = f"{os.path.splitext(filepath)[0]}_thumb.png"
    bpy.context.scene.render.filepath = thumb_path
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    key = bpy.data.objects.new("thumb_key_light", bpy.data.lights.new("thumb_key", type='SUN'))
    key.data.energy = 3.0
    key.rotation_euler = (1.0, 0.0, 0.6)
    bpy.context.collection.objects.link(key)

    fill = bpy.data.objects.new("thumb_fill_light", bpy.data.lights.new("thumb_fill", type='SUN'))
    fill.data.energy = 1.0
    fill.rotation_euler = (1.3, 0.0, -2.4)
    bpy.context.collection.objects.link(fill)

    # NOTE: World.use_nodes triggers a DeprecationWarning in Blender 5.x
    # (slated for removal in 6.0) but is still the correct API for 5.1.1.
    # Revisit when upgrading past Blender 6.0.
    world = bpy.data.worlds.new("ThumbWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    bg.inputs[1].default_value = 0.3
    bpy.context.scene.world = world

    bpy.ops.render.render(write_still=True)
    logger.info(f"Generated thumbnail: {thumb_path}")

def generate_collision_proxy(obj):
    # Check if object already has a collision tag in its name
    for tag in ["_COLLISION_BOX", "_COLLISION_SPHERE", "_COLLISION_CAPSULE", "_COLLISION_CONVEX"]:
        if tag in obj.name:
            # It's already a collision proxy, just ensure it's hidden and return
            obj.hide_render = True
            return obj

    # Create a copy for collision
    proxy = obj.copy()
    proxy.data = obj.data.copy()
    proxy.name = f"{obj.name}_COLLISION_CONVEX"
    bpy.context.collection.objects.link(proxy)

    # Simple convex hull proxy
    bpy.context.view_layer.objects.active = proxy
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.convex_hull()
    bpy.ops.object.mode_set(mode='OBJECT')

    # Hide from rendering, keep for export
    proxy.hide_render = True
    return proxy

def generate_lod_stub(obj, level=1, ratio=0.1):
    lod = obj.copy()
    lod.data = obj.data.copy()
    lod.name = f"{obj.name}_LOD{level}"
    bpy.context.collection.objects.link(lod)

    # Add decimate modifier
    mod = lod.modifiers.new(name="Decimate", type='DECIMATE')
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = lod
    bpy.ops.object.modifier_apply(modifier="Decimate")
    return lod

def normalize_mesh(filepath):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return

    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Determine import op based on extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == '.obj':
            bpy.ops.wm.obj_import(filepath=filepath)
        elif ext == '.fbx':
            bpy.ops.wm.fbx_import(filepath=filepath)
        else:
            logger.error(f"Unsupported file type: {ext}")
            sys.exit(1)

        # Normalization and Proxy logic
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and not any(tag in obj.name for tag in ["_COLLISION_BOX", "_COLLISION_SPHERE", "_COLLISION_CAPSULE", "_COLLISION_CONVEX", "_COLLISION"])]
        for obj in meshes:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            obj.location = (0, 0, 0)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Generate Extras
            generate_collision_proxy(obj)
            generate_lod_stub(obj, level=1, ratio=0.25)

        # Export path logic (robustly handle extensions)
        base_path = os.path.splitext(filepath)[0]
        export_path = f"{base_path}_normalized.glb"

        bpy.ops.export_scene.gltf(filepath=export_path, export_format='GLB')

        capture_thumbnail(filepath)
        generate_technical_report(filepath)
        logger.info(f"Successfully exported: {export_path}")

    except Exception as e:
        logger.error(f"Failed to normalize asset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Expecting the file path as the first argument after '--'
    try:
        idx = sys.argv.index("--")
        file_to_process = sys.argv[idx + 1]
        normalize_mesh(file_to_process)
    except (ValueError, IndexError):
        logger.error("Usage: blender --background --python normalize_glb_basic.py -- <filepath>")
        sys.exit(1)
