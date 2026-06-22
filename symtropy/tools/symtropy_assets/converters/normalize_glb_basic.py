import bpy
import sys
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BlenderAdapter")

def calculate_physical_properties(obj):
    # Bypass for now due to API changes in Blender 5.1
    return {
        "mass": 1.0,
        "com": [0.0, 0.0, 0.0],
        "inertia": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
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
    # Basic camera setup for thumbnail
    bpy.ops.object.camera_add(location=(5, -5, 5))
    cam = bpy.context.object
    cam.rotation_euler = (1.1, 0, 0.78)
    bpy.context.scene.camera = cam

    # Setup rendering
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    thumb_path = f"{os.path.splitext(filepath)[0]}_thumb.png"
    bpy.context.scene.render.filepath = thumb_path

    # Simple light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

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
