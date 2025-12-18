"""
Isaac Sim 3D Layout Final Module
对应论文阶段 (4) 3D Scene Assembly (Part 2: Simulation)
功能：将处理好的 3D 资产导入 NVIDIA Isaac Sim，应用物理属性，构建可交互的仿真场景。

主要步骤原理：
1. 资产转换 (Asset Conversion): 将 GLB 格式转换为 Isaac Sim 原生支持的 USD 格式。
2. 场景加载: 加载背景房间 (Room Environment)。
3. 物体放置 (Object Placement):
   - 尺度对齐 (Scale): 对比当前 USD 模型的 Bounding Box 和 VLM 估算的物理尺寸 (cm)，计算缩放比例。
   - 位置映射 (Position): 将计算出的 cm 级坐标转换为 Isaac Sim 的 m 级坐标，并处理坐标系差异（如 Z 轴偏移）。
   - 旋转应用 (Rotation): 应用 DRO 阶段计算出的精确旋转角度。
4. 物理属性 (Physics):
   - 碰撞体 (Collision): 为物体添加凸包 (Convex Hull) 或 盒子 (Box) 碰撞体，确保仿真中的物理交互。
   - 刚体 (Rigid Body): 启用重力、摩擦力等物理模拟属性。
"""

import os
import time
import json
import sys
import numpy as np
import math
import asyncio
import shutil
import threading
import queue
from isaacsim import SimulationApp

# 启动 Isaac Sim 应用实例
simulation_app = SimulationApp({"headless": False})

import omni
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from pxr import Usd, UsdGeom, Gf, UsdPhysics, UsdLux, Sdf, UsdShade
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.viewports import set_camera_view

# 启用资产转换扩展 (用于 GLB -> USD) 适配Isaac Sim
enable_extension("omni.kit.asset_converter")


async def convert(in_file, out_file, load_materials=False):
    """Convert asset file to USD format"""
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    converter_context.ignore_materials = not load_materials

    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(in_file, out_file, progress_callback, converter_context)
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success

def asset_convert(input_file, output_file):
    """转换资产格式 (GLB -> USD)"""
    print(f"Converting {input_file} to {output_file}...")
    status = asyncio.get_event_loop().run_until_complete(
        convert(input_file, output_file, load_materials=True)
    )
    if not status:
        print(f"Error: Failed to convert {input_file}")
        return False
    else:
        print(f"Success: Converted {input_file} to {output_file}")
        return True

def normalize_asset_name(name):
    """Smartly handle asset names"""
    words = name.split()
    capitalized_words = [word.capitalize() for word in words]
    return ''.join(capitalized_words)

def check_asset_exists(asset_path):
    """Check if asset file exists"""
    if not os.path.exists(asset_path):
        print(f"Error: Asset file '{asset_path}' does not exist")
        return False
    return True

def set_angles(prim_path, euler_angles):
    """设置物体欧拉角旋转 (应用 DRO 优化后的角度)"""
    obj = XFormPrim(prim_path)
    current_position, _ = obj.get_world_pose()
    euler_angles_rad = tuple(math.radians(angle) for angle in euler_angles)
    new_orientation = euler_angles_to_quat(euler_angles_rad)
    obj.set_world_pose(position=current_position, orientation=new_orientation)

def set_position(prim_path, position):
    """设置物体位置 (应用 TSA 对齐后的坐标)"""
    obj = XFormPrim(prim_path)
    _, current_orientation = obj.get_world_pose()
    new_position = Gf.Vec3d(position[0], position[1], position[2])
    obj.set_world_pose(position=new_position, orientation=current_orientation)

def toggle_physics_for_objects(stage, object_prims, enable_dynamic, include_table=True, table_prim_path=None):
    """
    切换物体的物理状态：Kinematic <-> Dynamic
    enable_dynamic: True = 启用动力学（物体会掉落）, False = 运动学（固定）
    include_table: 是否包括桌子（桌子也会受物理影响）
    table_prim_path: 桌子的 Prim 路径（动态传入，支持 Table_0, Table_1 等）
    """
    count = 0
    for prim_path in object_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            rigid_body = UsdPhysics.RigidBodyAPI(prim)
            if rigid_body:
                if enable_dynamic:
                    # 启用动力学：关闭 Kinematic，物体会受重力影响
                    rigid_body.GetKinematicEnabledAttr().Set(False)
                else:
                    # 运动学：启用 Kinematic，物体固定不动
                    rigid_body.GetKinematicEnabledAttr().Set(True)
                count += 1
    
    # 处理桌子（动态识别）
    if include_table and table_prim_path:
        table_prim = stage.GetPrimAtPath(table_prim_path)
        if table_prim.IsValid():
            rigid_body = UsdPhysics.RigidBodyAPI(table_prim)
            if rigid_body:
                if enable_dynamic:
                    rigid_body.GetKinematicEnabledAttr().Set(False)
                    count += 1
                    print(f"[Physics] 桌子 ({table_prim_path}) 也启用了动力学（会倾倒或稳定）")
                else:
                    rigid_body.GetKinematicEnabledAttr().Set(True)
                    count += 1
    
    if enable_dynamic:
        print(f"[Physics] 已启用 {count} 个物体的动力学模拟")
    else:
        print(f"[Physics] 已禁用 {count} 个物体的动力学模拟（物品固定）")

def toggle_collision_visibility(stage, show_collision=True):
    """
    切换碰撞包围盒的可见性（用于调试）
    """
    collision_count = 0
    for prim in stage.Traverse():
        if "CollisionCube" in str(prim.GetPath()):
            imageable = UsdGeom.Imageable(prim)
            if show_collision:
                imageable.MakeVisible()
                collision_count += 1
            else:
                imageable.MakeInvisible()
                collision_count += 1
    
    if show_collision:
        print(f"[BBox] 已显示 {collision_count} 个碰撞包围盒（绿色半透明立方体）")
    else:
        print(f"[BBox] 已隐藏 {collision_count} 个碰撞包围盒")

def input_listener_thread(cmd_queue, stop_event):
    """
    后台线程：监听用户输入命令
    格式: x 角度 角速度  (绕X轴以指定角速度旋转到目标角度，例: x 50 2)
          y 角度 角速度  (绕Y轴旋转)
          z 角度 角速度  (绕Z轴旋转)
          physics on/off (启用/禁用物理模拟)
          bbox on/off (显示/隐藏碰撞包围盒)
          reset (恢复所有物体到初始位置和姿态)
          quit  (退出)
    """
    print("\n" + "="*60)
    print("交互命令:")
    print("  输入 'x 角度 角速度' - 让桌子绕X轴旋转 (例: x 50 2)")
    print("                       角度=目标角度，角速度=度/秒")
    print("  输入 'y 角度 角速度' - 让桌子绕Y轴旋转")
    print("  输入 'z 角度 角速度' - 让桌子绕Z轴旋转")
    print("  输入 'physics on'    - 启用物理模拟（物品和桌子都会动）")
    print("  输入 'physics off'   - 禁用物理模拟")
    print("  输入 'bbox on'       - 显示碰撞包围盒")
    print("  输入 'bbox off'      - 隐藏碰撞包围盒")
    print("  输入 'reset'         - 恢复所有物体到初始状态")
    print("  输入 'quit'          - 退出程序")
    print("="*60 + "\n")
    
    while not stop_event.is_set():
        try:
            user_input = input(">> ").strip()
            if user_input:
                cmd_queue.put(user_input)
                if user_input.lower() == 'quit':
                    break
        except EOFError:
            break
        except Exception as e:
            print(f"输入错误: {e}")

# Jack Wang here for physical property
def create_rigid_collision(stage, prim_path, collision_approximation="boundingCube", is_kinematic=False, mass=1.0):
    """
    为物体添加刚体和碰撞属性 (Physics & Collision)
    FIX: 直接创建 Collision Shape Geometry + 正确启用物理属性
    """
    prim = stage.GetPrimAtPath(prim_path)
    
    # 1. 应用刚体 API (Rigid Body) 并启用重力
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    
    # CRITICAL: 显式启用刚体并设置属性
    if is_kinematic:
        # 运动学刚体（不受重力影响，用于地面/桌子）
        rigid_body.CreateKinematicEnabledAttr(True)
        print(f"  [Physics] Applied RigidBodyAPI (Kinematic) to {prim_path}")
    else:
        # 动力学刚体（受重力影响）
        rigid_body.CreateRigidBodyEnabledAttr(True)
        
        # 设置质量
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)
        
        print(f"  [Physics] Applied RigidBodyAPI (Dynamic, mass={mass}) to {prim_path}")
    
    # 2. 计算物体的局部 Bounding Box（重要：使用 LocalBound 而不是 WorldBound）
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    
    # 使用 ComputeLocalBound 获取局部坐标系的包围盒
    bbox = bbox_cache.ComputeLocalBound(prim)
    
    if bbox:
        bbox_range = bbox.GetRange()
        min_point = bbox_range.GetMin()
        max_point = bbox_range.GetMax()
        
        # 计算尺寸和中心（在局部坐标系中）
        size = [max_point[0] - min_point[0], 
                max_point[1] - min_point[1], 
                max_point[2] - min_point[2]]
        
        center = [(max_point[0] + min_point[0]) / 2.0,
                  (max_point[1] + min_point[1]) / 2.0,
                  (max_point[2] + min_point[2]) / 2.0]
        
        print(f"  [Physics] Local Bounding Box - Size: {size}, Center: {center}")
        
        # 3. 创建一个 Cube 作为碰撞形状
        collision_cube_path = f"{prim_path}/CollisionCube"
        collision_cube = UsdGeom.Cube.Define(stage, collision_cube_path)
        
        # 设置 Cube 的尺寸（USD Cube 的默认大小是 2x2x2，所以需要缩放）
        collision_cube.GetSizeAttr().Set(2.0)  # 默认大小
        
        # FIX: 按照 USD 标准顺序添加 Xform 操作：Translate -> Rotate -> Scale
        # 这样可以避免 "Incompatible xformOpOrder" 警告
        xformable = UsdGeom.Xformable(collision_cube)
        
        # 1. 先添加 Translate（位置）
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*center))
        
        # 2. 再添加 Scale（尺寸）
        scale = [size[0]/2.0, size[1]/2.0, size[2]/2.0]
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(*scale))
        
        # 4. 给 Cube 添加 CollisionAPI
        UsdPhysics.CollisionAPI.Apply(collision_cube.GetPrim())
        
        # 5. 设置碰撞体材质（半透明绿色，用于调试）
        material_path = f"{collision_cube_path}/Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.0, 1.0, 0.0))  # 绿色
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.3)  # 30% 不透明度
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        binding_api = UsdShade.MaterialBindingAPI(collision_cube)
        binding_api.Bind(material)
        
        # 6. 默认隐藏碰撞体（用户可以通过 'bbox on' 命令显示）
        imageable = UsdGeom.Imageable(collision_cube)
        imageable.MakeInvisible()
        
        print(f"  [Physics] Created Collision Cube (invisible, use 'bbox on' to show) at {collision_cube_path}")
        print(f"  [Physics] Collision Cube - Scale: {scale}, Position: {center}")
        
        return True
    else:
        print(f"  [Physics] ERROR: Could not compute bounding box for {prim_path}")
        return False

def calculate_bounding_box(stage, prim_path):
    """计算物体在 USD 中的 Bounding Box，返回尺寸和中心点"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"Error: Invalid prim at path {prim_path}")
        return None, None

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default'])
    bbox = bbox_cache.ComputeWorldBound(prim)
    
    if bbox:
        bounds = bbox.ComputeAlignedBox()
        min_point = bounds.GetMin()
        max_point = bounds.GetMax()
        
        size = [max_point[0] - min_point[0], 
                max_point[1] - min_point[1], 
                max_point[2] - min_point[2]]
        
        center = [(max_point[0] + min_point[0])/2, 
                 (max_point[1] + min_point[1])/2, 
                 (max_point[2] + min_point[2])/2]
        
        return size, center
    else:
        print(f"Cannot calculate bounding box for {prim_path}")
        return None, None

def set_object_scale(stage, prim_path, scale_factors):
    """设置物体缩放 (Scale)"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: Invalid prim at path {prim_path}")
        return

    xform = UsdGeom.Xformable(prim)
    scale_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == 'xformOp:scale':
            scale_op = op
            break

    if scale_op is None:
        scale_op = xform.AddScaleOp()

    scale_op.Set(Gf.Vec3f(*scale_factors))
    print(f"Applied scale {scale_factors} to {prim_path}")

# new
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

def _physics_token(name, default=None):
    return getattr(UsdPhysics.Tokens, name, default)

# 动态构建最佳可用碰撞近似映射 (Fallback 策略)
APPROX_CANDIDATES = {
    "box":               _physics_token("boundingCube"),
    "sphere":            _physics_token("boundingSphere"),
    "hull":              _physics_token("convexHull"),
    "vhacd":             _physics_token("convexDecomposition"),
    "mesh_simplify":     _physics_token("meshSimplification"),
    "none":              _physics_token("none"),
}
# Filter out None values
APPROX_MAP = {k: v for k, v in APPROX_CANDIDATES.items() if v is not None}

def print_supported_collision_approximations():
    print("[collider] Supported approximation tokens in this build:")
    for k, v in APPROX_MAP.items():
        print(f"  - {k:>13} -> {v}")

def set_collision_approx_for_submeshes(stage, root_prim_path, mode="box"):
    """
    Robustly set collision approximation for meshes under a root prim.
    """
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        print(f"[collider] invalid prim: {root_prim_path}")
        return

    # Direct string fallback if token lookup fails
    # Force use "convexHull" or "convexDecomposition" string if mode matches
    approx_value = mode
    if mode == "box": approx_value = "boundingCube"
    if mode == "hull": approx_value = "convexHull" 
    if mode == "vhacd": approx_value = "convexDecomposition"
    
    # Try to use Physics Tokens if available, but fallback to strings
    if mode in APPROX_MAP:
        approx_value = APPROX_MAP[mode]

    print(f"[collider] Processing {root_prim_path} with mode='{mode}' (value='{approx_value}')")

    count = 0
    
    # Iterate over ALL descendants
    for prim in Usd.PrimRange(root):
        # Check if it's a Mesh
        if prim.IsA(UsdGeom.Mesh):
            path_str = prim.GetPath().pathString
            
            # Filter out materials/looks
            if "/Materials" in path_str or "/Looks" in path_str or "material" in path_str.lower():
                continue
                
            print(f"  -> Found Mesh: {path_str}")
            
            # Apply Collision API
            coll = UsdPhysics.CollisionAPI.Apply(prim)
            
            # Set Approximation
            # Try all known API variations
            success = False
            attr = None
            
            # 1. GetApproximationAttr
            if hasattr(coll, "GetApproximationAttr"):
                attr = coll.GetApproximationAttr()
            # 2. GetPhysicsApproximationAttr
            elif hasattr(coll, "GetPhysicsApproximationAttr"):
                attr = coll.GetPhysicsApproximationAttr()
            # 3. Create...
            elif hasattr(coll, "CreateApproximationAttr"):
                attr = coll.CreateApproximationAttr()
            elif hasattr(coll, "CreatePhysicsApproximationAttr"):
                attr = coll.CreatePhysicsApproximationAttr()
                
            if attr:
                attr.Set(approx_value)
                success = True
            
            if success:
                count += 1
                # Also apply RigidBodyAPI to the ROOT object if this is a child mesh
                # to ensure they move together? No, RigidBody is on the Xform parent.
                # But Collision needs to be on the Mesh.
            else:
                print(f"  [Error] Failed to set approximation attr for {path_str}")

    if count == 0:
        print(f"[collider] WARNING: No meshes found under {root_prim_path} to set collision!")
        
        # Fallback: Force apply to the ROOT prim (Xform)
        # This is valid in USD Physics! It will approximate the collision for the whole subtree.
        print(f"  -> FORCE applying collision to ROOT: {root_prim_path}")
        
        coll = UsdPhysics.CollisionAPI.Apply(root)
        
        # For Xform roots, "convexHull" or "convexDecomposition" works well.
        # "none" (Triangle Mesh) might NOT work on Xform, so we fallback to convexHull if mode is none.
        force_mode = approx_value
        if mode == "none":
            force_mode = "convexDecomposition" # Fallback for table if mesh not found
            print(f"  -> Mode was 'none' but cannot apply to Xform, switching to '{force_mode}'")
            
        success = False
        attr = None
        
        if hasattr(coll, "GetApproximationAttr"): attr = coll.GetApproximationAttr()
        elif hasattr(coll, "CreateApproximationAttr"): attr = coll.CreateApproximationAttr()
        elif hasattr(coll, "GetPhysicsApproximationAttr"): attr = coll.GetPhysicsApproximationAttr()
        elif hasattr(coll, "CreatePhysicsApproximationAttr"): attr = coll.CreatePhysicsApproximationAttr()
            
        if attr:
            attr.Set(force_mode)
            print(f"  -> Successfully set root collision approximation to '{force_mode}'")
        else:
            print(f"  [Error] Failed to set root approximation attr")

    print(f"[collider] Successfully set collision for {count} meshes (or root).")



def add_object_to_scene(sized_mesh_path, usd_cache_dir, stage, object_name, object_data, rotation_angle ,is_main_object):
    """
    核心函数：将单个物体添加到场景中
    步骤：
    1. 转换/加载 USD 资产。
    2. 计算并应用缩放 (Scale)：根据 VLM 预测尺寸 (cm) 和当前模型尺寸 (Isaac units) 的比例。
    3. 设置位置 (Position)：从 cm 转换为 m，并调整坐标轴方向。
    4. 设置旋转 (Rotation)：应用 Z 轴旋转。
    5. 添加物理碰撞属性 (Collision)。
    """
    # Clean object name, remove special characters
    import re
    clean_object_name = re.sub(r'[^\w\s-]', '', object_name)
    clean_object_name = re.sub(r'^#+', '', clean_object_name).strip()
    
    if not clean_object_name:
        clean_object_name = f"Asset_{hash(object_name) % 10000}"
    
    print(f"Original name: '{object_name}' -> Cleaned name: '{clean_object_name}'")
    

    # Create unique prim path
    normalized_name = normalize_asset_name(clean_object_name)
    prim_path = f"/World/{normalized_name}"
    
    # GLB file path
    glb_path = os.path.join(sized_mesh_path, f"{object_name}.glb")
    
    # Check if GLB file exists
    if not check_asset_exists(glb_path):
        print(f"Skipping asset '{object_name}', file not found: {glb_path}")
        return False
    
    # USD file path
    usd_path = os.path.join(usd_cache_dir, f"{clean_object_name}.usd")
    
    # If USD file does not exist or GLB file is updated, perform conversion
    if not os.path.exists(usd_path) or os.path.getmtime(glb_path) > os.path.getmtime(usd_path):
        if not asset_convert(glb_path, usd_path):
            print(f"Cannot convert asset '{object_name}' to USD format")
            return False
    
    print(f"Adding asset: {object_name}")
    
    try:
        # Step 1: Add USD file to scene (引用 USD 文件)
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        print(f"Added asset '{object_name}' to scene")
        
        import omni.kit.app
        omni.kit.app.get_app().update()
        
        # FIX: Move rotation back to Step 4. 
        # Apply scaling in local space first (Step 2), then position, then rotate.
        
        # Step 4: Set rotation angle (应用旋转)
        # Fix: 模型是 Y-up，Isaac 是 Z-up，需要绕 X 轴旋转 -90 度 (270度) 才能正立
        actual_rotation_angle = rotation_angle
        rotation = [-90, 0, actual_rotation_angle]
        set_angles(prim_path, rotation)
        print(f"Applied rotation: {actual_rotation_angle}° around Z-axis with -90° X-axis fix")
        
        # Step 2: Get current size and calculate required scale (计算缩放)
        # FIX: Revert to standard scaling logic. Complex axis mapping caused distortion.
        # 使用均匀缩放 (Uniform Scale) 避免变形
        
        actual_size, actual_center = calculate_bounding_box(stage, prim_path)
        if actual_size:
            expected_size_cm = object_data["size"]  # Expected size in cm
            expected_size_m = [s / 100.0 for s in expected_size_cm]  # Convert to meters
            
            print(f"Expected size (m): {[round(s, 4) for s in expected_size_m]}")
            print(f"Current size (Local Bounds): {[round(s, 4) for s in actual_size]}")
            
            if actual_size[0] > 0 and actual_size[1] > 0 and actual_size[2] > 0:
                # Calculate scale factors for each axis
                s_x = expected_size_m[0] / actual_size[0]
                s_y = expected_size_m[1] / actual_size[1]
                s_z = expected_size_m[2] / actual_size[2]
                
                # FIX: Use Uniform Scale to prevent distortion
                # Take the average or geometric mean to maintain aspect ratio
                # Or use the axis with the largest dimension as the reference
                # Here we use the max dimension to ensure it fits roughly
                
                # However, for the TABLE, we might want non-uniform scale to match room
                # For small objects, we prefer uniform.
                
                if is_main_object:
                     # Main object (Table) might need non-uniform scaling to fit layout
                     # But we must be careful about axis alignment (Y-up vs Z-up)
                     # Since we apply rotation LATER, we scale in Local Frame.
                     # Most GLB are Y-up: Local Y is "Height". 
                     # Isaac Z is "Height". 
                     # Target Size is [L, W, H] (World Frame).
                     # So we should map: Target H -> Local Y.
                     
                     # Simple heuristic: Match sorted dimensions to minimize distortion
                     local_dims = sorted([(actual_size[0], 0), (actual_size[1], 1), (actual_size[2], 2)], key=lambda x: x[0])
                     target_dims = sorted(expected_size_m)
                     
                     scale_map = [1.0, 1.0, 1.0]
                     # Map smallest local to smallest target, etc.
                     for i in range(3):
                         axis_idx = local_dims[i][1]
                         scale_map[axis_idx] = target_dims[i] / local_dims[i][0]
                         
                     final_scale = scale_map
                     print(f"Smart Non-Uniform Scale (Sorted Mapping): {final_scale}")
                     
                else:
                    # Small objects: Use Uniform Scale
                    # Use the median scale factor to be safe
                    import statistics
                    uniform_scale = statistics.median([s_x, s_y, s_z])
                    final_scale = [uniform_scale, uniform_scale, uniform_scale]
                    print(f"Uniform Scale Factor: {uniform_scale}")

                # Apply scale
                set_object_scale(stage, prim_path, final_scale)
                
                # Recalculate scaled size
                omni.kit.app.get_app().update()
                final_size, final_center = calculate_bounding_box(stage, prim_path)
                if final_size:
                    print(f"Final size (m): {[round(s, 4) for s in final_size]}")
        
        # Step 3: Set position (位置映射)
        # position in layout_pose is already final, convert to meters
        pose_position = object_data["pose"]  # [x, y, z] in cm
        
        # Convert to Isaac Sim coordinate system: invert x-axis, cm to m, and adjust height to fit room floor
        room_floor_offset = 0.0
        
        adjusted_position = [
            -pose_position[0] / 100.0,  # invert x-axis
            pose_position[1] / 100.0,   # y-axis
            pose_position[2] / 100.0 + room_floor_offset    # z-axis
        ]
        
        print(f"Layout pose (cm): {pose_position}")
        print(f"Adjusted position (m, with room floor offset): {[round(p, 4) for p in adjusted_position]}")
        
        set_position(prim_path, adjusted_position)
        
        # Verify position after setting
        obj = XFormPrim(prim_path)
        final_position, _ = obj.get_world_pose()
        print(f"Final position: {[round(p, 4) for p in final_position]}")
        
        # Step 5: Add physical collision (添加碰撞体)
        # FIX: 所有物体（包括桌子）初始都设为 Kinematic（固定），避免一开始就掉落
        # 用户可以通过命令 'physics on' 来启用动力学模拟
        if is_main_object:
            print_supported_collision_approximations()
            # 桌子质量应该比小物件重很多
            create_rigid_collision(stage, prim_path, collision_approximation="boundingCube", 
                                   is_kinematic=True, mass=50.0)
            print(f"  [Physics] {object_name} is Kinematic initially (mass=50.0 kg, will have physics when enabled)")
                
        else:
            # 根据物体大小设置质量（小物体轻一些）
            estimated_volume = (object_data["size"][0] * object_data["size"][1] * object_data["size"][2]) / 1000000.0  # cm³ to m³
            estimated_mass = max(0.1, estimated_volume * 500)  # 假设密度 500 kg/m³
            
            # 初始也设为 Kinematic，等用户输入命令后再启用动力学
            create_rigid_collision(stage, prim_path, collision_approximation="boundingCube",
                                   is_kinematic=True, mass=estimated_mass)
            print(f"  [Physics] {object_name} is Kinematic initially (mass={estimated_mass:.2f} kg)")

        print(f"{object_name} successfully placed")
        return (True, prim_path)
        
    except Exception as e:
        print(f"Error adding asset '{object_name}': {e}")
        import traceback
        traceback.print_exc()
        return (False, None)

def load_layout_data(layout_pose_path, layout_rotation_path):
    """Load layout data (读取布局和旋转 JSON 数据)"""
    try:
        # Load pose data
        with open(layout_pose_path, 'r', encoding='utf-8') as f:
            pose_data = json.load(f)
        
        # Load rotation data
        with open(layout_rotation_path, 'r', encoding='utf-8') as f:
            rotation_data = json.load(f)
        
        print(f"Loaded layout data:")
        print(f"  - Pose data for {len(pose_data['objects'])} objects")
        print(f"  - Rotation data for {len(rotation_data)} entries")
        print(f"  - Main object: {pose_data['main_object']}")
        
        return pose_data, rotation_data
        
    except Exception as e:
        print(f"Error loading layout data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def clear_usd_cache(usd_cache_dir):
    """Clear USD cache directory"""
    if os.path.exists(usd_cache_dir):
        try:
            shutil.rmtree(usd_cache_dir)
            print(f"Cleared USD cache directory: {usd_cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to clear USD cache directory: {e}")
    
    # Recreate directory
    os.makedirs(usd_cache_dir, exist_ok=True)
    print(f"Created fresh USD cache directory: {usd_cache_dir}")

def load_room_environment(stage, pipeline_dir):
    """Load room environment USD file (加载背景房间)"""
    room_usd_path = os.path.join(pipeline_dir, "background_room/room.usd")
    
    if not os.path.exists(room_usd_path):
        print(f"Warning: Room USD file not found at {room_usd_path}")
        return False
    
    try:
        # Check USD file content first
        print(f"Checking room USD file: {room_usd_path}")
        temp_stage = Usd.Stage.Open(room_usd_path)
        if temp_stage:
            root_prim = temp_stage.GetDefaultPrim()
            if root_prim:
                print(f"Room USD default prim: {root_prim.GetPath()}")
            else:
                print("Room USD has no default prim")
            
            # List all prims
            all_prims = [prim for prim in temp_stage.Traverse()]
            print(f"Room USD contains {len(all_prims)} prims:")
            for prim in all_prims[:10]:  # Show only first 10
                print(f"  - {prim.GetPath()} (type: {prim.GetTypeName()})")
            if len(all_prims) > 10:
                print(f"  ... and {len(all_prims) - 10} more prims")
        
        # Method 1: Use add_reference_to_stage
        room_prim_path = "/World/Room"
        print(f"Trying to load room with add_reference_to_stage...")
        add_reference_to_stage(usd_path=room_usd_path, prim_path=room_prim_path)
        
        import omni.kit.app
        omni.kit.app.get_app().update()
        
        # Check load result
        room_prim = stage.GetPrimAtPath(room_prim_path)
        if room_prim and room_prim.IsValid():
            print(f"Room prim created at {room_prim_path}")
            # Check child prims
            children = room_prim.GetChildren()
            print(f"Room has {len(children)} children:")
            for child in children[:5]:  # Show only first 5
                print(f"  - {child.GetPath()} (type: {child.GetTypeName()})")
            
            if len(children) == 0:
                print("Room prim has no children, trying alternative method...")
                
                # Method 2: Merge stage directly
                try:
                    stage.GetRootLayer().subLayerPaths.append(room_usd_path)
                    omni.kit.app.get_app().update()
                    print("Tried sublayer method")
                except Exception as e2:
                    print(f"Sublayer method failed: {e2}")
                    
                    # Method 3: Use payload
                    try:
                        room_prim.GetPayloads().AddPayload(room_usd_path)
                        omni.kit.app.get_app().update()
                        print("Tried payload method")
                    except Exception as e3:
                        print(f"Payload method failed: {e3}")
        
        print(f"Successfully processed room environment from {room_usd_path}")
        return True
        
    except Exception as e:
        print(f"Error loading room environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def isaac_main(output_assets_dir):
    """Main function"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = script_dir

    sized_mesh_path = os.path.join(output_assets_dir, "centered_mesh")
    layout_pose_path = os.path.join(output_assets_dir, "layout_json", "layout_pose.json")
    layout_rotation_path = os.path.join(output_assets_dir, "layout_json", "layout_rotation.json")

    # Create USD cache directory and clear it
    usd_cache_dir = os.path.join(script_dir, "usd_cache")
    clear_usd_cache(usd_cache_dir)

    print("Isaac Sim 3D Layout Final System")
    print(f"Script directory: {script_dir}")
    print(f"Pipeline directory: {pipeline_dir}")
    print(f"Room USD path: {os.path.join(pipeline_dir, 'room.usd')}")
    print(f"Sized mesh path: {sized_mesh_path}")
    print(f"Layout pose path: {layout_pose_path}")
    print(f"Layout rotation path: {layout_rotation_path}")
    print(f"USD cache directory: {usd_cache_dir}")
    
    # Load layout data
    pose_data, rotation_data = load_layout_data(layout_pose_path, layout_rotation_path)
    if not pose_data or not rotation_data:
        print("Failed to load layout data")
        return
    
    try:
        # Initialize world and physics engine
        my_world = World(stage_units_in_meters=1.0, physics_prim_path="/World/physicsScene")
        stage = omni.usd.get_context().get_stage()
        
        # CRITICAL: 显式设置物理场景的重力
        physics_scene_path = "/World/physicsScene"
        physics_scene = UsdPhysics.Scene.Get(stage, physics_scene_path)
        if not physics_scene:
            physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
        
        # 设置重力方向和大小（Z轴向下，9.81 m/s²）
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        print(f"[Physics] Set gravity: direction=(0, 0, -1), magnitude=9.81 m/s²")
        
        # Load room environment
        print("Loading room environment...")
        if load_room_environment(stage, pipeline_dir):
            print("Room environment loaded successfully")
        else:
            print("Failed to load room environment, adding default ground plane")
            my_world.scene.add_default_ground_plane()
        
        # 添加物理地板
        print("Adding physics ground plane at Z=0...")
        my_world.scene.add_default_ground_plane(z_position=0.0)
        print("Physics ground plane added")
        
        print(f"World stage units: {get_stage_units()} meters per unit")
        
        # Add objects in calculation order (按计算顺序添加物体)
        calculation_order = pose_data["metadata"]["calculation_order"]
        objects_data = pose_data["objects"]
        main_object = pose_data["main_object"]
        
        successful_assets = 0
        is_main_object = False
        added_object_prims = []  # 保存所有非桌子物体的路径（用于切换物理状态）
        
        for object_name in calculation_order:
            if object_name not in objects_data:
                print(f"Warning: {object_name} not found in objects data")
                continue

            if object_name == main_object:
                is_main_object = True
                print(f"\n*** Processing main object: {object_name} ***")
            
            object_data = objects_data[object_name]
            
            # Get rotation angle, skip non-object data like view_angle
            rotation_angle = rotation_data.get(object_name, 0)
            if object_name == "view_angle":
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing object: {object_name}")
            print(f"Size: {object_data['size']} cm")
            print(f"Pose: {object_data['pose']} cm")
            print(f"Rotation angle: {rotation_angle}°")


            
            success, prim_path = add_object_to_scene(sized_mesh_path, usd_cache_dir, stage, object_name, object_data, rotation_angle, is_main_object)
            if success:
                successful_assets += 1
                # 保存非桌子物体的路径（用于切换物理状态）
                if not is_main_object and prim_path:
                    added_object_prims.append(prim_path)
            
            print(f"{'='*60}")

            is_main_object = False  # Reset flag
        
        print(f"\nSuccessfully added {successful_assets}/{len(calculation_order)} assets")
        print("All assets positioned on room floor (Z offset: -0.7696m)")
        
        # Add lights (添加灯光)
        # Dome light
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(1000)
        
        # Directional light
        distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant_light.CreateIntensityAttr(500)
        distant_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

        # --- Fix: Set Camera View ---
        print("\nSetting camera view...")
        set_camera_view(eye=np.array([1.5, 0.0, 0.8]), target=np.array([0.0, 0.0, -0.2]))
        
        # CRITICAL: 重置物理世界，初始化所有刚体和碰撞体
        print("\nInitializing physics...")
        my_world.reset()
        print("[Physics] Physics world reset complete - all rigid bodies initialized")
        
        print("\nStarting simulation...")
        print("Use the viewport camera to navigate and check object placement")
        print("Layout based on VLM analysis with calculated stacking relationships")
        
        # 启动输入监听线程
        cmd_queue = queue.Queue()
        stop_event = threading.Event()
        input_thread = threading.Thread(target=input_listener_thread, args=(cmd_queue, stop_event), daemon=True)
        input_thread.start()
        
        # 保存所有物体的初始状态（用于 reset）
        # 动态构造桌子路径（支持 table_0, table_1, table_2 等）
        main_object_name = main_object  # 例如 "table_0" 或 "table_1"
        import re
        clean_name = re.sub(r'[^\w\s-]', '', main_object_name)
        clean_name = re.sub(r'^#+', '', clean_name).strip()
        normalized_table_name = normalize_asset_name(clean_name)
        table_prim_path = f"/World/{normalized_table_name}"
        
        print(f"[System] 动态识别桌子：{main_object_name} → {table_prim_path}")
        
        initial_rotation = [-90, 0, 0]  # 初始旋转：Y-up转Z-up
        current_rotation = list(initial_rotation)
        
        # 保存所有物体的初始位置和旋转
        initial_states = {}
        initial_states[table_prim_path] = {
            'position': XFormPrim(table_prim_path).get_world_pose()[0],
            'orientation': XFormPrim(table_prim_path).get_world_pose()[1]
        }
        for prim_path in added_object_prims:
            obj = XFormPrim(prim_path)
            pos, ori = obj.get_world_pose()
            initial_states[prim_path] = {'position': pos, 'orientation': ori}
        
        print(f"[System] 已保存 {len(initial_states)} 个物体的初始状态")
        
        # 旋转动画状态
        rotation_animation = {
            'active': False,
            'axis': 'x',  # x, y, z
            'target_angle': 0,
            'angular_velocity': 0,  # 度/秒
            'current_user_angle': 0,  # 用户坐标系中的当前角度（0° = 水平桌面）
            'table_was_dynamic': False  # 记录桌子旋转前是否是动力学状态
        }
        
        # Run simulation with interactive control
        running = True
        step_count = 0
        max_steps = 55 * 5000  # 总步数
        dt = 0.01  # 时间步长（秒）
        
        while running and step_count < max_steps:
            # 处理用户命令
            try:
                while not cmd_queue.empty():
                    cmd = cmd_queue.get_nowait()
                    parts = cmd.lower().split()
                    
                    if cmd.lower() == 'quit':
                        print("退出程序...")
                        running = False
                        break
                    elif cmd.lower() == 'reset':
                        # 完整 reset：恢复所有物体到初始状态
                        print("[Reset] 正在恢复所有物体到初始状态...")
                        
                        # 先禁用物理（避免恢复时受力）
                        toggle_physics_for_objects(stage, added_object_prims, enable_dynamic=False, 
                                                   include_table=True, table_prim_path=table_prim_path)
                        
                        # 恢复所有物体的位置和旋转
                        for prim_path, state in initial_states.items():
                            obj = XFormPrim(prim_path)
                            obj.set_world_pose(position=state['position'], orientation=state['orientation'])
                        
                        current_rotation = list(initial_rotation)
                        rotation_animation['active'] = False
                        rotation_animation['current_user_angle'] = 0  # 重置用户角度为 0（水平）
                        
                        print(f"[Reset] 已恢复 {len(initial_states)} 个物体到初始状态")
                        print("[Reset] 物理已禁用，输入 'physics on' 重新启用")
                        
                    elif cmd.lower() == 'physics on':
                        toggle_physics_for_objects(stage, added_object_prims, enable_dynamic=True, 
                                                   include_table=True, table_prim_path=table_prim_path)
                    elif cmd.lower() == 'physics off':
                        toggle_physics_for_objects(stage, added_object_prims, enable_dynamic=False, 
                                                   include_table=True, table_prim_path=table_prim_path)
                    elif cmd.lower() == 'bbox on':
                        toggle_collision_visibility(stage, show_collision=True)
                    elif cmd.lower() == 'bbox off':
                        toggle_collision_visibility(stage, show_collision=False)
                    elif len(parts) == 2 and parts[0] == 'physics':
                        if parts[1] == 'on':
                            toggle_physics_for_objects(stage, added_object_prims, enable_dynamic=True, 
                                                       include_table=True, table_prim_path=table_prim_path)
                        elif parts[1] == 'off':
                            toggle_physics_for_objects(stage, added_object_prims, enable_dynamic=False, 
                                                       include_table=True, table_prim_path=table_prim_path)
                        else:
                            print(f"未知物理命令: physics {parts[1]}，请使用 'physics on' 或 'physics off'")
                    elif len(parts) == 2 and parts[0] == 'bbox':
                        if parts[1] == 'on':
                            toggle_collision_visibility(stage, show_collision=True)
                        elif parts[1] == 'off':
                            toggle_collision_visibility(stage, show_collision=False)
                        else:
                            print(f"未知包围盒命令: bbox {parts[1]}，请使用 'bbox on' 或 'bbox off'")
                    elif len(parts) >= 2:
                        # 处理旋转命令: x 角度 [角速度]
                        axis = parts[0]
                        if axis in ['x', 'y', 'z']:
                            try:
                                angle = float(parts[1])
                                angular_vel = float(parts[2]) if len(parts) >= 3 else 5.0  # 默认角速度 5 度/秒
                                
                                # 检查桌子是否是动力学状态
                                table_prim = stage.GetPrimAtPath(table_prim_path)
                                rigid_body = UsdPhysics.RigidBodyAPI(table_prim)
                                is_kinematic = rigid_body.GetKinematicEnabledAttr().Get() if rigid_body else True
                                
                                # 如果桌子是动力学的，先切换到 Kinematic 以便控制旋转
                                if not is_kinematic:
                                    rigid_body.GetKinematicEnabledAttr().Set(True)
                                    rotation_animation['table_was_dynamic'] = True
                                    print(f"[Rotation] 旋转期间桌子暂时切换为 Kinematic（可控制）")
                                else:
                                    rotation_animation['table_was_dynamic'] = False
                                
                                # CRITICAL FIX: 读取桌子的实际当前旋转
                                # 同步 current_user_angle，避免突然跳变
                                table_obj = XFormPrim(table_prim_path)
                                _, current_quat = table_obj.get_world_pose()
                                
                                # 将四元数转换为欧拉角
                                euler_rad = quat_to_euler_angles(current_quat)
                                euler_deg = [math.degrees(euler_rad[0]), 
                                            math.degrees(euler_rad[1]), 
                                            math.degrees(euler_rad[2])]
                                
                                # 更新 current_rotation 为实际值
                                current_rotation[0] = euler_deg[0]
                                current_rotation[1] = euler_deg[1]
                                current_rotation[2] = euler_deg[2]
                                
                                # 同步 current_user_angle（实际欧拉角 → 用户角度）
                                # 用户角度 = 实际欧拉角 - 基准(-90°)
                                if axis == 'x':
                                    rotation_animation['current_user_angle'] = current_rotation[0] + 90
                                elif axis == 'y':
                                    rotation_animation['current_user_angle'] = current_rotation[1]
                                elif axis == 'z':
                                    rotation_animation['current_user_angle'] = current_rotation[2]
                                
                                print(f"[Rotation Debug] 桌子实际角度: X={euler_deg[0]:.1f}°, 用户角度: {rotation_animation['current_user_angle']:.1f}°")
                                
                                # 设置旋转动画
                                rotation_animation['active'] = True
                                rotation_animation['axis'] = axis
                                rotation_animation['target_angle'] = angle
                                rotation_animation['angular_velocity'] = angular_vel
                                
                                print(f"[Rotation] 开始旋转：绕{axis.upper()}轴从 {rotation_animation['current_user_angle']:.1f}° 到 {angle}°，角速度 {angular_vel} °/s")
                            except ValueError:
                                print(f"无效参数，格式: {axis} 角度 [角速度]")
                        else:
                            print(f"未知轴: {axis}，请使用 x, y 或 z")
                    else:
                        print(f"无效命令: {cmd}")
            except queue.Empty:
                pass
            
            # 处理旋转动画
            if rotation_animation['active']:
                axis = rotation_animation['axis']
                target = rotation_animation['target_angle']
                current = rotation_animation['current_user_angle']
                vel = rotation_animation['angular_velocity']
                
                # 计算角度变化
                delta = target - current
                if abs(delta) < vel * dt:
                    # 到达目标
                    rotation_animation['current_user_angle'] = target
                    rotation_animation['active'] = False
                    
                    # 如果旋转前桌子是动力学的，旋转完成后恢复为动力学
                    if rotation_animation['table_was_dynamic']:
                        table_prim = stage.GetPrimAtPath(table_prim_path)
                        rigid_body = UsdPhysics.RigidBodyAPI(table_prim)
                        if rigid_body:
                            rigid_body.GetKinematicEnabledAttr().Set(False)
                            print(f"[Rotation] 旋转完成：{axis.upper()}轴 = {target}°，桌子恢复为 Dynamic（会受物理影响）")
                    else:
                        print(f"[Rotation] 旋转完成：{axis.upper()}轴 = {target}°")
                else:
                    # 继续旋转
                    step = vel * dt if delta > 0 else -vel * dt
                    rotation_animation['current_user_angle'] += step
                
                # 应用旋转（用户角度 → 实际欧拉角）
                # 用户角度是相对于水平桌面的，0° = 水平
                # 实际欧拉角需要加上 -90° 的基准（Y-up 转 Z-up）
                if axis == 'x':
                    current_rotation[0] = -90 + rotation_animation['current_user_angle']
                elif axis == 'y':
                    current_rotation[1] = rotation_animation['current_user_angle']
                elif axis == 'z':
                    current_rotation[2] = rotation_animation['current_user_angle']
                
                set_angles(table_prim_path, current_rotation)
            
            # 模拟一步
            my_world.step(render=True)
            time.sleep(dt)
            step_count += 1
        
        stop_event.set()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close simulation
        simulation_app.close()

if __name__ == "__main__":
    output_assets_dir = 'output_scene/scene_1/output_assets'
    isaac_main(output_assets_dir)
