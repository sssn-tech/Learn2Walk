import argparse
import torch
import random
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate the Unitree G1 robot.")
# 添加视频录制相关参数
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Enable video recording of the simulation.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Directory to save the recorded video frames.",
)
# 添加视频录制时间限制参数
parser.add_argument(
    "--record_duration",
    type=float,
    default=30.0,
    help="Duration of video recording in seconds. After this time, the simulation will close.",
)
# 添加关节数据记录和曲线图参数
parser.add_argument(
    "--plot_joints",
    action="store_true",
    default=False,
    help="Enable plotting of joint trajectories at the end of simulation.",
)
parser.add_argument(
    "--joint_data_file",
    type=str,
    default="joint_data.npz",
    help="File to save joint trajectory data.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.sim import SimulationContext
import omni.replicator.core as rep
from isaaclab.sensors.camera import Camera, CameraCfg
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils import convert_dict_to_backend
import numpy as np

##
# Pre-defined configs
##
from isaaclab_assets import G1_CFG  # Only G1 is needed


def setup_recording_camera() -> Camera:
    """设置用于录制视频的相机"""
    # 创建相机位置的变换节点
    prim_utils.create_prim("/World/RecordingCamera", "Xform")
    
    # 配置相机
    camera_cfg = CameraCfg(
        prim_path="/World/RecordingCamera/CameraSensor",
        update_period=0,  # 每一帧都更新
        height=720,       # 视频高度
        width=1280,       # 视频宽度
        data_types=["rgb"],  # 只需要RGB图像用于视频录制
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
    )
    
    # 创建相机
    camera = Camera(cfg=camera_cfg)
    return camera


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor, dict, Camera]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origin for G1
    origins = torch.tensor([
        [0.0, 0.0, 0.0],
    ]).to(device=sim.device)

    # G1 robot
    g1 = Articulation(G1_CFG.replace(prim_path="/World/G1"))
    robots = [g1]
    
    # 创建物体放置位置
    object_positions = [
        [1.0, 1.0, 0.0],   # 原来是 [2.0, 2.0, 0.0]
        [-1.0, 1.0, 0.0],  # 原来是 [-2.0, 2.0, 0.0]
        [1.0, -1.0, 0.0],  # 原来是 [2.0, -2.0, 0.0]
        [-1.0, -1.0, 0.0]  # 原来是 [-2.0, -2.0, 0.0]
    ]
    
    for i, position in enumerate(object_positions):
        prim_utils.create_prim(f"/World/ObjectOrigin{i}", "Xform", translation=position)
    
    # 创建四种基本几何体
    # 1. 圆锥
    cone_cfg = RigidObjectCfg(
        prim_path="/World/ObjectOrigin0/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.3,
            height=0.6,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.3), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone_object = RigidObject(cfg=cone_cfg)
    
    # 2. 立方体
    cube_cfg = RigidObjectCfg(
        prim_path="/World/ObjectOrigin1/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 1.0), metallic=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cube_object = RigidObject(cfg=cube_cfg)
    
    # 3. 球体
    sphere_cfg = RigidObjectCfg(
        prim_path="/World/ObjectOrigin2/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 1.0, 0.3), metallic=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    sphere_object = RigidObject(cfg=sphere_cfg)
    
    # 4. 圆柱体
    cylinder_cfg = RigidObjectCfg(
        prim_path="/World/ObjectOrigin3/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.2,
            height=0.8,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.3), metallic=0.3),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cylinder_object = RigidObject(cfg=cylinder_cfg)
    
    # 将所有几何体对象存入字典
    scene_entities = {
        "cone": cone_object,
        "cube": cube_object,
        "sphere": sphere_object,
        "cylinder": cylinder_object
    }
    
    # 如果需要录制视频，设置录制相机
    camera = None
    if args_cli.record_video:
        camera = setup_recording_camera()

    return robots, origins, scene_entities, camera

def select_main_joints(robot):
    """选择要跟踪的16个主要关节"""
    joint_names = robot.data.joint_names
    main_joints = []
    
    # 选择一组代表性关节
    important_patterns = [
        "left_hip", "right_hip", 
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "torso", "neck", "head"
    ]
    
    # 构建16个主要关节列表
    count = 0
    for pattern in important_patterns:
        for i, name in enumerate(joint_names):
            if pattern in name and count < 16:
                main_joints.append((i, name))
                count += 1
                if count >= 16:
                    break
                
    # 如果不足16个，填充其他关节
    if count < 16:
        for i, name in enumerate(joint_names):
            if i not in [idx for idx, _ in main_joints] and count < 16:
                main_joints.append((i, name))
                count += 1
    
    return main_joints[:16]  # 确保只返回16个关节

def plot_joint_trajectories(joint_data, joint_names, output_file="joint_trajectories.png"):
    """绘制关节轨迹图"""
    fig, axs = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle("G1 Robot Joint Trajectories", fontsize=16)
    
    # 获取时间序列
    time_steps = np.arange(len(joint_data[0])) * 0.005  # 使用仿真时间步长
    
    # 在4x4网格中绘制16个关节的轨迹
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            if idx < len(joint_data):
                axs[i, j].plot(time_steps, joint_data[idx], 'b-')
                axs[i, j].set_title(joint_names[idx], fontsize=10)
                axs[i, j].set_xlabel('Time (s)', fontsize=8)
                axs[i, j].set_ylabel('Joint Position (rad)', fontsize=8)
                axs[i, j].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
    plt.savefig(output_file, dpi=300)
    print(f"[INFO]: Joint trajectories plot saved to {output_file}")
    return fig

def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor, 
                  entities: dict[str, RigidObject], camera: Camera = None):

    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    start_time = time.time()
    
    # 选择要跟踪的16个关节
    main_joints = select_main_joints(robots[0])
    main_joint_indices = [idx for idx, _ in main_joints]
    main_joint_names = [name for _, name in main_joints]
    
    # 初始化关节数据存储
    joint_data = [[] for _ in range(len(main_joints))]

    # 初始化机器人
    for index, robot in enumerate(robots):
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origins[index]

        # 设置右腿踢腿动作（显著角度）
        joint_names = robot.data.joint_names
        for i, name in enumerate(joint_names):
            if "right_hip_pitch_joint" in name:
                joint_pos[:, i] = -1.2  # 向前抬腿
            elif "right_knee_joint" in name:
                joint_pos[:, i] = 2.0   # 弯曲膝盖
            elif "right_ankle_pitch_joint" in name:
                joint_pos[:, i] = -1.0  # 抬脚尖

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_root_pose_to_sim(root_state[:, 0:7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.reset()
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
    
    # 初始化场景物体
    # 圆锥位置初始化
    cone_object = entities["cone"]
    root_state = cone_object.data.default_root_state.clone()
    root_state[:, :3] += torch.tensor([[1.0, 1.0, 0.3]], device=sim.device)  # 原来是 [2.0, 2.0, 0.3]
    cone_object.write_root_pose_to_sim(root_state[:, :7])
    cone_object.write_root_velocity_to_sim(root_state[:, 7:])
    cone_object.reset()

    # 立方体位置初始化
    cube_object = entities["cube"]
    root_state = cube_object.data.default_root_state.clone()
    root_state[:, :3] += torch.tensor([[-1.0, 1.0, 0.2]], device=sim.device)  # 原来是 [-2.0, 2.0, 0.2]
    cube_object.write_root_pose_to_sim(root_state[:, :7])
    cube_object.write_root_velocity_to_sim(root_state[:, 7:])
    cube_object.reset()

    # 球体位置初始化
    sphere_object = entities["sphere"]
    root_state = sphere_object.data.default_root_state.clone()
    root_state[:, :3] += torch.tensor([[1.0, -1.0, 0.3]], device=sim.device)  # 原来是 [2.0, -2.0, 0.3]
    sphere_object.write_root_pose_to_sim(root_state[:, :7])
    sphere_object.write_root_velocity_to_sim(root_state[:, 7:])
    sphere_object.reset()

    # 圆柱体位置初始化
    cylinder_object = entities["cylinder"]
    root_state = cylinder_object.data.default_root_state.clone()
    root_state[:, :3] += torch.tensor([[-1.0, -1.0, 0.4]], device=sim.device)  # 原来是 [-2.0, -2.0, 0.4]
    cylinder_object.write_root_pose_to_sim(root_state[:, :7])
    cylinder_object.write_root_velocity_to_sim(root_state[:, 7:])
    cylinder_object.reset()

    # 如果需要录制视频，设置录制器和相机位置
    rep_writer = None
    if args_cli.record_video and camera is not None:
        # 确保输出目录存在
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args_cli.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 创建Replicator写入器
        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=4,
        )

        # 设置相机位置
        camera_position = torch.tensor([[4.0, 0.0, 3.0]], device=sim.device)
        camera_target = torch.tensor([[0.0, 0.0, 0.3]], device=sim.device)
        camera.set_world_poses_from_view(camera_position, camera_target)

        print(f"[INFO]: Video recording enabled. Frames will be saved to {output_dir}")
        print(f"[INFO]: Recording will stop after {args_cli.record_duration} seconds")

    while simulation_app.is_running():
        # 检查录制时间是否已达到限制
        elapsed_time = time.time() - start_time
        if args_cli.record_video and elapsed_time >= args_cli.record_duration:
            print(f"[INFO]: Recording time limit reached ({args_cli.record_duration}s). Stopping simulation.")
            break
            
        # 为每个机器人生成随机关节目标
        # 显式定义上下肢关节的随机范围（单位：弧度）
        upper_body_joint_ranges = {
            "shoulder": (-3, 3),
            "elbow": (-3, 3),
            "wrist": (-1.0, 1.0),
            "hand": (-1.0, 1.0),
            "finger": (-1.0, 1.0),
            "zero": (-1.0, 1.0),
            "one": (-1.0, 1.0),
            "two": (-1.0, 1.0),
            "three": (-1.0, 1.0),
            "four": (-1.0, 1.0),
            "five": (-1.0, 1.0),
            "six": (-1.0, 1.0),
        }

        lower_body_joint_ranges = {
            "hip": (-0.5, 0.5),
            "knee": (-0.6, 0.6),
            "ankle": (-0.4, 0.4),
            "torso": (-0.3, 0.3),
        }

        # 为每个机器人生成随机关节目标
        for robot in robots:
            joint_names = robot.data.joint_names
            joint_pos = robot.data.joint_pos.clone()

            for i, name in enumerate(joint_names):
                set = False
                # 跳过右腿关节（pitch方向）保持踢腿动作
                if any(key in name for key in ["right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"]):
                    continue  # 不修改踢腿姿态
                # 匹配上肢关节
                for key, (low, high) in upper_body_joint_ranges.items():
                    if key in name:
                        joint_pos[:, i] = torch.rand(1, device=sim.device) * (high - low) + low
                        set = True
                        break
                # 匹配下肢关节
                if not set:
                    for key, (low, high) in lower_body_joint_ranges.items():
                        if key in name:
                            joint_pos[:, i] = torch.rand(1, device=sim.device) * (high - low) + low
                            set = True
                            break
                # 如果没有匹配到任何设置，则使用默认小幅扰动
                if not set:
                    joint_pos[:, i] = torch.rand(1, device=sim.device) * 0.2 - 0.1  # [-0.1, 0.1]

            robot.set_joint_position_target(joint_pos)
            robot.write_data_to_sim()
            
        # 每隔一段时间随机移动一个物体
        if count % 200 == 0 and count > 0:
            try:
                # 随机选择一个物体
                entity_keys = list(entities.keys())
                random_entity_key = random.choice(entity_keys)
                random_entity = entities[random_entity_key]
                
                # 获取当前位置并应用一个小的随机速度变化
                current_pos = random_entity.data.root_pos_w.clone()
                
                # 应用一个较小的随机速度
                random_vel = torch.rand(6, device=sim.device) * 1.0 - 0.5  # 更保守的速度范围
                random_vel[2] = abs(random_vel[2]) * 0.5  # 确保z方向的速度是较小的向上速度
                
                # 写入新的速度
                random_entity.write_root_velocity_to_sim(random_vel.unsqueeze(0))
                
                print(f"[INFO]: Applied random velocity to {random_entity_key}")
            except Exception as e:
                print(f"[WARNING]: Could not apply velocity: {e}")

        # 模拟物理
        sim.step()
        sim_time += sim_dt
        count += 1

        # 相机更新和视频录制
        if args_cli.record_video and camera is not None and rep_writer is not None:
            try:
                # 更新相机数据
                camera.update(dt=sim_dt)
                
                # 检查相机是否有输出数据
                if "rgb" in camera.data.output and camera.data.output["rgb"].shape[0] > 0:
                    # 转换为numpy格式
                    single_cam_data = convert_dict_to_backend(
                        {"rgb": camera.data.output["rgb"][0]}, backend="numpy"
                    )
                    
                    # 获取相机信息
                    single_cam_info = camera.data.info[0]
                    
                    # 打包成replicator格式
                    rep_output = {"annotators": {}}
                    for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                        if info is not None:
                            rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                        else:
                            rep_output["annotators"][key] = {"render_product": {"data": data}}
                    
                    # 添加触发时间信息
                    rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
                    
                    # 写入帧
                    rep_writer.write(rep_output)
                    
                    if count % 100 == 0:
                        print(f"[INFO]: Recorded frame {count}, time: {elapsed_time:.2f}s / {args_cli.record_duration}s")
            except Exception as e:
                print(f"[WARNING]: Error during camera update: {e}")

        # 更新机器人状态
        for robot in robots:
            robot.update(sim_dt)
            
            # 记录选定关节的位置数据
            if args_cli.plot_joints:
                for i, joint_idx in enumerate(main_joint_indices):
                    # 获取关节位置，将tensor转换为标量
                    joint_pos_value = robot.data.joint_pos[0, joint_idx].item()
                    joint_data[i].append(joint_pos_value)
            
        # 更新所有几何体状态
        for entity in entities.values():
            entity.update(sim_dt)

    # 保存和绘制关节数据
    if args_cli.plot_joints and joint_data:
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), args_cli.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存关节数据
        np.savez(os.path.join(output_dir, args_cli.joint_data_file), 
                 joint_data=joint_data, 
                 joint_names=main_joint_names)
        
        # 绘制关节轨迹图
        plot_joint_trajectories(joint_data, main_joint_names, 
                               os.path.join(output_dir, "joint_trajectories.png"))
        
        print(f"[INFO]: Joint data saved to {os.path.join(output_dir, args_cli.joint_data_file)}")

    return joint_data, main_joint_names

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 设置视角查看场景
    sim.set_camera_view(eye=[4.0, 0.0, 3.0], target=[0.0, 0.0, 0.3])

    robots, origins, scene_entities, camera = design_scene(sim)

    sim.reset()

    print("[INFO]: Setup complete...")

    joint_data, joint_names = run_simulator(sim, robots, origins, scene_entities, camera)

    # 如果设置了绘制关节数据，显示图表（不依赖于仿真）
    if args_cli.plot_joints and len(joint_data) > 0:
        plt.figure()
        plt.show()

if __name__ == "__main__":
    main()
    simulation_app.close()