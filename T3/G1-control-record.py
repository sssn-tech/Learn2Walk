import argparse
import torch
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate the Unitree G1 robot using H1 policy.")
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
# 添加模型路径参数
parser.add_argument(
    "--model_path",
    type=str,
    default="H1-walking.pt",
    help="Path to the H1 model checkpoint file.",
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
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

import omni.replicator.core as rep
from isaaclab.sensors.camera import Camera, CameraCfg
import isaacsim.core.utils.prims as prim_utils
from isaaclab.utils import convert_dict_to_backend
import numpy as np

##
# Pre-defined configs
##
from unitree import G1_MINIMAL_CFG, map_g1_obs_to_h1_obs, map_h1_action_to_g1

# 加载H1模型
def load_h1_model(model_path):
    """加载H1预训练模型"""
    print(f"[INFO]: Loading H1 model from {model_path}")
    try:
        # 加载checkpoint
        ckpt = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        print(f"[DEBUG]: Checkpoint keys: {list(ckpt.keys())}")
        
        # 创建H1的Actor网络
        from torch import nn
        
        # 创建Actor网络 - 使用H1的结构（输入69维，输出19维）
        actor = nn.Sequential(
            nn.Linear(69, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 19)
        )
        
        # 创建一个简单的包装器类来模拟原始模型的接口
        class ModelWrapper:
            def __init__(self, actor):
                self.actor = actor
                self.eval_mode = True
            
            def eval(self):
                self.eval_mode = True
                self.actor.eval()
                return self
            
            def act_inference(self, obs):
                # 直接返回动作，以及虚拟的额外返回值
                with torch.no_grad():
                    action = self.actor(obs)
                return action, None, None, None
        
        # 尝试加载权重
        if 'model_state_dict' in ckpt:
            # 提取actor相关的权重
            actor_state_dict = {}
            for key, value in ckpt['model_state_dict'].items():
                # 只选择actor部分的权重
                if key.startswith('actor.'):
                    # 移除'actor.'前缀
                    new_key = key[len('actor.'):]
                    actor_state_dict[new_key] = value
            
            # 如果找到了actor权重，加载它们
            if actor_state_dict:
                actor.load_state_dict(actor_state_dict)
                print("[INFO]: Actor weights loaded successfully.")
            else:
                print("[WARNING]: No actor weights found in checkpoint. Trying different key format.")
                # 尝试不同的键格式
                for key, value in ckpt['model_state_dict'].items():
                    if 'actor' in key or 'policy' in key:
                        parts = key.split('.')
                        if len(parts) > 1:
                            new_key = '.'.join(parts[1:])  # 去掉第一段
                            actor_state_dict[new_key] = value
                
                if actor_state_dict:
                    actor.load_state_dict(actor_state_dict)
                    print("[INFO]: Actor weights loaded with alternative format.")
                else:
                    print("[WARNING]: Could not find actor weights. Using random initialization.")
        
        model = ModelWrapper(actor)
        model.eval()
        
        print("[INFO]: H1 model loaded successfully.")
        return model
    except Exception as e:
        print(f"[ERROR]: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None
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


def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor, Camera]:
    """Designs the scene."""
    # Ground-plane - 增加摩擦系数
    # Ground-plane
    # cfg = sim_utils.GroundPlaneCfg()
    # cfg.func("/World/defaultGroundPlane", cfg)
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # 为地面添加物理材质以增加摩擦力
    ground_prim_path = "/World/defaultGroundPlane"
    physics_material_path = ground_prim_path + "/physicsProperties"
    prim_utils.create_prim(physics_material_path, "PhysicsMaterialAPI")
    
    # 设置物理材质属性
    from pxr import UsdPhysics
    physics_material = UsdPhysics.MaterialAPI.Apply(prim_utils.get_prim_at_path(physics_material_path))
    physics_material.CreateStaticFrictionAttr().Set(1.0)    # 增大静摩擦系数
    physics_material.CreateDynamicFrictionAttr().Set(0.8)   # 增大动摩擦系数
    physics_material.CreateRestitutionAttr().Set(0.1)       # 设置恢复系数
    
    # Lights
    # =====================
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origin for G1
    origins = torch.tensor([
        [0.0, 0.0, 0.0],
    ]).to(device=sim.device)

    # G1 robot - 使用minimal配置加速仿真
    g1 = Articulation(G1_MINIMAL_CFG.replace(prim_path="/World/G1"))
    robots = [g1]
    
    # 如果需要录制视频，设置录制相机
    camera = None
    if args_cli.record_video:
        camera = setup_recording_camera()

    return robots, origins, camera

def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor, camera: Camera = None):
    """Runs the simulation loop with H1 policy control."""
    # 加载H1模型
    h1_model = load_h1_model(args_cli.model_path)
    if h1_model is None:
        print("[ERROR]: Failed to load H1 model. Exiting...")
        return

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 初始化机器人
    for index, robot in enumerate(robots):
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origins[index]

        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.write_root_pose_to_sim(root_state[:, 0:7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.reset()
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()

    # 目标速度设置 - 前进指令
    command_x = 0.05  # 前进速度
    command_y = 0.0  # 侧向速度
    command_yaw = 0.0  # 转向速度
    velocity_commands = torch.tensor([[command_x, command_y, command_yaw]], device=sim.device)

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
        camera_position = torch.tensor([[3.0, 0.0, 2.25]], device=sim.device)
        camera_target = torch.tensor([[0.0, 0.0, 0.3]], device=sim.device)
        camera.set_world_poses_from_view(camera_position, camera_target)

        print(f"[INFO]: Video recording enabled. Frames will be saved to {output_dir}")

    # 初始化上一步动作为零
    last_actions = torch.zeros((1, 37), device=sim.device)

    while simulation_app.is_running():
        # 1. 获取G1的观测
        robot = robots[0]  # 只有一个机器人

        # 构建G1机器人的policy_obs
        base_lin_vel = robot.data.root_lin_vel_b.clone()
        base_ang_vel = robot.data.root_ang_vel_b.clone()
        projected_gravity = robot.data.projected_gravity_b.clone()
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = robot.data.joint_vel.clone()

        # 构建完整的G1 policy_obs (一共123维)
        g1_policy_obs = torch.cat([
            base_lin_vel,                  # [0:3]
            base_ang_vel,                  # [3:6]
            projected_gravity,             # [6:9]
            velocity_commands,             # [9:12]
            joint_pos,                     # [12:49]
            joint_vel,                     # [49:86]
            last_actions,                  # [86:123]
        ], dim=1)

        # 2. 将G1观测转换为H1观测
        h1_obs_np = map_g1_obs_to_h1_obs(g1_policy_obs[0].cpu().numpy())
        h1_obs = torch.tensor(h1_obs_np, dtype=torch.float32).unsqueeze(0)

        # 3. 推理H1模型获取动作
        with torch.no_grad():
            h1_actions, _, _, _ = h1_model.act_inference(h1_obs)
            h1_actions = h1_actions.cpu().numpy()[0]  # 转为numpy并取第一个样本
        h1_actions = h1_actions
        # 4. 将H1动作转换为G1动作
        g1_actions_np = map_h1_action_to_g1(h1_actions)
        g1_actions = torch.tensor(g1_actions_np, dtype=torch.float32, device=sim.device).unsqueeze(0)

        # 5. 应用动作到G1机器人
        robot.set_joint_position_target(g1_actions)
        robot.write_data_to_sim()

        # 保存当前动作用于下一步观测
        last_actions = g1_actions.clone()

        # 模拟物理
        sim.step()
        sim_time += sim_dt
        count += 1

        # 相机更新和视频录制
        if args_cli.record_video and camera is not None and rep_writer is not None:
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
                    print(f"[INFO]: Recorded frame {count}, simulation time: {sim_time:.2f}s")

        # 更新机器人状态
        for robot in robots:
            robot.update(sim_dt)

        # 每500步更改速度命令，使机器人执行不同动作
        if count % 500 == 0:
            if command_x > 0:
                # 改为左转
                command_x = 0.5
                command_y = 0.0
                command_yaw = 0.5
                print(f"[INFO]: Changing command to left turn at time {sim_time:.2f}s")
            elif command_yaw > 0:
                # 改为右转
                command_x = 0.5
                command_y = 0.0
                command_yaw = -0.5
                print(f"[INFO]: Changing command to right turn at time {sim_time:.2f}s")
            else:
                # 恢复直行
                command_x = 1.0
                command_y = 0.0
                command_yaw = 0.0
                print(f"[INFO]: Changing command to forward at time {sim_time:.2f}s")
            
            velocity_commands = torch.tensor([[command_x, command_y, command_yaw]], device=sim.device)

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 0.3])

    robots, origins, camera = design_scene(sim)

    sim.reset()

    print("[INFO]: Setup complete...")

    run_simulator(sim, robots, origins, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()