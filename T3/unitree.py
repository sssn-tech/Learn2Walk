import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    # 修改初始状态以匹配H1的配置
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),  # 修改：从0.74调整为1.05，与H1保持一致
        joint_pos={
            # 下肢关节 - 映射自H1
            ".*_hip_yaw_joint": 0.0,  # 对应H1的 *_hip_yaw
            ".*_hip_roll_joint": 0.0,  # 对应H1的 *_hip_roll
            ".*_hip_pitch_joint": -0.28,  # 对应H1的 *_hip_pitch (-16度)
            ".*_knee_joint": 0.79,  # 对应H1的 *_knee (45度)
            ".*_ankle_pitch_joint": -0.52,  # 对应H1的 *_ankle (-30度)
            ".*_ankle_roll_joint": 0.0,  # G1特有，H1没有，设为0
            
            # 躯干关节
            "torso_joint": 0.0,  # 对应H1的 torso
            
            # 上肢关节 - 映射自H1
            ".*_shoulder_pitch_joint": 0.28,  # 对应H1的 *_shoulder_pitch
            ".*_shoulder_roll_joint": 0.0,  # 对应H1的 *_shoulder_roll
            ".*_shoulder_yaw_joint": 0.0,  # 对应H1的 *_shoulder_yaw
            ".*_elbow_pitch_joint": 0.52,  # 对应H1的 *_elbow
            ".*_elbow_roll_joint": 0.0,  # G1特有，H1没有，设为0
            
            # 手部关节 - G1特有，保持原始值或设为中立位置
            ".*_five_joint": 0.0,  # G1特有，设为中立位置
            ".*_three_joint": 0.0,  # G1特有，设为中立位置
            ".*_six_joint": 0.0,  # G1特有，设为中立位置
            ".*_four_joint": 0.0,  # G1特有，设为中立位置
            ".*_zero_joint": 0.0,  # G1特有，设为中立位置
            "left_one_joint": 1.0,  # 保持原配置中的值
            "right_one_joint": -1.0,  # 保持原配置中的值
            "left_two_joint": 0.52,  # 保持原配置中的值
            "right_two_joint": -0.52,  # 保持原配置中的值
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot with initial state mapped from H1."""

# 保持minimal配置的更新
G1_MINIMAL_CFG = G1_CFG.copy()
G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes and initial state mapped from H1."""

# =============================
# H1到G1关节映射表
# =============================
# 说明：
# 1. 该映射表用于将H1机器人的关节动作（action）或关节角度（position）映射到G1机器人。
# 2. G1比H1多出的自由度（如踝关节roll、肘关节roll等）在迁移时统一设为0。
# 3. 使用时可按此顺序重排H1策略输出，并为G1多余关节补零。

H1_TO_G1_JOINT_MAP = {
    # 下肢（左腿）
    "left_hip_yaw": "left_hip_yaw_joint",
    "left_hip_roll": "left_hip_roll_joint",
    "left_hip_pitch": "left_hip_pitch_joint",
    "left_knee": "left_knee_joint",
    "left_ankle": "left_ankle_pitch_joint",
    # G1多出的踝关节roll
    "left_ankle_roll_joint": 0.0,

    # 下肢（右腿）
    "right_hip_yaw": "right_hip_yaw_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_knee": "right_knee_joint",
    "right_ankle": "right_ankle_pitch_joint",
    # G1多出的踝关节roll
    "right_ankle_roll_joint": 0.0,

    # 躯干
    "torso": "torso_joint",

    # 上肢（左臂）
    "left_shoulder_pitch": "left_shoulder_pitch_joint",
    "left_shoulder_roll": "left_shoulder_roll_joint",
    "left_shoulder_yaw": "left_shoulder_yaw_joint",
    "left_elbow": "left_elbow_pitch_joint",
    # G1多出的肘关节roll
    "left_elbow_roll_joint": 0.0,

    # 上肢（右臂）
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_pitch_joint",
    # G1多出的肘关节roll
    "right_elbow_roll_joint": 0.0,
}

# 使用说明：
# 1. 遍历G1的关节顺序，若在映射表中为字符串，则取H1对应关节的值；若为0.0，则直接补零。
# 2. 适用于action、position等向量的迁移和重排。

h1_joint_names = ["left_hip_yaw", "right_hip_yaw", "torso", "left_hip_roll", "right_hip_roll", "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_pitch", "right_hip_pitch", "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee", "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle", "right_ankle", "left_elbow", "right_elbow"]
g1_joint_names = ["left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint", "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint", "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint", "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint", "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint", "right_one_joint", "left_two_joint", "right_two_joint"]


def map_h1_action_to_g1(h1_action, h1_joint_names=h1_joint_names, g1_joint_names=g1_joint_names):
    # H1关节名到action值的映射
    h1_action_dict = dict(zip(h1_joint_names, h1_action))
    g1_action = []
    for g1_joint in g1_joint_names:
        # 反查映射表，找到H1关节名
        h1_key = None
        for k, v in H1_TO_G1_JOINT_MAP.items():
            if v == g1_joint:
                h1_key = k
                break
        if h1_key is not None and h1_key in h1_action_dict:
            g1_action.append(h1_action_dict[h1_key])
        elif g1_joint in H1_TO_G1_JOINT_MAP and H1_TO_G1_JOINT_MAP[g1_joint] == 0.0:
            g1_action.append(0.0)
        else:
            # 其它G1关节（如手指等），也补零
            g1_action.append(0.0)
    return g1_action


# =============================
# G1 observation到H1 observation的映射函数
# =============================
# 说明：
# 1. 该函数用于将G1环境的policy observation向量，重排为H1策略可用的输入格式。
# 2. 只对joint_pos、joint_vel、actions三段做重排，其余部分（如base速度、命令、height_scan等）直接传递。
# 3. 输入为单个环境的policy observation（1维向量），输出为H1顺序的observation。
# 4. 需传入G1和H1的关节顺序列表。

def map_g1_obs_to_h1_obs(policy_obs_row, g1_joint_names=g1_joint_names, h1_joint_names=h1_joint_names):
    """
    将G1环境的policy observation向量，重排为H1策略可用的输入格式。
    
    Args:
        policy_obs_row: G1环境输出的单个环境的policy observation
        g1_joint_names: G1机器人的关节名称列表
        h1_joint_names: H1机器人的关节名称列表
        
    Returns:
        h1_obs: 重排后符合H1策略输入格式的observation向量
    """
    # 创建输出向量，初始为空
    h1_obs = []
    
    # 1. 前12维直接复制 [0:12] (base_lin_vel, base_ang_vel, projected_gravity, velocity_commands)
    h1_obs.extend(policy_obs_row[0:12])
    
    # 2. 重排joint_pos部分 [12:49] -> [12:31]
    g1_joint_pos = policy_obs_row[12:49]  # G1的37个关节位置
    g1_joint_pos_dict = dict(zip(g1_joint_names, g1_joint_pos))  # 构建G1关节名到位置的映射
    
    # 为H1创建关节位置部分
    h1_joint_pos = []
    for h1_joint in h1_joint_names:
        # 找到对应的G1关节名
        g1_joint = H1_TO_G1_JOINT_MAP.get(h1_joint)
        if g1_joint in g1_joint_pos_dict:
            h1_joint_pos.append(g1_joint_pos_dict[g1_joint])
        else:
            # 如果没找到对应的G1关节（理论上不应该发生），使用0作为默认值
            h1_joint_pos.append(0.0)
    
    h1_obs.extend(h1_joint_pos)
    
    # 3. 重排joint_vel部分 [49:86] -> [31:50]
    g1_joint_vel = policy_obs_row[49:86]  # G1的37个关节速度
    g1_joint_vel_dict = dict(zip(g1_joint_names, g1_joint_vel))  # 构建G1关节名到速度的映射
    
    # 为H1创建关节速度部分
    h1_joint_vel = []
    for h1_joint in h1_joint_names:
        # 找到对应的G1关节名
        g1_joint = H1_TO_G1_JOINT_MAP.get(h1_joint)
        if g1_joint in g1_joint_vel_dict:
            h1_joint_vel.append(g1_joint_vel_dict[g1_joint])
        else:
            # 如果没找到对应的G1关节，使用0作为默认值
            h1_joint_vel.append(0.0)
    
    h1_obs.extend(h1_joint_vel)
    
    # 4. 重排actions部分 [86:123] -> [50:69]
    g1_actions = policy_obs_row[86:123]  # G1的37个关节动作
    g1_actions_dict = dict(zip(g1_joint_names, g1_actions))  # 构建G1关节名到动作的映射
    
    # 为H1创建动作部分
    h1_actions = []
    for h1_joint in h1_joint_names:
        # 找到对应的G1关节名
        g1_joint = H1_TO_G1_JOINT_MAP.get(h1_joint)
        if g1_joint in g1_actions_dict:
            h1_actions.append(g1_actions_dict[g1_joint])
        else:
            # 如果没找到对应的G1关节，使用0作为默认值
            h1_actions.append(0.0)
    
    h1_obs.extend(h1_actions)
    
    # 确保输出向量维度为69
    assert len(h1_obs) == 69, f"输出向量维度应为69，当前为{len(h1_obs)}"
    
    return h1_obs

# 使用说明：
# 1. policy_obs_row为G1环境输出的单个环境的policy observation（如policy_obs[0]）。
# 2. g1_joint_names和h1_joint_names为你从环境中提取的有序关节名列表。
# 3. 返回的h1_obs可直接送入H1策略推理。
