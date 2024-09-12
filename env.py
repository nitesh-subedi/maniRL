# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# import gym

# from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
# from omni.isaac.core.utils.prims import get_prim_at_path

# compute_aabb


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import carb
import matplotlib.pyplot as plt
import cv2

log_dir = "./camera_img"
lower_green = np.array([40, 50, 20])
upper_green = np.array([80, 255, 255])

class maniEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=512,
        seed=0,
        config={"headless": True, "anti_aliasing": 0},
    ) -> None:
        
        from omni.isaac.kit import SimulationApp
        
        self._simulation_app = SimulationApp(launch_config=config)

        from omni.isaac.core.utils.extensions import enable_extension

        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        self._simulation_app.set_setting("/ngx/enabled", False)

        enable_extension("omni.services.streamclient.webrtc")
        
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        from omni.isaac.core import World
        from omni.isaac.franka import Franka
        from omni.isaac.core.objects import VisualCuboid
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.dynamic_control import _dynamic_control
        from omni.isaac.sensor import Camera
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        import omni
        from omni.physx.scripts import deformableUtils, physicsUtils
        from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema

        # Get stage
        self.stage = omni.usd.get_context().get_stage()
        
        # Set up world
        self._my_world = World(physics_dt=physics_dt, 
                               rendering_dt=rendering_dt, 
                               stage_units_in_meters=1.0, 
                               set_defaults = False
                               )
        self._my_world.scene.add_default_ground_plane()

        default_prim_path = str(self.stage.GetDefaultPrim().GetPath())

        # Physics scene
        scene = UsdPhysics.Scene.Define(self.stage, default_prim_path + "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        # Make cylinder
        _, cylinder1_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
        # _, cylinder2_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
        _, cube1_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
        
        
        
        # Get the prim of the cylinder
        cylinder1_prim = self.stage.GetPrimAtPath(cylinder1_path)
        cylinder1_mesh = UsdGeom.Xformable(cylinder1_prim)
        cylinder1_mesh.AddTranslateOp().Set(Gf.Vec3d(0.75, 0, 0.5))
        cylinder1_mesh.AddTransformOp().Set(Gf.Matrix4d().SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,0),290)))
        # cylinder_mesh.AddTransformOp().Set(Gf.Matrix4d())
        cylinder1_mesh.AddScaleOp().Set(Gf.Vec3f(0.2, 0.2, 1))


        # # Get the prim of the cylinder
        # cylinder2_prim = self.stage.GetPrimAtPath(cylinder2_path)
        # cylinder2_mesh = UsdGeom.Xformable(cylinder2_prim)
        # cylinder2_mesh.AddTranslateOp().Set(Gf.Vec3d(0.75, 0, 1.5))
        # cylinder2_mesh.AddTransformOp().Set(Gf.Matrix4d().SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,0),290)))
        # # cylinder_mesh.AddTransformOp().Set(Gf.Matrix4d())
        # cylinder2_mesh.AddScaleOp().Set(Gf.Vec3f(0.15, 0.15, 1))
        
        cube1_prim = self.stage.GetPrimAtPath(cube1_path)
        cube1_mesh = UsdGeom.Xformable(cube1_prim)
        cube1_mesh.AddTranslateOp().Set(Gf.Vec3d(0.75, 0.1, 0.8))
        cube1_mesh.AddTransformOp().Set(Gf.Matrix4d().SetRotateOnly(Gf.Rotation(Gf.Vec3d(0,0,0),290)))
        # cylinder_mesh.AddTransformOp().Set(Gf.Matrix4d())
        cube1_mesh.AddScaleOp().Set(Gf.Vec3f(0.05, 0.25, 0.05))

        simulation_resolution = 20
        
        # Add deformable body
        deformableUtils.add_physx_deformable_body(
            self.stage,
            cylinder1_mesh.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            self_collision=False,
        )


        deformableUtils.add_physx_deformable_body(
            self.stage,
            cube1_mesh.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            self_collision=False,
        )

        # Create a deformable body material
        deformable_material1_path = omni.usd.get_stage_next_free_path(self.stage, "/cylinder1", True)
        deformable_material2_path = omni.usd.get_stage_next_free_path(self.stage, "/cylinder2", True)

        deformableUtils.add_deformable_body_material(
            self.stage,
            deformable_material1_path,
            youngs_modulus=7.5e9,
            poissons_ratio=0.1,
            damping_scale=0.0,
            dynamic_friction=0.5,
            density=10
        )

        deformableUtils.add_deformable_body_material(
            self.stage,
            deformable_material2_path,
            youngs_modulus=3000.0,
            poissons_ratio=0.49,
            damping_scale=0.0,
            dynamic_friction=0.5,
            density = 1
        )
        # Set the created deformable body material on the deformable body
        physicsUtils.add_physics_material_to_prim(self.stage, cylinder1_mesh.GetPrim(), deformable_material1_path) 
        # physicsUtils.add_physics_material_to_prim(self.stage, cylinder2_mesh.GetPrim(), deformable_material2_path) 
        physicsUtils.add_physics_material_to_prim(self.stage, cube1_mesh.GetPrim(), deformable_material2_path) 

        attachment1_path = cylinder1_mesh.GetPath().AppendElementString("attachment1")  
        attachment1 = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment1_path)
        attachment1.GetActor0Rel().SetTargets([cylinder1_mesh.GetPath()])
        attachment1.GetActor1Rel().SetTargets(["/World/defaultGroundPlane/GroundPlane/CollisionPlane"])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment1.GetPrim()) 


        # attachment2_path = cylinder2_mesh.GetPath().AppendElementString("attachment2")  
        # attachment2 = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment2_path)
        # attachment2.GetActor0Rel().SetTargets([cylinder2_mesh.GetPath()])
        # attachment2.GetActor1Rel().SetTargets([cylinder1_mesh.GetPath()])
        # PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment2.GetPrim()) 
    
        attachment2_path = cube1_mesh.GetPath().AppendElementString("attachment2")  
        attachment2 = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment2_path)
        attachment2.GetActor0Rel().SetTargets([cube1_mesh.GetPath()])
        attachment2.GetActor1Rel().SetTargets([cylinder1_mesh.GetPath()])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment2.GetPrim()) 

        assets_root_path = get_assets_root_path()
        
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        
        self.goal = self._my_world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube",
                name="my_goal_cube",
                position=np.array([1.5, 0., 1]),
                color=np.array([0, 1.0, 0]),
                size=0.2,
            )
        )
        
        self.franka = self._my_world.scene.add(
            Franka(
                prim_path="/franka", 
                name="my_franka", 
                # usd_path=franka_asset_path, 
                position = np.array([0, 0, 0]),
            )
        )

        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([-.5, 0.0, 0.2]),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, -20, 0]), degrees=True),
            )
        
        self.camera.initialize()
        self.camera.set_focal_length(2)
        # camera.add_motion_vectors_to_frame()

        
        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(7,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(256, 256, 3), 
            dtype=np.uint8
        )
        
        # self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(16,), dtype=np.float32)

        self.reset_counter = 0
        
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        
        return

    def get_dt(self):
        return self._dt

    def step(self, action):

        # previous_franka_position, previous_franka_orientation = self.franka.get_world_pose()
        # print(previous_franka_position, previous_franka_orientation)
        # previous_gripper_pose = self.franka.

        joint_angles = np.pad(action, (0,2), 'constant', constant_values=(0.5))
        # joint_efforts = np.pad(action, (0,2), 'constant', constant_values=(10))
        # print(np.max(joint_angles), np.min(joint_angles))

        # we apply our actions to the robot
        for i in range(self._skip_frame):

            self.articulation = self.dc.get_articulation("/franka")

            # Call this each frame of simulation step if the state of the articulation is changing.
            self.dc.wake_up_articulation(self.articulation)
            # num_joints = self.dc.get_articulation_joint_count(self.art)
            # num_dofs = self.dc.get_articulation_dof_count(self.art)
            # num_bodies = self.dc.get_articulation_body_count(self.art)

            self.dc.set_articulation_dof_position_targets(self.articulation, joint_angles)
            # self.dc.set_articulation_dof_efforts(self.articulation, joint_efforts)            
            self._my_world.step(render=False)

        observations = self.get_observations()
        # print(type(observations), observations.shape)
        
        info = {}
        terminated = False
        truncated = False
        
        hsv_image = cv2.cvtColor(observations, cv2.COLOR_BGR2HSV)

        # Calculate reward: the number of red pixels
        reward = float(cv2.countNonZero(cv2.inRange(hsv_image, lower_green, upper_green)))
        # print(reward)

        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            truncated = True
        
        # if current_dist_to_goal < 0.1:
        #     terminated = True
        return observations, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self._my_world.reset()
        self.reset_counter = 0
        # Randomize goal location
        self.goal.set_world_pose(np.array([0.9*np.random.rand()+1.1, 0, 0.25*np.random.rand()+0.65]))
        observations = self.get_observations()
        info = {}

        return observations, info

    def get_observations(self):
        self._my_world.render()
        camera_frame = self.camera.get_rgba()[:, :, :3]

        # imgplot = plt.imshow(camera_frame)
        # plt.savefig(log_dir+'/imgplot')
        
        return camera_frame
        # return np.concatenate(
        #     [
        #         jetbot_world_position,
        #         jetbot_world_orientation,
        #         jetbot_linear_velocity,
        #         jetbot_angular_velocity,
        #         goal_world_position,
        #     ]
        # )

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    
    # def create_attachment(self, mesh_object1, mesh_object2):
    #     from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
    #     attachment_path = mesh_object1.GetPath().AppendElementString("attachment")  
    #     attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
    #     attachment.GetActor0Rel().SetTargets([mesh_object1.GetPath()])
    #     attachment.GetActor1Rel().SetTargets(["/World/defaultGroundPlane/GroundPlane/CollisionPlane"])
    #     PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim()) 
        
    #     pass

