from omni.importer.urdf import _urdf
from omni.isaac.dynamic_control import _dynamic_control
import omni
import numpy as np
from pxr import PhysicsSchemaTools


class Robot:
    def __init__(self, urdf_path: str, world=None, simulation_app=None) -> None:
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.fix_base = True
        import_config.make_default_prim = False
        import_config.self_collision = True
        import_config.collision_from_visuals = True
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        import_config.default_drive_strength = 0.08
        import_config.default_position_drive_damping = 0.1
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 1
        import_config.density = 0.0

        # Finally import the robot
        result, self.robot_prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf_path,
                                                                 import_config=import_config)
        self.world = world
        self.world.step()
        self.simulation_app = simulation_app
        self.simulation_app.update()
        if not result:
            raise RuntimeError("Failed to import URDF")

        # Initialize empty lists to hold the DOF pointers
        self.left_arm_dof_ptrs = []
        self.right_arm_dof_ptrs = []

        self.dc = None
        self.root_articulation = None

        omni.timeline.get_timeline_interface().play()
        self.world.step()
        self.world.step()
        self.initialise_control_interface()
        print(
            f"Robot initialised successfully, {self.root_articulation}, {self.dc.wake_up_articulation(self.root_articulation)}")
        if not self.dc.wake_up_articulation(self.root_articulation):
            raise RuntimeError("Failed to wake up articulation.")

        # self.robot_links = ["left_arm_l1", "left_arm_l2", "left_arm_l3", "left_arm_l4", "left_arm_l5",
        #                     "left_arm_l6", "right_arm_l1", "right_arm_l2", "right_arm_l3", "right_arm_l4",
        #                     "right_arm_l5", "right_arm_l6"]
        self.robot_links = ["left_arm_l1", "left_arm_l2", "left_arm_l3", "left_arm_l4", "left_arm_l5",
                            "left_arm_l6"]
        self.collision_paths = [f"/mybuddy/{link}/collisions" for link in self.robot_links]
        self.collision_paths.append("/World/defaultGroundPlane/GroundPlane/CollisionPlane")
        self.collision_checker_interface = omni.physx.get_physx_scene_query_interface()
        self.collision_meshes_for_checking = []
        self.initialise_collision_api()
        self.ee_dc=self.dc.get_rigid_body("/mybuddy/left_arm_l6")

    @staticmethod
    def on_hit(hit):
        # print("collision with ", hit.collision)
        return True

    def initialise_collision_api(self):
        for collision_path in self.collision_paths:
            mesh_1, mesh_2 = PhysicsSchemaTools.encodeSdfPath(collision_path)
            self.collision_meshes_for_checking.append([mesh_1, mesh_2])

    def check_collision(self):
        for i, collision_mesh in enumerate(self.collision_meshes_for_checking):
            num_hits = self.collision_checker_interface.overlap_mesh(collision_mesh[0], collision_mesh[1], self.on_hit,
                                                                     False)
            if i == 5:
                if num_hits >= 3:
                    print(f'Collision between {collision_mesh[0]} and {collision_mesh[1]}')
                    print(num_hits)
                    return True, True
            if num_hits >= 4:
                # print(f'Collision between {collision_mesh[0]} and {collision_mesh[1]}')
                print(num_hits)
                return True, False
        return False, False

    def initialise_control_interface(self):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.root_articulation = self.dc.get_articulation("/mybuddy/base_link")
        # Assuming `dc` is your simulation controller and `root_articulation` is your robot's root articulation
        left_arm_joints = ["left_arm_j1", "left_arm_j2", "left_arm_j3", "left_arm_j4", "left_arm_j5", "left_arm_j6"]
        right_arm_joints = ["right_arm_j1", "right_arm_j2", "right_arm_j3", "right_arm_j4", "right_arm_j5",
                            "right_arm_j6"]

        # Get DOF pointers for the left arm joints
        for joint_name in left_arm_joints:
            dof_ptr = self.dc.find_articulation_dof(self.root_articulation, joint_name)
            self.left_arm_dof_ptrs.append(dof_ptr)

        # Get DOF pointers for the right arm joints
        for joint_name in right_arm_joints:
            dof_ptr = self.dc.find_articulation_dof(self.root_articulation, joint_name)
            self.right_arm_dof_ptrs.append(dof_ptr)

    def get_prim_path(self):
        return self.robot_prim_path

    def send_angles(self, arm_index: int, angles: list[float], degrees: bool = False):
        if self.dc is None:
            raise RuntimeError("Control interface not initialised. Please initialise the control interface first.")
        else:
            self.dc.wake_up_articulation(self.root_articulation)

        if degrees:
            angles = np.deg2rad(angles)

        if arm_index == 0:
            for dof_ptr, angle in zip(self.left_arm_dof_ptrs, angles):
                self.dc.set_dof_position_target(dof_ptr, angle)

        elif arm_index == 1:
            for dof_ptr, angle in zip(self.right_arm_dof_ptrs, angles):
                self.dc.set_dof_position_target(dof_ptr, angle)

        else:
            raise ValueError("Invalid arm index. Please provide 0 for left arm and 1 for right arm.")
    
    def get_ee_position(self):
        return self.dc.get_rigid_body_pose(self.ee_dc).p