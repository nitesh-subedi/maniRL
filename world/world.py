from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.sensor import Camera
import numpy as np
import omni
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics
from omni.isaac.core import PhysicsContext
from omni.isaac.core.utils.stage import add_reference_to_stage


class SimulationEnv:
    def __init__(self, config):
        self._world = World(physics_dt=config["physics_dt"],
                            rendering_dt=config["rendering_dt"],
                            stage_units_in_meters=1.0,
                            set_defaults=False)
        self._world.scene.add_default_ground_plane()
        self.stage = omni.usd.get_context().get_stage()
        self.config = config

        self.physics_context = PhysicsContext()
        self.physics_context.set_solver_type("TGS")
        self.physics_context.set_broadphase_type("GPU")
        self.physics_context.enable_gpu_dynamics(True)
        # Physics scene
        default_prim_path = str(self.stage.GetDefaultPrim().GetPath())
        scene = UsdPhysics.Scene.Define(self.stage, default_prim_path + "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    def initialise(self, usd_path):
        self.import_plant_mesh(usd_path)
        self.add_goal_cube()
        self.add_camera()

    def import_plant_mesh(self, usd_path):
        plant = add_reference_to_stage(usd_path=usd_path, prim_path="/Plant")
        xform = UsdGeom.Xformable(plant)
        xform.ClearXformOpOrder()
        translateOp = xform.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0, -0.22, 0.0))
        rotateOp = xform.AddRotateXYZOp()
        rotateOp.Set(Gf.Vec3d(0, 0, 0))
        scaleOp = xform.AddScaleOp()
        scaleOp.Set(Gf.Vec3d(0.05, 0.05, 0.05))

        # Get childerns
        meshes = plant.GetAllChildren()
        meshes = [mesh.GetAllChildren()[0] for mesh in meshes]
        meshes = meshes[1:]
        prim_dict = dict(zip(plant.GetAllChildrenNames()[1:], meshes))
        self.make_deformable(prim_dict)

        # # Contact offset
        # contact_offset_attr = self.cylinder_prim.GetAttribute("physxCollision:contactOffset")
        # contact_offset_attr.Set(0.001)

        self.attach_cylinder_to_ground(prim_dict)

    def make_deformable(self, prim_dict, simulation_resolution=10):
        key, value = list(prim_dict.items())[0]
        deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=10,
                self_collision=False,
            )
        
        for key, value in list(prim_dict.items())[1:]:
            deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )

        # self.deformable_cylinder_material_path = omni.usd.get_stage_next_free_path(self.stage, "/cylinder1", True)
        # deformableUtils.add_deformable_body_material(
        #     self.stage,
        #     self.deformable_cylinder_material_path,
        #     youngs_modulus=2e7,
        #     poissons_ratio=0.15,
        #     damping_scale=0.0,
        #     dynamic_friction=0.5,
        #     density=1000
        # )

        # physicsUtils.add_physics_material_to_prim(self.stage, self.cylinder_mesh.GetPrim(),
        #                                           self.deformable_cylinder_material_path)

    def attach_cylinder_to_ground(self, prim_dict):
        key, value = list(prim_dict.items())[0]
        attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
        stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
        stalk_attachment.GetActor0Rel().SetTargets([value.GetPath()])
        stalk_attachment.GetActor1Rel().SetTargets(["/World/defaultGroundPlane/GroundPlane/CollisionPlane"])
        auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())
        # Set attributes to reduce initial movement and gap
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:deformableVertexOverlapOffset').Set(0.01)
        # auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:rigidSurfaceSamplingDistance').Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableVertexAttachments').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableRigidSurfaceAttachments').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableCollisionFiltering').Set(True)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:collisionFilteringOffset').Set(0.01)
        auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableFilteringPairs').Set(True)
    
        for key, value in list(prim_dict.items())[1:]:
            attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
            stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
            stalk_attachment.GetActor0Rel().SetTargets([value.GetPath()])
            stalk_attachment.GetActor1Rel().SetTargets(["/Plant/stalk/plant_023"])
            auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())


    def add_goal_cube(self):
        self.goal_cube = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube",
                name="my_goal_cube",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([0, 1.0, 0]),
                size=0.08,
            )
        )

    def add_camera(self):
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([0.0, -0.04, 0.36]),
            frequency=20,
            resolution=(256, 256),
        )
        self.camera.initialize()
        self.camera.set_focal_length(1.58)
        self.camera.set_clipping_range(0.1, 1000000.0)

        self.camera_prim = self.stage.GetPrimAtPath("/World/camera")
        # Ensure the camera prim is valid
        if not self.camera_prim:
            raise ValueError("Camera prim does not exist at the specified path.")

        # Add the orientation attribute if it doesn't exist
        xform = UsdGeom.Xformable(self.camera_prim)
        if not xform.GetXformOpOrderAttr().IsValid():
            xform.AddOrientOp()

        # Get the orientation attribute
        orient_attr = self.camera_prim.GetAttribute("xformOp:orient")
        # Define the quaternion components
        w = 6.123233995736766e-17
        x = -4.329780281177466e-17
        y = 0.53833
        z = 0.84274

        # Create the quaternion using Gf.Quatd
        quaternion = Gf.Quatd(w, Gf.Vec3d(x, y, z))

        # Set the orientation attribute
        orient_attr.Set(quaternion)

    def get_image(self):
        return self.camera.get_rgba()[:, :, :3]

    @property
    def world(self):
        return self._world
