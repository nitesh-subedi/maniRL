from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.sensor import Camera
import numpy as np
import omni
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import UsdGeom, Gf, UsdPhysics, PhysxSchema
from omni.isaac.core import PhysicsContext


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

    def initialise(self):
        self.add_deformable_cylinder()
        self.add_goal_cube()
        self.add_camera()

    def add_deformable_cylinder(self):
        _, self.cylinder_path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
        self.cylinder_prim = self.stage.GetPrimAtPath(self.cylinder_path)
        self.cylinder_mesh = UsdGeom.Xformable(self.cylinder_prim)

        # Clear previous transformop
        self.cylinder_mesh.ClearXformOpOrder()
        translateOp = self.cylinder_mesh.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0, -0.22, 0.2))
        existing_scaleOps = [op for op in self.cylinder_mesh.GetOrderedXformOps() if
                             op.GetOpType() == UsdGeom.XformOp.TypeScale]
        if existing_scaleOps:
            scaleOp = existing_scaleOps[0]
        else:
            scaleOp = self.cylinder_mesh.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)

        scaleOp.Set(Gf.Vec3d(0.08, 0.08, 0.4))
        self.make_deformable()

        # Contact offset
        contact_offset_attr = self.cylinder_prim.GetAttribute("physxCollision:contactOffset")
        contact_offset_attr.Set(0.01)

        self.attach_cylinder_to_ground()

    def make_deformable(self, simulation_resolution=10):
        deformableUtils.add_physx_deformable_body(
            self.stage,
            self.cylinder_mesh.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            self_collision=False,
        )
        self.deformable_cylinder_material_path = omni.usd.get_stage_next_free_path(self.stage, "/cylinder1", True)
        deformableUtils.add_deformable_body_material(
            self.stage,
            self.deformable_cylinder_material_path,
            youngs_modulus=2e7,
            poissons_ratio=0.15,
            damping_scale=0.0,
            dynamic_friction=0.5,
            density=1000
        )

        physicsUtils.add_physics_material_to_prim(self.stage, self.cylinder_mesh.GetPrim(),
                                                  self.deformable_cylinder_material_path)

    def attach_cylinder_to_ground(self):
        # Attach the cylinder to the ground
        cylinder_attachment_path = self.cylinder_mesh.GetPath().AppendElementString("attachment1")
        self.cylinder_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, cylinder_attachment_path)
        self.cylinder_attachment.GetActor0Rel().SetTargets([self.cylinder_mesh.GetPath()])
        self.cylinder_attachment.GetActor1Rel().SetTargets(["/World/defaultGroundPlane/GroundPlane/CollisionPlane"])

        self.auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(self.cylinder_attachment.GetPrim())

        # Set attributes to reduce initial movement and gap
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:deformableVertexOverlapOffset').Set(0.06)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:rigidSurfaceSamplingDistance').Set(0.01)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableVertexAttachments').Set(
            True)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableRigidSurfaceAttachments').Set(True)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableCollisionFiltering').Set(True)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:collisionFilteringOffset').Set(0.01)
        self.auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:enableDeformableFilteringPairs').Set(True)

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
