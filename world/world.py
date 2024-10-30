from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.sensor import Camera
import numpy as np
import omni
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf
from omni.isaac.core import PhysicsContext
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.prims as prims_utils
import gc
import omni.replicator.core as rep
import warp as wp


class SimulationEnv:
    def __init__(self, config):
        self._world = World(physics_dt=config["physics_dt"],
                            rendering_dt=config["rendering_dt"],
                            stage_units_in_meters=1.0,
                            set_defaults=False,
                            backend="torch",
                            device="cuda")
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

    def initialise(self, usd_path, env_usd, hdr_path):
        self.make_ground_plane_small()
        sphere_path = Sdf.Path("/World/defaultGroundPlane/SphereLight")
        sphere_prim = self.stage.GetPrimAtPath(sphere_path)
        intensity_attribute = sphere_prim.GetAttribute('intensity')
        intensity_attribute.Set(0.0)
        self.import_lighting(hdr_path)
        self.import_environment(env_usd)
        self.import_plant_mesh(usd_path)
        self.add_goal_cube()
        self.add_camera()

        # Setup the render product for the camera
        import omni.syntheticdata._syntheticdata as sd
        self.rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        # Setup annotators that will report groundtruth
        self.rgb = rep.AnnotatorRegistry.get_annotator("LdrColorSDIsaacConvertRGBAToRGB")
        self.rgb.attach(self.camera_render_product)


        import omni.graph.core as og

        keys = og.Controller.Keys
        (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/push_graph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("buffer", "omni.syntheticdata.SdPostRenderVarTextureToBuffer"),
                ],
                keys.SET_VALUES: [
                    ("buffer.inputs:renderVar", self.rv)
                ],
            },
        )
    

    def make_ground_plane_small(self):
        path = Sdf.Path("/World/defaultGroundPlane")
        prim = self.stage.GetPrimAtPath(path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translateOp = xform.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        rotateOp = xform.AddRotateXYZOp()
        rotateOp.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        scaleOps = xform.GetOrderedXformOps()
        scaleOp = None
        for op in scaleOps:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scaleOp = op
                break

        # If scaleOp exists, use it; otherwise, create a new one.
        if not scaleOp:
            scaleOp = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)  # Ensure using double precision

        # Set the scale
        scaleOp.Set(Gf.Vec3d(0.007, 0.007, 0.007))

    
    def import_lighting(self, hdr_path:str):
        dome_light = prims_utils.create_prim(
            "/World/Dome_light",
            "DomeLight",
            # position=np.array([1.0, 1.0, 1.0]),
            attributes={
                "inputs:texture:file": hdr_path,
                "inputs:intensity": 1000,
                "inputs:exposure": 1.0,
            }
        )

    
    def import_environment(self, usd_path):
        field = add_reference_to_stage(usd_path=usd_path, prim_path="/Field")
        xform = UsdGeom.Xformable(field)
        xform.ClearXformOpOrder()
        translateOp = xform.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0, 0, 0.01))
        rotateOp = xform.AddRotateXYZOp()
        rotateOp.Set(Gf.Vec3d(0, 0, 0))
        scaleOp = xform.AddScaleOp()
        scaleOp.Set(Gf.Vec3d(0.01, 0.01, 0.01))


    def import_plant_mesh(self, usd_path):
        plant = add_reference_to_stage(usd_path=usd_path, prim_path="/Plant")
        xform = UsdGeom.Xformable(plant)
        xform.ClearXformOpOrder()
        translateOp = xform.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0, -0.15, 0.0))
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
                color=np.array([1.0, 0.0, 0.0]),
                size=0.08,
            )
        )

    # def add_camera(self):
    #     self.camera = Camera(
    #         prim_path="/World/camera",
    #         position=np.array([0.0, -0.04, 0.36]),
    #         frequency=20,
    #         resolution=(256, 256),
    #     )
    #     self.camera.initialize()
    #     self.camera.set_focal_length(1.58)
    #     self.camera.set_clipping_range(0.01, 1000000.0)

    #     self.camera_prim = self.stage.GetPrimAtPath("/World/camera")
    #     # Ensure the camera prim is valid
    #     if not self.camera_prim:
    #         raise ValueError("Camera prim does not exist at the specified path.")

    #     # Add the orientation attribute if it doesn't exist
    #     xform = UsdGeom.Xformable(self.camera_prim)
    #     if not xform.GetXformOpOrderAttr().IsValid():
    #         xform.AddOrientOp()

    #     # Get the orientation attribute
    #     orient_attr = self.camera_prim.GetAttribute("xformOp:orient")
    #     # Define the quaternion components
    #     w = 6.123233995736766e-17
    #     x = -4.329780281177466e-17
    #     y = 0.53833
    #     z = 0.84274

    #     # Create the quaternion using Gf.Quatd
    #     quaternion = Gf.Quatd(w, Gf.Vec3d(x, y, z))

    #     # Set the orientation attribute
    #     orient_attr.Set(quaternion)

    def add_camera(self):
        RESOLUTION = (256, 256)
        self.camera = rep.create.camera(
            position=(0.0, -0.04, 0.36),
            rotation=(0, 0, 90),
            focal_length=15.8,
            focus_distance = 0.0,
            clipping_range=(0.01, 1000000.0),
            f_stop = 0.0
        )
        self.camera_render_product = rep.create.render_product(self.camera, RESOLUTION)

    
    def add_ee_camera(self):
        self.ee_camera = Camera(
            prim_path="/mybuddy/left_arm_l6/ee_camera",
            translation=np.array([0.0, 0.0, 0.04]),
            frequency=20,
            resolution=(256, 256),
        )
        self.ee_camera.initialize()
        self.ee_camera.set_focal_length(1.58)
        self.ee_camera.set_clipping_range(0.01, 1000000.0)

        self.ee_camera_prim = self.stage.GetPrimAtPath("/mybuddy/left_arm_l6/ee_camera")
        # Ensure the camera prim is valid
        if not self.ee_camera_prim:
            raise ValueError("Camera prim does not exist at the specified path.")

        # Add the orientation attribute if it doesn't exist
        xform = UsdGeom.Xformable(self.ee_camera_prim)
        if not xform.GetXformOpOrderAttr().IsValid():
            xform.AddOrientOp()

        # Get the orientation attribute
        orient_attr = self.ee_camera_prim.GetAttribute("xformOp:orient")
        # Define the quaternion components
        w = 0.00
        x = 0.0
        y = 1.0
        z = 0.0

        # Create the quaternion using Gf.Quatd
        quaternion = Gf.Quatd(w, Gf.Vec3d(x, y, z))

        # Set the orientation attribute
        orient_attr.Set(quaternion)

    def get_image(self):
        # Step - Randomize and render
        # rep.orchestrator.step()
        return self.rgb.get_data()["data"]
    
    def get_ee_image(self):
        return self.ee_camera.get_rgba()[:, :, :3]

    @property
    def world(self):
        return self._world
