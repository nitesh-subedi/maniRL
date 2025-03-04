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
import omni.replicator.core as rep
import omni.isaac.core.utils.render_product as rp
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import asyncio
from omni.kit.viewport.utility import get_active_viewport


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


        plane_prim = prims_utils.create_prim(
             prim_path="/World/Background_Cube",
             prim_type="Cube",
             position=np.array([0.0, -0.7, 1.0]),
             scale=np.array([1, 0.01, 1]),
         )
            
    def sphere_light(self, intensity, radius):
        sphere_path = Sdf.Path("/World/defaultGroundPlane/SphereLight")
        sphere_prim = self.stage.GetPrimAtPath(sphere_path)
        light_intensity_attribute = sphere_prim.GetAttribute('intensity')
        light_intensity_attribute.Set(intensity)
        radius_attr = sphere_prim.GetAttribute('radius')
        radius_attr.Set(radius)
        # exposure_attr = sphere_prim.GetAttribute('exposure')
        # exposure_attr.Set(0.5)
        translateop = UsdGeom.Xformable(sphere_prim)
        translateop.ClearXformOpOrder()
        translateOp = translateop.AddTranslateOp()
        translateOp.Set(Gf.Vec3d(0.0, 0.0, 2.0))

    def initialise(self, usd_path, env_usd, hdr_path):
        self.sphere_light(9000, 1.0)
        # self.make_ground_plane_small()
        # self.import_lighting(hdr_path)
        # self.import_environment(env_usd)
        self.import_plant_mesh(usd_path, prim_name="/Plant", translation=[0.0, -0.2, 0.0], rotation=[0, 0, 0])
        self.import_plant_mesh(usd_path, prim_name="/Plant2", translation=[0.1, -0.2, 0.0], rotation=[0, 0, 90])
        self.import_plant_mesh(usd_path, prim_name="/Plant3", translation=[-0.1, -0.2, 0.0], rotation=[0, 0, 45])
        self.import_plant_mesh(usd_path, prim_name="/Plant4", translation=[0.12, -0.2, 0.0], rotation=[0, 0, 180])
        self.add_goal_cube()
        self.add_rgb_camera()
        self.add_depth_camera()
        self.viewport = get_active_viewport()
        viewport_cam_rp_path = self.viewport.render_product_path
        self.viewport_cam_rp = self.stage.GetPrimAtPath(viewport_cam_rp_path)
        viewport_cam = self.viewport.get_active_camera()
        viewport_cam_prim = self.stage.GetPrimAtPath(viewport_cam)
        translation_attr = viewport_cam_prim.GetAttribute("xformOp:translate")
        translation_attr.Set(Gf.Vec3d(-0.5,-0.5, 0.15))
        rotation_attr = viewport_cam_prim.GetAttribute("xformOp:rotateXYZ")
        rotation_attr.Set(Gf.Vec3d(0, 0, -90))

        # Setup the render product for the camera
        import omni.syntheticdata._syntheticdata as sd
        self.rgb_rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        self.depth_rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Depth.name)
        asyncio.run(self.setup_and_evaluate())

    async def setup_and_evaluate(self):
        import omni.graph.core as og
        keys = og.Controller.Keys
        (texture_graph_handle, texture_list_of_nodes, _, _) = og.Controller.edit(
            {"graph_path": "/push_graph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("rgb_texture_to_device_buffer", "omni.syntheticdata.SdPostRenderVarTextureToBuffer"),
                    ("depth_texture_to_device_buffer", "omni.syntheticdata.SdPostRenderVarTextureToBuffer"),
                    ("rgb_device_to_host_buffer", "omni.syntheticdata.SdPostRenderVarToHost"),
                    ("depth_device_to_host_buffer", "omni.syntheticdata.SdPostRenderVarToHost"),
                ],
                keys.SET_VALUES: [
                    ("rgb_texture_to_device_buffer.inputs:renderVar", self.rgb_rv),
                    ("depth_texture_to_device_buffer.inputs:renderVar", self.depth_rv),
                ],
                keys.CONNECT: [
                    ("rgb_texture_to_device_buffer.outputs:renderVar", "rgb_device_to_host_buffer.inputs:renderVar"),
                    ("depth_texture_to_device_buffer.outputs:renderVar", "depth_device_to_host_buffer.inputs:renderVar"),
                ],
            },
        )

        # Await the evaluation
        await og.Controller.evaluate(texture_graph_handle)
        
        # Retrieve attributes
        rgb_device_to_host_buffer = texture_list_of_nodes[2]
        depth_device_to_host_buffer = texture_list_of_nodes[3]
        rgb_new_render_var = rgb_device_to_host_buffer.get_attribute("outputs:renderVar").get(on_gpu=True)
        depth_new_render_var = depth_device_to_host_buffer.get_attribute("outputs:renderVar").get(on_gpu=True)
        
        print(f"New render var for RGB: {rgb_new_render_var}")
        print(f"New render var for Depth: {depth_new_render_var}")
        # Setup annotators that will report groundtruth
        self.rgb = rep.AnnotatorRegistry.get_annotator("LdrColorSDIsaacConvertRGBAToRGB")
        self.depth = rep.AnnotatorRegistry.get_annotator("DepthLinearized")
        self.rgb.renderVar = rgb_new_render_var
        self.depth.renderVar = depth_new_render_var
        self.rgb.attach(self.rgb_camera_render_product)
        self.depth.attach(self.depth_camera_render_product)
    

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
        self.dome_light = prims_utils.create_prim(
            "/World/Dome_light",
            "DomeLight",
            # orientation=euler_angles_to_quat(np.random.uniform(0, np.pi * 2, 3)),
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


    def import_plant_mesh(self, usd_path, prim_name, translation, rotation):
        plant = add_reference_to_stage(usd_path=usd_path, prim_path=prim_name)
        xform = UsdGeom.Xformable(plant)
        xform.ClearXformOpOrder()
        plant_translateOp = xform.AddTranslateOp()
        plant_translateOp.Set(Gf.Vec3d(*translation))
        rotateOp = xform.AddRotateXYZOp()
        rotateOp.Set(Gf.Vec3d(*rotation))
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

        self.attach_cylinder_to_ground(prim_dict, prim_name)
        # key, self.stem_prim = list(prim_dict.items())[0] # Step prim path
        # # print(f"Stem prim path: {self.stem_prim.GetPath()}")
        # self.poif = XFormPrim("/Tracker1", position = Gf.Vec3d(0.077, 0.09, 0.36))
        # # Attach the self.poif to the stem
        # attachment_path = self.stem_prim.GetPath().AppendElementString(f"attachment_tracker")
        # stalk_attachment = PhysxSchema.PhysxPhysicsAttachment.Define(self.stage, attachment_path)
        # stalk_attachment.GetActor0Rel().SetTargets([self.stem_prim.GetPath()])
        # stalk_attachment.GetActor1Rel().SetTargets(["/Tracker1"])
        # auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())
        # Set attributes to reduce initial movement and gap
        # auto_attachment_api.GetPrim().GetAttribute('physxAutoAttachment:deformableVertexOverlapOffset').Set(0.01)
        


    def make_deformable(self, prim_dict, simulation_resolution=10):
        key, value = list(prim_dict.items())[0]
        # Add deformable material
        deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)

        # Create the material
        deformableUtils.add_deformable_body_material(
            self.stage,
            deformable_material_path,
            youngs_modulus=7.5e19,
            poissons_ratio=0.1,
            damping_scale=0.0,
            dynamic_friction=0.5,
            density=1000
        )
        deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )
        physicsUtils.add_physics_material_to_prim(self.stage, value.GetPrim(), deformable_material_path)
        
        for key, value in list(prim_dict.items())[1:]:
            deformableUtils.add_physx_deformable_body(
                self.stage,
                value.GetPath(),
                collision_simplification=True,
                simulation_hexahedral_resolution=simulation_resolution,
                self_collision=False,
            )
            physicsUtils.add_physics_material_to_prim(self.stage, value.GetPrim(), deformable_material_path)
         
        
    def get_stem_location(self):
        position, orientation = self.poif.get_world_pose()
        return position
       

    def attach_cylinder_to_ground(self, prim_dict, prim_name):
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
            stalk_attachment.GetActor1Rel().SetTargets([f"{prim_name}/stalk/plant_023"])
            auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(stalk_attachment.GetPrim())


    def add_goal_cube(self):
        self.goal_cube = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_1",
                name="my_goal_cube_1",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([1.0, 0.0, 0.0]),
                size=0.08,
            )
        )
        self.goal_cube_2 = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_2",
                name="my_goal_cube_2",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([0.0, 1.0, 0.0]),
                size=0.08,
            )
        )
        self.goal_cube_3 = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_3",
                name="my_goal_cube_3",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([0.0, 0.0, 1.0]),
                size=0.08,
            )
        )
        self.goal_cube_4 = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_4",
                name="my_goal_cube_4",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([1.0, 0.0, 1.0]),
                size=0.08,
            )
        )
        self.goal_cube_5 = self._world.scene.add(
            VisualCuboid(
                prim_path="/goal_cube_5",
                name="my_goal_cube_5",
                position=np.array([0, -0.4, 0.22]),
                color=np.array([0.0, 1.0, 1.0]),
                size=0.08,
            )
        )

    def add_rgb_camera(self):
        RESOLUTION = (256, 256)
        self.rgb_camera = rep.create.camera(
            position=(0.0, -0.04, 0.36),
            rotation=(0, 0, 90),
            focal_length=15.8,
            focus_distance = 0.0,
            clipping_range=(0.01, 1000000.0),
            f_stop = 0.0
        )
        self.rgb_camera_render_product = rep.create.render_product(self.rgb_camera, RESOLUTION)
    
    def add_depth_camera(self):
        RESOLUTION = (256, 256)
        self.depth_camera = rep.create.camera(
            position=(0.0, -0.04, 0.36),
            rotation=(0, 0, 90),
            focal_length=15.8,
            focus_distance = 0.0,
            clipping_range=(0.01, 1000000.0),
            f_stop = 0.0
        )
        self.depth_camera_render_product = rep.create.render_product(self.depth_camera, RESOLUTION)

    
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
        return self.rgb.get_data()["data"] , self.depth.get_data()
    
    def get_ee_image(self):
        return self.ee_camera.get_rgba()[:, :, :3]

    @property
    def world(self):
        return self._world