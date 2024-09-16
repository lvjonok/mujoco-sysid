import numpy as np
import mujoco

class ActuatorMotor:
    def __init__(self) -> None:
        self.dyn = np.array([1, 0, 0])
        self.gain = np.array([1, 0, 0])
        self.bias = np.array([0, 0, 0])

    def __repr__(self) -> str:
        return f"ActuatorMotor(dyn={self.dyn}, gain={self.gain}, bias={self.bias})"


def update_actuator(model, actuator_id, actuator):
    """
    Update actuator in model
    model - mujoco.MjModel
    actuator_id - int or str (name) (for reference see, named access to model elements)
    actuator - ActuatorMotor, ActuatorPosition, ActuatorVelocity
    """

    model.actuator(actuator_id).dynprm[:3] = actuator.dyn
    model.actuator(actuator_id).gainprm[:3] = actuator.gain
    model.actuator(actuator_id).biasprm[:3] = actuator.bias
    model.actuator(actuator_id).ctrlrange = None


def load_model_with_scene(MJCF_PATH):
    spec= mujoco.MjSpec()
    spec.from_file(MJCF_PATH)

    floor_height = 0 
    wb = spec.worldbody

    ground = wb.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 0.05]
    ground.pos = [0, 0, floor_height]
    ground.material = "floor_material"

    floor_texture = spec.add_texture()
    floor_texture.name = "groundplane"
    floor_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_texture.width = 320
    floor_texture.height = 320
    floor_texture.rgb1 = np.array([0.98, 0.98, 0.98])
    floor_texture.rgb2 = np.array([0.98, 0.98, 0.98])
    floor_texture.builtin = 2
    floor_texture.nchannel = 3
    floor_texture.mark = 1
    floor_texture.markrgb = [0, 0, 0]

    floor = spec.add_material()
    floor.name = "floor_material"
    raw = ["" for _ in range(10)]
    raw[1] = "groundplane"
    floor.textures = raw
    floor.texuniform = 1
    floor.texrepeat = np.array([5, 5])
    floor.reflectance = 0.1
    floor.shininess = 0.6

    skybox = spec.add_texture()
    skybox.name = "skybox"
    skybox.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    skybox.width = 512
    skybox.height = 3072
    skybox.rgb1 = np.zeros(3) * 0.3
    skybox.rgb2 = np.ones(3)
    skybox.builtin = 1
    skybox.nchannel = 3

    height = 12
    width = 10

    poses = [
        np.array([-width, -width, height]),
        np.array([width, -width, height]),
        np.array([0, width, height]),
    ]

    dirs = [
        [1, 1, -1],
        [-1, 1, -1],
        [1, -1, -1],
    ]

    lights = []
    for i in range(len(dirs)):
        light = spec.worldbody.add_light()
        light.active = 1
        light.pos = poses[i]
        light.dir = dirs[i]

        light.specular = np.ones(3) * 0.3
        light.diffuse = np.ones(3) * 0.2
        light.directional = 0

        lights.append(light)

    model = spec.compile()
    return model
