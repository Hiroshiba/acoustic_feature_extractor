from enum import Enum


class FieldType(str, Enum):
    free = "free"
    diffuse = "diffuse"


class VolumeType(str, Enum):
    rms_power = "rms_power"
    mse_power = "mse_power"
    rms_db = "rms_db"
    mse_db = "mse_db"
