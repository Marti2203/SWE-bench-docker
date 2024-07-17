import os
from dataclasses import dataclass


@dataclass
class LogConfig:
    directory: str
    suffix: str


@dataclass
class ImageMeta:
    testbed_name: str
    image_type: str
    repo_dir: str


def get_log_path(instance_id, model, log_config):
    if log_config.suffix:
        log_file_name = f"{instance_id}.{model}.{log_config.suffix}.eval.log"
    else:
        log_file_name = f"{instance_id}.{model}.eval.log"
    log_file = os.path.join(log_config.directory, log_file_name)
    return log_file
