import base64
import json
import logging
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from docker.models.containers import Container

import docker
from swebench_docker.constants import MAP_VERSION_TO_INSTALL
from swebench_docker.image import LogConfig
import subprocess as sp

logger = logging.getLogger(__name__)

DockerOptions = dict[str, Any]

DOCKER_EVALUATION_COMMAND = "./entrypoint.sh"


async def run_docker_evaluation(
    task_instance: dict,
    namespace: str,
    log_config: LogConfig,
    coverage_target_path: str | None = None,
    timeout: int = 900,
    verbose: bool = False,
    base64_instance: bool = True,
):
    docker_image = get_docker_image(task_instance, namespace)

    docker_options, tmpfile_path = get_docker_options(
        task_instance, log_config, timeout
    )

    instance_id = task_instance["instance_id"]
    log_prefix = f"[{instance_id}][{docker_image}]"

    c = run_docker_cmd(
        docker_image,
        docker_options,
        DOCKER_EVALUATION_COMMAND,
        log_prefix,
        keep_container=coverage_target_path is not None,
    )

    if tmpfile_path:
        # Ensure the temporary file is deleted after the Docker process completes
        os.unlink(tmpfile_path)

    if coverage_target_path:
        assert c
        (code, res) = c.exec_run(
            [
                "bash",
                "-c",
                "find $PWD | grep /.coverage | grep -v .coveragerc | head -n 1",
            ],
            stdout=True,
            stderr=True,
            demux=False,
            stream=False,
        )

        sp.run(
            f"docker cp {c.id}:{res.decode().strip()} {coverage_target_path}",
            shell=True,
            stdout=sp.DEVNULL if not verbose else None,
            stderr=sp.DEVNULL if not verbose else None,
        )

        c.stop()
        c.remove()


def get_codebase(
    task_instance: dict,
    namespace: str,
    log_config: LogConfig,
    codebase_target_path: str,
    verbose: bool = False,
    timeout: int = 900,
):

    docker_image = get_docker_image(task_instance, namespace)

    docker_options, tmpfile_path = get_docker_options(
        task_instance, log_config, timeout
    )

    instance_id = task_instance["instance_id"]
    log_prefix = f"[{instance_id}][{docker_image}]"

    c = run_docker_cmd(
        docker_image,
        docker_options,
        DOCKER_EVALUATION_COMMAND,
        log_prefix,
        keep_container=True,
    )

    x = c.exec_run("bash -c 'echo $PWD'", stdout=True, stderr=True, demux=False)

    root_dir = x[1].decode()

    sp.run(
        f"docker cp {c.id}:{root_dir.strip()}/{task_instance['repo'].replace('/','__')}/. {codebase_target_path}",
        shell=True,
        stdout=sp.DEVNULL if not verbose else None,
        stderr=sp.DEVNULL if not verbose else None,
    )

    c.stop()
    c.remove()

    pass


def get_docker_image(task_instance, namespace):
    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]

    image_prefix = "swe-bench"

    repo_name = task_instance["repo"].replace("/", "_")

    if specifications.get("instance_image", False):
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
    else:
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"
        )

    return docker_image


def get_docker_options(task_instance, log_config, timeout) -> tuple[DockerOptions, str]:
    docker_options_for_instance, tmpfile_path = instance_docker_options(task_instance)

    container_log_dir = get_container_log_dir(task_instance)
    docker_options_for_swe_docker = swe_docker_options(
        container_log_dir, log_config, timeout
    )

    docker_options = merge_docker_options(
        docker_options_for_instance, docker_options_for_swe_docker
    )

    return docker_options, tmpfile_path


def merge_docker_options(a: DockerOptions, b: DockerOptions) -> DockerOptions:
    result = deepcopy(a)

    for k, v in b.items():
        if k in result:
            if isinstance(result[k], list) and isinstance(v, list):
                result[k].extend(v)
            else:
                raise RuntimeError
        else:
            result[k] = deepcopy(v)

    return result


def instance_docker_options(task_instance):
    swebench_docker_fork_dir = os.environ.get("SWEBENCH_DOCKER_FORK_DIR")

    if swebench_docker_fork_dir:
        # Create a temporary file to store the task_instance JSON
        tmpfile_path = tempfile.mktemp(suffix=".json")
        Path(tmpfile_path).write_text(json.dumps(task_instance))

        docker_options_for_instance = _mount_instance_docker_options(
            tmpfile_path, swebench_docker_fork_dir
        )
    else:
        tmpfile_path = ""

        docker_options_for_instance = _env_instance_docker_options(task_instance)
    return docker_options_for_instance, tmpfile_path


def _mount_instance_docker_options(
    instance_file: str, swebench_docker_fork_dir: str
) -> DockerOptions:
    return {
        "user": "root",
        "volumes": [
            f"{swebench_docker_fork_dir}/swebench_docker:/opt/swebench_docker:ro",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench_docker:ro",
            f"{swebench_docker_fork_dir}/swebench_docker:/home/swe-bench/swebench:ro",
            f"{instance_file}:/home/swe-bench/task_instance.json:ro",
        ],
    }


def _env_instance_docker_options(task_instance: dict) -> DockerOptions:
    # Base64 encode the instance JSON to be sure it can be passed as an environment variable
    instance_b64 = base64.b64encode(json.dumps(task_instance).encode("utf-8")).decode(
        "utf-8"
    )
    return {"environment": [f"INSTANCE={instance_b64}"]}


def get_container_log_dir(task_instance):
    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][
        task_instance["version"]
    ]

    # TODO: Change this when deciding
    if "packages" in specifications and specifications["packages"] == "environment.yml":
        container_log_dir = "/home/swe-bench/logs"
    else:
        container_log_dir = "/opt/logs"

    return container_log_dir


def swe_docker_options(container_log_dir, log_config, timeout) -> DockerOptions:
    return {
        "volumes": [f"{log_config.directory}:{container_log_dir}"],
        "environment": [
            f"LOG_DIR={container_log_dir}",
            f"TIMEOUT={timeout}",
            f"LOG_SUFFIX={log_config.suffix}",
        ],
    }


def run_docker_cmd(
    docker_image: str,
    docker_options: DockerOptions,
    docker_command: str,
    log_prefix: str,
    keep_container: bool = False,
) -> Container | None:
    container, creation_options = create_waiting_container(docker_image, docker_options)

    try:
        start_time = time.time()

        container.start()
        # Note: Somehow, using demux=True will crash the evaluation.
        returncode, output_b = container.exec_run(docker_command)

        elapsed_seconds = time.time() - start_time
    except Exception as e:
        logger.warning(f"{log_prefix} Error running container: {e}")
        return
    finally:
        if not keep_container:
            container.stop()
            container.remove(force=True)

    output = output_b.decode()
    docker_options_s = json.dumps(creation_options, indent=2)

    #print("OUTPUT IS", output)
    #print("ERROR CODE IS ", returncode)

    if returncode != 0:
        logger.warning(f"{log_prefix} Error running container:")
        logger.warning(f"Docker creation options: {docker_options_s}")
        logger.warning(f"Docker command: {docker_command}")
        logger.warning(f"Stdout+Stderr - {output}")
    elif "Evaluation succeeded" not in output:
        logger.warning(
            f"{log_prefix} \
                        Container ran successfully in {elapsed_seconds} seconds, but evaluation failed."
        )
        logger.warning(f"Docker creation options: {docker_options_s}")
        logger.warning(f"Docker command: {docker_command}")
        logger.warning(f"stdout+stderr - {output}")
    else:
        logger.info(
            f"{log_prefix} Container ran successfully in {elapsed_seconds} seconds."
        )
    return container if keep_container else None


def create_waiting_container(
    docker_image: str, creation_options: DockerOptions
) -> tuple[Container, DockerOptions]:
    client = docker.from_env()

    creation_options = deepcopy(creation_options)
    creation_options["entrypoint"] = "/bin/bash"
    creation_options["stdin_open"] = True
    creation_options["tty"] = True

    container = client.containers.create(docker_image, **creation_options)

    return container, creation_options
