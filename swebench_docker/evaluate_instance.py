"""Executed from WITHIN a docker image, to run tests on a task instance"""

import base64
import json
import logging
import os
import sys

from swebench_docker.constants import KEY_PREDICTION, PatchType
from swebench_docker.context_manager import TaskEnvContextManager
from swebench_docker.image import ImageMeta, LogConfig
from swebench_docker.utils import extract_minimal_patch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("evaluate_instance")


def main(
    task_instance: dict,
    image_meta: ImageMeta,
    log_config: LogConfig,
    timeout: int | None,
):
    logger.info(
        "Instance ID: "
        + task_instance["instance_id"]
        + "\nTestbed: "
        + image_meta.testbed_name
        + "\nLog dir: "
        + log_config.directory
    )

    with TaskEnvContextManager(
        task_instance,
        image_meta,
        log_config,
        timeout=timeout,
    ) as tcm:
        # Attempt to apply prediction
        prediction_patch = task_instance[KEY_PREDICTION]

        patch_type = PatchType.PATCH_PRED_TRY

        # If prediction patch doesn't apply, try to do some minor patch refactoring and try again
        if not applicable(prediction_patch, patch_type, tcm):
            prediction_patch = extract_minimal_patch(prediction_patch)
            patch_type = PatchType.PATCH_PRED_MINIMAL_TRY
            if not tcm.apply_patch(prediction_patch, patch_type=patch_type):
                logger.warning("Failed to apply prediction patch")
                sys.exit(1)

        tcm.apply_patch(prediction_patch, patch_type=patch_type, revert=True)

        # Set prediction patch label based on whether patch was edited
        if patch_type is PatchType.PATCH_PRED_MINIMAL_TRY:
            patch_type = PatchType.PATCH_PRED_MINIMAL
        else:
            patch_type = PatchType.PATCH_PRED

        # Run testing script
        task_instance[KEY_PREDICTION] = prediction_patch
        test_patch = task_instance["test_patch"]
        if (
            not applicable(prediction_patch, patch_type, tcm)
            or not applicable(test_patch, PatchType.PATCH_TEST, tcm)
            or not tcm.run_tests_task(task_instance)
        ):
            logger.warning("Evaluation failed")
            sys.exit(1)

        logger.info("Evaluation succeeded")


def applicable(patch: str, patch_type: PatchType, tcm: TaskEnvContextManager) -> bool:
    return (not patch) or tcm.apply_patch(patch, patch_type=patch_type)


def get_task_instance() -> dict:
    TASK_INSTANCE_JSON = "/home/swe-bench/task_instance.json"
    if os.path.exists(TASK_INSTANCE_JSON):
        with open(TASK_INSTANCE_JSON, "r") as f:
            task_instance = json.load(f)
    else:
        instance_b64 = os.getenv("INSTANCE")
        assert instance_b64 is not None, "INSTANCE environment variable is not set"

        task_instance = json.loads(base64.b64decode(instance_b64).decode("utf-8"))
    return task_instance


def get_image_metadata() -> ImageMeta:
    testbed_name = os.getenv("TESTBED_NAME")
    assert testbed_name is not None, "TESTBED_NAME environment variable is not set"

    repo_dir = os.getenv("REPO_DIR") or os.getenv("TESTBED")
    assert repo_dir is not None, "REPO_DIR environment variable is not set"

    image_type = os.getenv("IMAGE_TYPE", "conda")

    return ImageMeta(testbed_name, image_type, repo_dir)


def get_log_config() -> LogConfig:
    log_dir = os.getenv("LOG_DIR")
    assert log_dir is not None, "LOG_DIR environment variable is not set"

    log_suffix = os.getenv("LOG_SUFFIX", "")

    return LogConfig(log_dir, log_suffix)


if __name__ == "__main__":
    task_instance = get_task_instance()
    log_config = get_log_config()
    image_meta = get_image_metadata()

    if timeout_str := os.getenv("TIMEOUT"):
        timeout = int(timeout_str)
    else:
        timeout = None

    main(
        task_instance=task_instance,
        image_meta=image_meta,
        log_config=log_config,
        timeout=timeout,
    )
