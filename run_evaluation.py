#!/usr/bin/env python3

"""Run evaluation"""
import argparse
import asyncio
import logging
from os.path import exists, isdir

from swebench import get_eval_refs

from swebench_docker.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    MAP_REPO_TO_TEST_FRAMEWORK,
)
from swebench_docker.image import LogConfig, get_log_path
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_directives, load_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation")


async def main(
    predictions_path: str,
    swe_bench_tasks: str,
    namespace: str,
    log_dir: str,
    log_suffix: str = "",
    skip_existing: bool = False,
    timeout: int = 900,
    num_processes: int = -1,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.

    Args:
        predictions_path (str): Path to the predictions file.
        swe_bench_tasks (str): Path to the SWE-bench tasks file OR HF dataset name.
        namespace (str): Docker repository namespace.
        log_dir (str): Path to the directory where logs will be saved.
        log_suffix (str): Suffix to append to log file names.
        skip_existing (bool): Whether to skip evaluations for predictions that already have logs.
        timeout (int): Timeout for each evaluation.
        num_processes (int): Number of processes to run in parallel (-1 = unlimited)

    Raises:
        ValueError: If log_dir is not a directory, testbed is not a directory, or swe_bench_tasks does not exist.
    """
    if not isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")

    if not any(predictions_path.endswith(x) for x in [".json", ".jsonl"]):
        raise ValueError("Predictions path must be .json or .jsonl file")

    predictions = load_predictions(predictions_path)
    logger.info(f"# of predictions loaded: {len(predictions)}")

    tasks = list(get_eval_refs(swe_bench_tasks).values())
    tasks_map = {t[KEY_INSTANCE_ID]: t for t in tasks}

    log_config = LogConfig(log_dir, log_suffix)
    predictions = filter_predictions(
        predictions,
        tasks_map.keys(),
        skip_existing,
        log_config,
    )

    task_instances = [build_task_instance(p, tasks_map) for p in predictions]
    task_instances.sort(key=lambda x: x[KEY_INSTANCE_ID])

    await run_task_instances(
        task_instances, namespace, log_config, timeout, num_processes
    )


def filter_predictions(predictions, tasks_ids, skip_existing, log_config):
    def has_necessary_fields(pred):
        return all(x in pred for x in [KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION])

    if not all(has_necessary_fields(pred) for pred in predictions):
        raise ValueError(
            f"Every prediction must have {KEY_INSTANCE_ID}, {KEY_MODEL}, and {KEY_PREDICTION} fields"
        )

    result = [p for p in predictions if p[KEY_INSTANCE_ID] in tasks_ids]
    logger.info(f"# of predictions with valid task ID: {len(result)}")

    not_in_tasks = [p for p in predictions if p[KEY_INSTANCE_ID] not in tasks_ids]
    if len(not_in_tasks) > 0:
        logger.warning(
            "Predictions for the following instance_ids were not "
            + "found in the tasks file and will not be considered: "
            + ", ".join(not_in_tasks)
        )

    if skip_existing:
        result = [
            p
            for p in predictions
            if not exists(get_log_path(p[KEY_INSTANCE_ID], p[KEY_MODEL], log_config))
        ]
        logger.info(f"# of predictions not already evaluated: {len(result)}")

    return result


def build_task_instance(prediction, tasks_map):
    instance_id = prediction[KEY_INSTANCE_ID]
    task = tasks_map[instance_id]

    test_type = MAP_REPO_TO_TEST_FRAMEWORK[task["repo"]]
    test_directives = get_test_directives(task)
    test_cmd = f"{test_type} {' '.join(test_directives)}"

    task_instance = {
        "repo": task["repo"],
        "version": task["version"],
        "base_commit": task["base_commit"],
        KEY_INSTANCE_ID: prediction[KEY_INSTANCE_ID],
        KEY_MODEL: prediction[KEY_MODEL],
        KEY_PREDICTION: prediction[KEY_PREDICTION],
        "test_patch": task["test_patch"],
        "test_directives": test_directives,
        "test_cmd": test_cmd,
    }
    return task_instance


async def run_task_instances(
    task_instances, namespace, log_config, timeout, num_processes
):
    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(task_instances))

    async with asyncio.TaskGroup() as tg:
        for task_instance in task_instances:
            task = run_docker_evaluation_throttled(
                sem,
                task_instance,
                namespace,
                log_config,
                timeout,
            )
            tg.create_task(task)


async def run_docker_evaluation_throttled(sem, *args, **kwargs):
    async with sem:
        return await run_docker_evaluation(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_path", type=str, help="Path to predictions file", required=True
    )
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to dataset file or HF datasets name",
        required=True,
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        help="(Optional) Suffix to append to log file names",
        default="",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="(Optional) Skip existing logs"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="(Optional) Timeout in seconds (default: 900)",
        default=1800,
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        help="(Optional) Number of processes to run in parallel (-1 for unlimited)",
        default=-1,
    )
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
