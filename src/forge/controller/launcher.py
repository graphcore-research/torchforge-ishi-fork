# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launcher specific logic (i.e. SLURM, k8s when supported, etc.)"""

from __future__ import annotations

import atexit
import logging
import os
import shlex
import textwrap
from typing import Any

from kubernetes import client
from forge.controller.base import BaseLauncher
from forge.types import Launcher, LauncherConfig
from monarch.actor import ProcMesh
from monarch.job import JobState, JobTrait, SlurmJob
from monarch.job.kubernetes import KubernetesJob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


JOB_NAME_KEY = "job_name"
LAUNCHER_KEY = "launcher"
DEFAULT_KUBERNETES_IMAGE = "nvcr.io/nvidia/pytorch:25.03-py3"
DEFAULT_MONARCH_PORT = 26600

_WORKER_LOOP_SCRIPT = textwrap.dedent("""\
    import os
    import socket
    from monarch.actor import run_worker_loop_forever

    port = os.environ.get("MONARCH_PORT", "26600")
    hostname = socket.getfqdn()
    address = f"tcp://{hostname}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


def get_meshes_from_config(cfg: LauncherConfig) -> dict[str, int]:
    """Extract mesh requirements from launcher config.

    Args:
        cfg: The launcher configuration

    Returns:
        Dictionary mapping mesh names to number of hosts required
    """
    meshes: dict[str, int] = {}

    # Add services that need remote hosts
    # Expand services with multiple replicas into per-replica meshes
    for service_name, service_cfg in cfg.services.items():
        hosts = getattr(service_cfg, "hosts", None)
        if hosts and hosts > 0:
            base_mesh_name = service_cfg.mesh_name or service_name
            num_replicas = service_cfg.num_replicas
            for replica_idx in range(num_replicas):
                mesh_name = f"{base_mesh_name}_{replica_idx}"
                meshes[mesh_name] = hosts

    # Add actors that need remote hosts
    for actor_name, actor_cfg in cfg.actors.items():
        hosts = getattr(actor_cfg, "hosts", None)
        if hosts and hosts > 0:
            mesh_name = actor_cfg.mesh_name or actor_name
            meshes[mesh_name] = hosts

    return meshes


def build_kubernetes_pooled_job_state(
    pooled_host_mesh: Any,
    meshes: dict[str, int],
) -> JobState:
    """Rebuild logical HostMeshes by slicing a pooled Kubernetes HostMesh.

    Args:
        pooled_host_mesh: The single HostMesh returned by KubernetesJob.state().
        meshes: Ordered mapping of logical Forge mesh names to requested host counts.

    Returns:
        JobState exposing the original logical mesh names backed by HostMesh slices.

    Raises:
        RuntimeError: If the pooled HostMesh size does not match the requested host count.
    """
    total_hosts = sum(meshes.values())
    actual_hosts = pooled_host_mesh.extent["hosts"]
    if actual_hosts != total_hosts:
        raise RuntimeError(
            "Pooled Kubernetes HostMesh size mismatch: "
            f"expected {total_hosts} host(s) but received {actual_hosts}."
        )

    host_offset = 0
    logical_meshes: dict[str, Any] = {}
    for forge_mesh_name, host_count in meshes.items():
        logical_meshes[forge_mesh_name] = pooled_host_mesh.slice(
            hosts=slice(host_offset, host_offset + host_count)
        )
        host_offset += host_count

    return JobState(logical_meshes)


def build_kubernetes_worker_pod_spec(
    image: str,
    gpus_per_node: int,
    gc_user: str,
) -> Any:
    worker_command = textwrap.dedent(
        f"""\
        set -euo pipefail
        export PIP_CONSTRAINT=""
        export USER="root"
        set -a
        source "/newdata/$GC_USER/.env"
        set +a
        export UV_CACHE_DIR="/newdata/$GC_USER/.cache/uv"
        export PIP_CACHE_DIR="/newdata/$GC_USER/.cache/pip"
        export UV_PROJECT_ENVIRONMENT="/tmp/ishikori-worker-venv"
        cd "/newdata/$GC_USER/ishikori"
        python -m pip install uv
        uv sync --frozen
        exec uv run python -u -c {shlex.quote(_WORKER_LOOP_SCRIPT)}
        """
    )
    gpu_resources = {"nvidia.com/gpu": str(gpus_per_node)}

    return client.V1PodSpec(
        security_context=client.V1PodSecurityContext(fs_group=0),
        tolerations=[
            client.V1Toleration(
                key="nvidia.com/gpu",
                value="true",
                operator="Equal",
                effect="NoSchedule",
            )
        ],
        containers=[
            client.V1Container(
                name="worker",
                image=image,
                command=["/bin/bash", "-lc", worker_command],
                env=[
                    client.V1EnvVar(
                        name="MONARCH_PORT", value=str(DEFAULT_MONARCH_PORT)
                    ),
                    client.V1EnvVar(name="GC_USER", value=gc_user),
                ],
                readiness_probe=client.V1Probe(
                    tcp_socket=client.V1TCPSocketAction(port=DEFAULT_MONARCH_PORT),
                    period_seconds=4,
                    timeout_seconds=2,
                    failure_threshold=60,
                ),
                resources=client.V1ResourceRequirements(
                    requests=gpu_resources,
                    limits=gpu_resources,
                ),
                volume_mounts=[
                    client.V1VolumeMount(name="newdata", mount_path="/newdata"),
                    client.V1VolumeMount(name="devshm", mount_path="/dev/shm"),
                ],
            )
        ],
        volumes=[
            client.V1Volume(
                name="newdata",
                host_path=client.V1HostPathVolumeSource(path="/newdata"),
            ),
            client.V1Volume(
                name="devshm",
                empty_dir=client.V1EmptyDirVolumeSource(
                    medium="Memory",
                    size_limit="128Gi",
                ),
            ),
        ],
    )


class Slurmlauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a single SlurmJob for all resources.

        This pre-allocates all meshes defined in the config in one Slurm job.

        Returns:
            A tuple of (job, job_state) containing the SlurmJob handle and its state.
        """
        # Collect all mesh requirements from config
        meshes = get_meshes_from_config(self.cfg)

        # If no remote resources needed, skip job creation
        if not meshes:
            return

        # Build slurm_args from config
        slurm_args = [f"--{key}={value}" for key, value in self.cfg.slurm_args.items()]

        # Create a single SlurmJob with all meshes
        logger.info(f"Creating SlurmJob with meshes: {meshes}")
        job = SlurmJob(
            meshes=meshes,  # e.g., {"generator_0": 1, "generator_1": 1, "trainer": 2}
            slurm_args=slurm_args,
            job_name=self.cfg.job_name + "_workers" or "forge_job",
            time_limit="72:00:00",  # Default to 72 hours
            gpus_per_node=self.cfg.gpus_per_node,
            cpus_per_task=self.cfg.cpus_per_task,
            mem=self.cfg.mem,
        )

        # Apply the job to allocate resources
        logger.info("Submitting SlurmJob...")
        job.apply()
        logger.info("SlurmJob submitted, waiting for allocation...")

        # Register cleanup handler
        atexit.register(job.kill)

        # Wait for job allocation
        logger.info("Getting job state (this will block until nodes are allocated)...")
        job_state = job.state(cached_path=None)

        logger.info("SlurmLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


class KubernetesLauncher(BaseLauncher):
    def __init__(
        self,
        cfg: LauncherConfig,
    ):
        self.cfg = cfg

    async def initialize(self) -> tuple[JobTrait, JobState]:
        """Initialize the launcher and create a single KubernetesJob for all resources."""
        meshes = get_meshes_from_config(self.cfg)

        if not meshes:
            return

        kubernetes_args = self.cfg.kubernetes_args
        gc_user = os.environ["GC_USER"]
        worker_image = kubernetes_args.get("image", DEFAULT_KUBERNETES_IMAGE)
        pooled_mesh_name = f"mesh0{gc_user}"[:63]
        total_hosts = sum(meshes.values())

        job = KubernetesJob(
            namespace=kubernetes_args["namespace"],
            timeout=kubernetes_args.get("timeout"),
        )

        job.add_mesh(
            name=pooled_mesh_name,
            num_replicas=total_hosts,
            pod_spec=build_kubernetes_worker_pod_spec(
                image=worker_image,
                gpus_per_node=self.cfg.gpus_per_node,
                gc_user=gc_user,
            ),
            labels=kubernetes_args.get("labels"),
        )

        logger.info(f"Creating KubernetesJob with meshes: {meshes}")
        job.apply()
        logger.info("KubernetesJob submitted, waiting for allocation...")

        atexit.register(job.kill)

        raw_state = job.state(cached_path=None)
        try:
            pooled_host_mesh = getattr(raw_state, pooled_mesh_name)
        except AttributeError as err:
            raise RuntimeError(
                "KubernetesJob did not return the expected pooled HostMesh "
                f"'{pooled_mesh_name}'."
            ) from err
        job_state = build_kubernetes_pooled_job_state(pooled_host_mesh, meshes)

        logger.info("KubernetesLauncher initialization complete.")
        return job, job_state

    async def remote_setup(self, procs: ProcMesh) -> None:
        return


def get_launcher(cfg: LauncherConfig | None = None) -> BaseLauncher | None:
    if not cfg:
        return None
    if cfg.launcher == Launcher.SLURM:
        return Slurmlauncher(cfg)
    elif cfg.launcher == Launcher.KUBERNETES:
        return KubernetesLauncher(cfg)
    elif cfg.launcher == Launcher.MAST:
        try:
            from forge.fb.mast_launcher import MastLauncher

            return MastLauncher(cfg)
        except ImportError as err:
            raise ValueError("MAST is not available, cannot launch MAST jobs.") from err

    else:
        raise ValueError(f"Unsupported config provided, got {cfg}")
