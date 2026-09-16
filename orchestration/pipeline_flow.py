# orchestration/pipeline_flow.py
"""
CreditPulse — End-to-End Pipeline Orchestration (Phase 5)

Runs the full pipeline in dependency order:
    Bronze load → Silver clean → Gold build → ML → Survival → CLV → Uplift

Each step is a Prefect task with:
  - Retry logic for transient failures
  - A 10-minute timeout so a hung task can't block the pipeline forever
  - Structured logging to the Prefect UI

Run manually:
    python orchestration/pipeline_flow.py

Schedule (optional):
    prefect deployment build orchestration/pipeline_flow.py:creditpulse_pipeline \
        --name "nightly" --cron "0 6 * * *" --apply
"""

import sys
import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SQL_DIR = PROJECT_ROOT / "sql"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def _run_script(script_path: Path, extra_args=None, logger=None) -> None:
    """Run a Python script as a subprocess, streaming output to Prefect logs.

    Prefers the project's .venv Python so the flow works even if the shell
    that launched Prefect hadn't sourced the venv.
    """
    if logger is None:
        logger = get_run_logger()
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    cmd = [python_exe, str(script_path)] + (extra_args or [])
    logger.info(f"Running: {' '.join(str(c) for c in cmd[1:])}")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info(f"  │ {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.warning(f"  │ {line}")

    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} exited with code {result.returncode}")


# ============================================================
# PIPELINE TASKS
# ============================================================

@task(name="load_bronze", retries=2, retry_delay_seconds=30, timeout_seconds=600)
def load_bronze():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "load_to_postgres.py", logger=logger)
    logger.info("✅ Bronze layer loaded")


@task(name="clean_to_silver", retries=2, retry_delay_seconds=30, timeout_seconds=600)
def clean_to_silver():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "clean_bronze_to_silver.py", logger=logger)
    logger.info("✅ Silver layer rebuilt")


@task(name="build_gold", retries=1, retry_delay_seconds=30, timeout_seconds=300)
def build_gold():
    logger = get_run_logger()
    sql_path = SQL_DIR / "create_gold_tables.sql"
    _run_script(SCRIPTS_DIR / "run_sql_file.py",
                extra_args=[str(sql_path)], logger=logger)
    logger.info("✅ Gold layer rebuilt")


@task(name="create_survival_view", retries=1, retry_delay_seconds=15, timeout_seconds=120)
def create_survival_view():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "create_survival_view.py", logger=logger)
    logger.info("✅ Survival view created")


@task(name="train_ml_models", retries=1, retry_delay_seconds=60, timeout_seconds=600)
def train_ml_models():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "train_ml_models.py", logger=logger)
    logger.info("✅ Lead-scoring model trained and scored")


@task(name="survival_analysis", retries=1, retry_delay_seconds=60, timeout_seconds=600)
def survival_analysis():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "survival_analysis.py", logger=logger)
    logger.info("✅ Survival model fitted")


@task(name="clv_modeling", retries=1, retry_delay_seconds=60, timeout_seconds=600)
def clv_modeling():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "clv_modeling.py", logger=logger)
    logger.info("✅ CLV computed")


@task(name="uplift_modeling", retries=1, retry_delay_seconds=60, timeout_seconds=600)
def uplift_modeling():
    logger = get_run_logger()
    _run_script(SCRIPTS_DIR / "uplift_modeling.py", logger=logger)
    logger.info("✅ Uplift model fitted")


# ============================================================
# PIPELINE FLOW
# ============================================================

@flow(
    name="creditpulse_pipeline",
    description="End-to-end CreditPulse pipeline: Bronze → Silver → Gold → ML → Survival → CLV → Uplift",
    log_prints=True,
)
def creditpulse_pipeline():
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("CREDITPULSE PIPELINE — Prefect orchestration")
    logger.info("=" * 60)

    # Phase 1-3: Data pipeline
    load_bronze()
    clean_to_silver()
    build_gold()
    create_survival_view()

    # Phase 6-9: ML + analytics stack
    train_ml_models()
    survival_analysis()
    clv_modeling()
    uplift_modeling()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    creditpulse_pipeline()