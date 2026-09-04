
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as _dt
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import smtplib
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - reported at load time
    yaml = None

__version__ = "5.0.0"

PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)\b")
QUANTIZATION_VALUES = ("none", "bnb_4bit")
METRIC_REFUSALS_RE = re.compile(r"Refusals observed:\s*(\d+)(?:\s*/\s*(\d+))?", re.I)
METRIC_KL_RE = re.compile(r"KL divergence:\s*([0-9.eE+-]+)", re.I)


class ConfigError(Exception):
    """Configuration or usage error with a user-actionable message."""


class PipelineStageError(Exception):
    """A requested post-processing stage (gguf/ollama/hf) failed."""


# --------------------------------------------------------------------------- #
# Configuration model
# --------------------------------------------------------------------------- #

@dataclass
class AbliterationConfig:
    model: Optional[str] = None            # HF id or local path; validated at run time
    output_subdir: str = ""                # per-model working/export dir name override
    n_trials: int = 200                    # real Settings.n_trials (default 200)
    quantization: str = "none"             # real: none | bnb_4bit
    export_strategy: str = "merge"         # real ExportStrategy value, e.g. merge
    device_map: Optional[str] = None       # real --device-map (auto/cpu/...); None = omit
    max_shard_size: Optional[str] = None   # optional --max-shard-size, e.g. "4GB"
    batch_size: Optional[int] = None       # optional --batch-size
    max_batch_size: Optional[int] = None   # optional --max-batch-size
    attempts: int = 1                      # subprocess retries
    timeout_minutes: float = 180.0


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    tasks: list = field(default_factory=lambda: ["mmlu", "truthfulqa_mc2"])
    limit: Optional[int] = None            # examples per task (lm_eval --limit)
    output_subdir: str = "benchmarks"


@dataclass
class HardwareConfig:
    device: Optional[str] = None           # legacy informational (NOT a CLI flag)
    device_map: Optional[str] = None       # -> --device-map; "auto" for GPU boxes
    num_workers: int = 1
    vram_gb: float = 0.0                   # informational only
    batch_size_autotune: bool = False      # informational only


@dataclass
class PipelineConfig:
    convert_to_gguf: bool = False
    gguf_outtype: str = "f16"              # llama.cpp convert --outtype
    convert_script: Optional[str] = None   # explicit llama.cpp convert script path
    ollama_import: bool = False
    ollama_model_name: Optional[str] = None  # tag if not derived from model id
    upload_to_hf: bool = False
    hf_repo_id: Optional[str] = None
    hf_token_env: str = "HF_TOKEN"
    abort_on_error: bool = True


@dataclass
class EmailConfig:
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    username: Optional[str] = None
    password: Optional[str] = None
    from_addr: Optional[str] = None
    to_addrs: list = field(default_factory=list)
    use_tls: bool = True
    notify_on: str = "all"                 # all | success | failure


@dataclass
class WebhookConfig:
    url: Optional[str] = None
    notify_on: str = "all"


@dataclass
class NotificationConfig:
    enable: bool = False
    email: EmailConfig = field(default_factory=EmailConfig)
    discord: WebhookConfig = field(default_factory=WebhookConfig)
    slack: WebhookConfig = field(default_factory=WebhookConfig)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    notes: str = ""


@dataclass
class Config:
    abliteration: AbliterationConfig = field(default_factory=AbliterationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    resume: bool = False
    dry_run: bool = False
    output_root: str = "output"
    log_level: str = "INFO"
    unresolved_env: tuple = ()             # (var_name, ...) found unset during load

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: Optional[dict], logger: Optional[logging.Logger] = None) -> "Config":
        """Build a Config from a (possibly partial) dict without crashing.

        Unknown keys and keys belonging to another section are skipped with a
        warning.  Every field keeps its dataclass default when absent, so a
        bare {} or a notification-only dict are both valid.
        """
        logger = logger or logging.getLogger("heretic_final")
        raw = raw or {}
        section_map = {
            "abliteration": AbliterationConfig,
            "benchmark": BenchmarkConfig,
            "hardware": HardwareConfig,
            "pipeline": PipelineConfig,
            "notification": NotificationConfig,
            "wandb": WandbConfig,
        }
        known_top = {
            "resume", "dry_run", "output_root", "log_level",
            *section_map.keys(),
        }
        cfg = cls()

        # 1) scalar top-level fields
        for key in ("resume", "dry_run", "output_root", "log_level"):
            if key in raw and raw[key] is not None:
                setattr(cfg, key, raw[key])

        # 2) section dicts
        for section, dcls in section_map.items():
            data = raw.get(section)
            if data is None:
                continue
            if not isinstance(data, dict):
                logger.warning("config section '%s' is not a mapping; ignored", section)
                continue
            obj = dcls()
            valid = {f.name for f in dataclasses.fields(dcls)}
            for k, v in data.items():
                if k not in valid:
                    logger.warning("config: unknown key '%s.%s' ignored", section, k)
                    continue
                setattr(obj, k, v)
            setattr(cfg, section, obj)

        # 3) legacy flattened keys (old example layouts) -> fold into abliteration
        legacy_abl = {
            "model", "quantization", "optimization_trials", "early_stopping",
            "max_refusals", "n_trials", "export_strategy", "device_map",
        }
        abl = cfg.abliteration
        for k, v in raw.items():
            if k in known_top or k in legacy_abl is False and k in section_map:
                continue
            if k == "optimization_trials":
                abl.n_trials = int(v)
            elif k in ("early_stopping", "max_refusals"):
                logger.warning(
                    "config key '%s' is not a real heretic-llm setting and was ignored",
                    k,
                )
            elif k in legacy_abl and k != "max_refusals":
                if k == "n_trials":
                    abl.n_trials = int(v)
                elif k == "quantization":
                    abl.quantization = str(v)
                elif k == "export_strategy":
                    abl.export_strategy = str(v)
                elif k == "model":
                    abl.model = str(v)
                elif k == "device_map":
                    abl.device_map = str(v)
            elif k not in known_top:
                logger.warning("config: unknown top-level key '%s' ignored", k)

        cfg._validate_static(logger)
        return cfg

    def _validate_static(self, logger: Optional[logging.Logger] = None) -> None:
        logger = logger or logging.getLogger("heretic_final")
        q = self.abliteration.quantization
        if q not in QUANTIZATION_VALUES:
            logger.warning(
                "quantization '%s' is not a real heretic value (%s); passing it "
                "will make the engine CLI reject the run",
                q, "|".join(QUANTIZATION_VALUES),
            )

    # ------------------------------------------------------------------ #
    # (de)serialization helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("unresolved_env", None)
        return d

    def to_yaml(self) -> str:
        if yaml is None:  # pragma: no cover
            raise ConfigError("PyYAML is required to write YAML (pip install pyyaml)")
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def dump(self, path: str) -> None:
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # convenience accessors
    # ------------------------------------------------------------------ #
    def model_list(self) -> list:
        m = self.abliteration.model
        return [m] if m else []

    def effective_workers(self) -> int:
        return max(1, int(self.hardware.num_workers or 1))

    def notify_channels_enabled(self) -> bool:
        n = self.notification
        if not n.enable:
            return False
        e = n.email
        has_email = bool(e.smtp_server and e.smtp_port and e.to_addrs)
        return has_email or bool(n.discord.url) or bool(n.slack.url)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def make_logger(name: str = "heretic_final",
                console: bool = True,
                logfile: Optional[str] = None,
                level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(_formatter())
        logger.addHandler(ch)
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(_formatter())
        logger.addHandler(fh)
    return logger


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# Environment substitution (with warnings for unresolved placeholders)
# --------------------------------------------------------------------------- #

def _substitute_env(text: str,
                    logger: Optional[logging.Logger] = None,
                    origin: str = "") -> str:
    """Replace ${VAR}/$VAR from os.environ.

    Unset variables are left verbatim so they stay visible in the final
    config and can disable the notification path instead of being silently
    emptied.  A warning naming each missing variable is emitted.
    """
    logger = logger or logging.getLogger("heretic_final")
    missing: list = []

    def _repl(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        value = os.environ.get(name)
        if value is None:
            if name not in missing:
                missing.append(name)
                logger.warning(
                    "environment variable '%s' is not set%s; placeholder kept "
                    "verbatim - affected optional/notification settings will "
                    "be treated as disabled",
                    name, f" (in {origin})" if origin else "",
                )
            return m.group(0)
        return value

    return PLACEHOLDER_RE.sub(_repl, text)


def _resolve_env_in_value(value: Any, logger: logging.Logger, origin: str) -> Any:
    if isinstance(value, str):
        return _substitute_env(value, logger=logger, origin=origin)
    if isinstance(value, list):
        return [_resolve_env_in_value(v, logger, origin) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env_in_value(v, logger, f"{origin}.{k}") for k, v in value.items()}
    return value


def _unresolved_in(value: Any) -> list:
    """Return names of ${...} placeholders still present in value."""
    if isinstance(value, str):
        return [m.group(1) or m.group(2) for m in PLACEHOLDER_RE.finditer(value)]
    if isinstance(value, list):
        out: list = []
        for v in value:
            out.extend(_unresolved_in(v))
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_unresolved_in(v))
        return out
    return []


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

EXAMPLE_CONFIG_YAML = """\
# Heretic Enhanced v5 - generated starter config
# Real heretic-llm 1.4.0 settings only. Run:  python heretic_final.py -c config.yaml
abliteration:
  model: Qwen/Qwen3-4B-Instruct-2507   # HF id or local dir; REQUIRED to run
  n_trials: 150
  quantization: none                   # real values: none | bnb_4bit
  export_strategy: merge               # real ExportStrategy value (e.g. merge)
  device_map: auto                     # real --device-map; omit for CPU-only
  # max_shard_size: 4GB
  # batch_size: 8
  # max_batch_size: 16
  attempts: 1
  timeout_minutes: 180
hardware:
  # device: cuda            # informational only - never passed to the CLI
  device_map: auto          # -> --device-map auto (real flag)
  num_workers: 1
  vram_gb: 0.0
  batch_size_autotune: false
benchmark:
  enabled: true
  tasks: ["mmlu", "truthfulqa_mc2"]
  limit: null
pipeline:
  convert_to_gguf: false
  ollama_import: false
  upload_to_hf: false
  hf_repo_id: null
  abort_on_error: true
notification:
  enable: false
  email:
    smtp_server: ${SMTP_SERVER}
    smtp_port: 587
    username: ${EMAIL_USER}
    password: ${EMAIL_PASS}
    from_addr: ${EMAIL_FROM}
    to_addrs: ["${EMAIL_TO}"]
    use_tls: true
    notify_on: all
  discord:
    url: ${DISCORD_WEBHOOK}
    notify_on: all
  slack:
    url: ${SLACK_WEBHOOK}
    notify_on: all
wandb:
  enabled: false
  project: null
  entity: null
output_root: output
resume: false
dry_run: false
log_level: INFO
"""


def load_config(path: str, logger: logging.Logger) -> Config:
    if yaml is None:
        raise ConfigError("PyYAML is required (pip install pyyaml)")
    p = Path(path)
    if not p.is_file():
        raise ConfigError(
            f"config file not found: {path}\n"
            f"Generate one with: python heretic_final.py --write-config config.yaml"
        )
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config root of {path} must be a mapping")

    resolved = _resolve_env_in_value(raw, logger, origin=str(p))
    cfg = Config.from_dict(resolved, logger=logger)
    unresolved = _unresolved_in(resolved)
    cfg.unresolved_env = tuple(dict.fromkeys(unresolved))
    if unresolved and cfg.notification.enable:
        names = ", ".join(sorted(set(unresolved)))
        logger.warning(
            "unresolved env placeholders present (%s); notification channels "
            "referencing them will be disabled", names,
        )
    return cfg


# --------------------------------------------------------------------------- #
# Model id / path helpers
# --------------------------------------------------------------------------- #

def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model_id).strip("_") or "model"


def model_out_dir(output_root: str, model_id: str, subdir: str = "") -> Path:
    base = Path(output_root) / safe_name(model_id)
    return base / subdir if subdir else base


# --------------------------------------------------------------------------- #
# Engine CLI construction  (the part that must match real heretic exactly)
# --------------------------------------------------------------------------- #

def build_heretic_command(cfg: Config,
                          model: str,
                          logger: logging.Logger) -> list:
    """Construct the real-heretic command line.

    Verified contract for heretic-llm 1.4.0:
      * pydantic-settings CLI source: --kebab-case flags with implicit values.
      * main.py injects "--model" immediately before the LAST argv element,
        therefore the model id must be the final token of the command.
      * No such flag as --max-refusals / --device / --config exists.
    """
    a = cfg.abliteration
    if not model:
        raise ConfigError("no model configured (set abliteration.model or pass --model)")
    engine_cmd = os.environ.get("HERETIC_ENGINE") or shutil.which("heretic") or "heretic"

    cmd: list = [engine_cmd]
    cmd += ["--n-trials", str(int(a.n_trials))]

    if a.quantization not in QUANTIZATION_VALUES:
        raise ConfigError(
            f"invalid quantization '{a.quantization}' (real values: none | bnb_4bit)"
        )
    cmd += ["--quantization", a.quantization]

    if not a.export_strategy:
        raise ConfigError("abliteration.export_strategy must not be empty")
    cmd += ["--export-strategy", str(a.export_strategy).lower()]

    device_map = a.device_map or cfg.hardware.device_map
    if device_map:
        if str(device_map).lower() in ("cuda", "gpu"):
            logger.warning(
                "device_map '%s' is not a real heretic value; using 'auto' "
                "(real --device-map accepts e.g. auto/cpu)", device_map,
            )
            device_map = "auto"
        cmd += ["--device-map", str(device_map)]
    elif cfg.hardware.device and str(cfg.hardware.device).lower() not in ("cpu", "none", ""):
        logger.warning(
            "hardware.device is not a real heretic CLI setting and is ignored; "
            "set hardware.device_map or abliteration.device_map to pass --device-map"
        )

    if a.max_shard_size:
        cmd += ["--max-shard-size", str(a.max_shard_size)]
    if a.batch_size is not None:
        cmd += ["--batch-size", str(int(a.batch_size))]
    if a.max_batch_size is not None:
        cmd += ["--max-batch-size", str(int(a.max_batch_size))]

    # MODEL MUST BE LAST - real main.py inserts --model before the final token.
    cmd.append(model)
    return cmd


# --------------------------------------------------------------------------- #
# Abliteration engine
# --------------------------------------------------------------------------- #

@dataclass
class EngineOutcome:
    success: bool
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    error: str = ""
    command: list = field(default_factory=list)


class AbliterationEngine:
    """Runs one `heretic <model> ...` job in a per-model working directory."""

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("heretic_final")

    def run(self, model: str, out_dir: Path) -> EngineOutcome:
        cmd = build_heretic_command(self.config, model, self.logger)
        a = self.config.abliteration
        timeout_s = float(a.timeout_minutes) * 60.0
        attempts = max(1, int(a.attempts))

        if self.config.dry_run:
            self.logger.info("DRY RUN (would execute): %s (in %s)", " ".join(cmd), out_dir)
            return EngineOutcome(success=True, command=cmd,
                                 metrics={"dry_run": True})

        out_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Running: %s (in %s)", " ".join(cmd), out_dir)

        last_err = ""
        for attempt in range(1, attempts + 1):
            self.logger.info("Attempt %d/%d: %s in %s", attempt, attempts,
                             " ".join(cmd), out_dir)
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(out_dir),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_s,
                    check=False,
                )
                duration = time.time() - t0
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                # keep a durable copy of the engine transcript
                _append_text(out_dir / "engine.log", stdout + stderr)
                if proc.returncode == 0:
                    metrics = self._parse_metrics(stdout)
                    if not metrics:
                        metrics = self._parse_metrics(stderr)
                    self.logger.info("Completed %s in %.2f min", model, duration / 60.0)
                    return EngineOutcome(
                        success=True, returncode=0, stdout=stdout, stderr=stderr,
                        duration_s=duration, metrics=metrics, command=cmd,
                    )
                last_err = (
                    f"engine exited with code {proc.returncode}\n"
                    f"stderr: {(stderr or stdout).strip()[-2000:]}"
                )
                self.logger.error("Engine failed (attempt %d/%d): %s",
                                  attempt, attempts, last_err)
            except subprocess.TimeoutExpired:
                last_err = f"engine timed out after {timeout_s:.0f}s"
                self.logger.error("Attempt %d/%d: %s", attempt, attempts, last_err)
            except FileNotFoundError as exc:
                raise ConfigError(
                    f"engine executable not found: {cmd[0]}. Install heretic-llm "
                    f"(pip install heretic-llm) or set HERETIC_ENGINE.") from exc
            except OSError as exc:
                last_err = f"failed to start engine: {exc}"
                self.logger.error("Attempt %d/%d: %s", attempt, attempts, last_err)
            if attempt < attempts:
                time.sleep(2)

        return EngineOutcome(success=False, error=last_err, command=cmd)

    @staticmethod
    def _parse_metrics(text: str) -> dict:
        metrics: dict = {}
        m = METRIC_REFUSALS_RE.search(text)
        if m:
            metrics["refusals"] = int(m.group(1))
            if m.group(2):
                metrics["refusals_total"] = int(m.group(2))
        m = METRIC_KL_RE.search(text)
        if m:
            metrics["kl_divergence"] = float(m.group(1))
        if re.search(r"abliteration complete", text, re.I):
            metrics["abliteration_complete"] = True
        return metrics


# --------------------------------------------------------------------------- #
# Post pipeline: GGUF / Ollama / HF upload
# --------------------------------------------------------------------------- #

def _find_convert_script(config: Config, logger: logging.Logger) -> Optional[str]:
    """Locate llama.cpp's HF->GGUF converter."""
    explicit = config.pipeline.convert_script
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return str(p)
        logger.warning("configured convert_script not found: %s", explicit)
    env_script = os.environ.get("LLAMA_CPP_CONVERT")
    if env_script and Path(env_script).is_file():
        return env_script
    for name in ("convert_hf_to_gguf.py", "convert.py"):
        found = shutil.which(name)
        if found:
            return found
    # common llama.cpp checkout locations
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _has_hf_dir(directory: Path) -> bool:
    return directory.is_dir() and (
        (directory / "config.json").is_file()
        or any(directory.glob("*.safetensors"))
        or any(directory.glob("pytorch_model*.bin"))
    )


def _find_engine_artifact(out_dir: Path, logger: logging.Logger) -> Path:
    """The merged model produced by `heretic ... --export-strategy merge`.

    Real heretic writes the merged/exported model into the working directory
    it is executed in (which the wrapper sets to the per-model out dir).
    """
    sub = out_dir / "merged"
    if _has_hf_dir(sub):
        return sub
    if _has_hf_dir(out_dir):
        return out_dir
    # search a couple of known layouts before giving up
    for cand in out_dir.iterdir():
        if cand.is_dir() and _has_hf_dir(cand):
            return cand
    return out_dir


def _run_stage(command: list, cwd: Path, logger: logging.Logger,
               what: str, timeout_s: float = 3600.0) -> str:
    logger.info("Stage %s: %s (in %s)", what, " ".join(command), cwd)
    proc = subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout_s, check=False,
    )
    tail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-2000:]
    if proc.returncode != 0:
        raise PipelineStageError(
            f"{what} failed (exit {proc.returncode}): {tail}"
        )
    return (proc.stdout or "") + (proc.stderr or "")


def _stage_gguf(config: Config, model: str, out_dir: Path,
                logger: logging.Logger) -> dict:
    pl = config.pipeline
    src = _find_engine_artifact(out_dir, logger)
    if not _has_hf_dir(src):
        listing = "\n".join(p.name for p in out_dir.iterdir())[:500] or "(empty)"
        raise PipelineStageError(
            f"GGUF conversion requested but no HuggingFace model artifact was "
            f"found under {out_dir} to convert.\nDirectory contents:\n{listing}"
        )
    converter = _find_convert_script(config, logger)
    if converter is None:
        raise PipelineStageError(
            "convert_to_gguf requested but llama.cpp convert script not found. "
            "Install llama.cpp or point pipeline.convert_script at "
            "convert_hf_to_gguf.py (or set LLAMA_CPP_CONVERT)."
        )
    gguf_out = out_dir / f"{safe_name(model)}.gguf"
    cmd = [
        sys.executable, converter, str(src),
        "--outfile", str(gguf_out),
        "--outtype", pl.gguf_outtype or "f16",
    ]
    _run_stage(cmd, cwd=out_dir, logger=logger, what="gguf-convert",
               timeout_s=float(config.abliteration.timeout_minutes) * 60.0)
    if not gguf_out.is_file():
        raise PipelineStageError(f"GGUF converter reported success but {gguf_out} is missing")
    logger.info("GGUF artifact written: %s", gguf_out)
    return {"gguf_path": str(gguf_out)}


def _stage_ollama(config: Config, model: str, out_dir: Path,
                  logger: logging.Logger, artifacts: dict) -> dict:
    if shutil.which("ollama") is None:
        raise PipelineStageError(
            "ollama_import requested but the 'ollama' binary is not on PATH"
        )
    gguf = Path(artifacts.get("gguf_path") or out_dir / f"{safe_name(model)}.gguf")
    if not gguf.is_file():
        raise PipelineStageError(
            f"ollama import requested but GGUF file {gguf} does not exist "
            f"(enable pipeline.convert_to_gguf first)"
        )
    tag = config.pipeline.ollama_model_name or safe_name(model).lower().replace("_", "-")
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(f"FROM {gguf.name}\n", encoding="utf-8")
    _run_stage(
        ["ollama", "create", tag, "-f", str(modelfile)],
        cwd=out_dir, logger=logger, what="ollama-create", timeout_s=1800.0,
    )
    logger.info("Ollama model imported as '%s'", tag)
    return {"ollama_model": tag}


def _stage_hf_upload(config: Config, model: str, out_dir: Path,
                     logger: logging.Logger, artifacts: dict) -> dict:
    repo = config.pipeline.hf_repo_id
    if not repo:
        raise PipelineStageError(
            "upload_to_hf requested but pipeline.hf_repo_id is not set"
        )
    token = os.environ.get(config.pipeline.hf_token_env or "HF_TOKEN")
    if not token:
        raise PipelineStageError(
            f"upload_to_hf requested but env var '{config.pipeline.hf_token_env or 'HF_TOKEN'}' "
            f"is not set (huggingface-cli will also need it)"
        )
    if shutil.which("huggingface-cli") is None:
        raise PipelineStageError(
            "upload_to_hf requested but 'huggingface-cli' is not installed "
            "(pip install huggingface_hub)"
        )
    src = artifacts.get("gguf_path")
    if src:
        target = Path(src)
        cmd = ["huggingface-cli", "upload", repo, str(target)]
    else:
        src = _find_engine_artifact(out_dir, logger)
        if not _has_hf_dir(src):
            raise PipelineStageError(
                f"no model artifact found under {out_dir} to upload"
            )
        cmd = ["huggingface-cli", "upload", repo, str(src)]
    _run_stage(cmd, cwd=out_dir, logger=logger, what="hf-upload", timeout_s=3600.0)
    logger.info("Uploaded to HF repo %s", repo)
    return {"hf_repo": repo}


def _run_post_pipeline(config: Config, model: str, out_dir: Path,
                       logger: logging.Logger) -> dict:
    """Return dict of stage artifacts; raise PipelineStageError on failure."""
    artifacts: dict = {}
    pl = config.pipeline
    if pl.convert_to_gguf:
        artifacts.update(_stage_gguf(config, model, out_dir, logger))
    if pl.ollama_import:
        artifacts.update(_stage_ollama(config, model, out_dir, logger, artifacts))
    if pl.upload_to_hf:
        artifacts.update(_stage_hf_upload(config, model, out_dir, logger, artifacts))
    return artifacts


# --------------------------------------------------------------------------- #
# Benchmarking (best effort, informational)
# --------------------------------------------------------------------------- #

def run_benchmarks(config: Config, model: str, out_dir: Path,
                   logger: logging.Logger) -> dict:
    bench = config.benchmark
    if not bench.enabled or not bench.tasks:
        return {"benchmarks": {}}
    if shutil.which("lm_eval") is None and shutil.which("lm-eval") is None:
        logger.warning(
            "benchmarking enabled but 'lm_eval' is not installed; skipped "
            "(pip install lm-eval-harness). This does not fail the run."
        )
        return {"benchmarks": {"skipped": "lm_eval not installed"}}
    model_dir = _find_engine_artifact(out_dir, logger)
    tasks = ",".join(bench.tasks)
    cmd = ["lm_eval",
           "--model", "hf",
           "--model_args", f"pretrained={model_dir}",
           "--tasks", tasks,
           "--output_path", str(out_dir / bench.output_subdir)]
    if bench.limit:
        cmd += ["--limit", str(bench.limit)]
    bdir = out_dir / bench.output_subdir
    bdir.mkdir(parents=True, exist_ok=True)
    logger.info("Benchmark: %s", " ".join(cmd))
    try:
        _run_stage(cmd, cwd=out_dir, logger=logger, what="benchmark",
                   timeout_s=float(config.abliteration.timeout_minutes) * 60.0)
        return {"benchmarks": {"status": "completed", "tasks": bench.tasks}}
    except PipelineStageError as exc:
        logger.error("Benchmarks failed (informational, not fatal): %s", exc)
        return {"benchmarks": {"status": "failed", "error": str(exc)}}


# --------------------------------------------------------------------------- #
# Notifications
# --------------------------------------------------------------------------- #

class Notifier:
    """Send email / discord / slack notifications.

    Every channel fails soft: exceptions and refused connections are logged as
    errors and return False -- they never crash the pipeline (round-3 fix).
    Channels whose config still contains unresolved ${VAR} placeholders are
    treated as disabled with a warning (round-4 fix).
    """

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("heretic_final")
        self.unresolved: set = set(config.unresolved_env or ())

    # -- helpers --------------------------------------------------------- #
    def _channel_disabled(self, label: str, value: str) -> bool:
        """Return True if a config string is unusable (empty/unresolved)."""
        if value is None or not str(value).strip():
            return True
        if "${" in str(value):
            if label:
                self.logger.warning(
                    "notification channel '%s' disabled: unresolved env "
                    "placeholder in %r", label, value,
                )
            return True
        return False

    def _should_notify(self, notify_on: str, success: bool) -> bool:
        mode = (notify_on or "all").lower()
        return mode == "all" or (mode == "success" and success) or (
            mode == "failure" and not success)

    # -- email ----------------------------------------------------------- #
    def _send_email(self, subject: str, body: str, success: bool) -> bool:
        e = self.config.notification.email
        if self._channel_disabled("email.smtp_server", e.smtp_server):
            return False
        if not e.to_addrs:
            self.logger.warning("email enabled but no to_addrs configured")
            return False
        to_addrs = [a for a in e.to_addrs if not self._channel_disabled("email.to", a)]
        if not to_addrs:
            return False
        from_addr = e.from_addr or e.username
        if self._channel_disabled("email.from_addr", from_addr):
            return False
        if e.username and self._channel_disabled("email.username", e.username):
            return False
        if e.username and self._channel_disabled("email.password", e.password or ""):
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.set_content(body)

        try:
            with smtplib.SMTP(str(e.smtp_server), int(e.smtp_port),
                              timeout=30) as smtp:
                if e.use_tls:
                    smtp.starttls()
                if e.username:
                    smtp.login(e.username, e.password)
                smtp.send_message(msg)
            self.logger.info("Email notification sent to %s", ", ".join(to_addrs))
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Email failed: %s", exc)
            return False

    # -- webhooks -------------------------------------------------------- #
    def _post_webhook(self, url: str, payload: dict) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "heretic-enhanced/5.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Webhook POST failed (%s): %s", url[:64], exc)
            return False

    def _send_webhook(self, channel: str, cfg_url: Any, text: str,
                      success: bool) -> bool:
        if self._channel_disabled(channel, cfg_url):
            return False
        url = str(cfg_url)
        payload = {"content": text} if channel == "discord" else {"text": text}
        if self._post_webhook(url, payload):
            self.logger.info("%s notification sent", channel.title())
            return True
        return False

    # -- public ---------------------------------------------------------- #
    def summarize(self, results: list, dry_run: bool = False) -> None:
        if not self.config.notification.enable:
            return
        n = self.config.notification
        failed = [r for r in results if not r.get("success")]
        subject = ("[Heretic] dry-run planned" if dry_run else
                   "[Heretic] run failed (%d/%d)" % (len(failed), len(results))
                   if failed else "[Heretic] run succeeded (%d models)" % len(results))
        lines = [subject, ""]
        for r in results:
            icon = "OK" if r.get("success") else "FAIL"
            lines.append(f"  [{icon}] {r['model']}")
            if not r.get("success") and r.get("error"):
                lines.append(f"        {r['error']}")
        body = "\n".join(lines)
        success = not failed

        if not self._should_notify(n.email.notify_on, success) and \
                not self._should_notify(n.discord.notify_on, success) and \
                not self._should_notify(n.slack.notify_on, success):
            return

        if self._should_notify(n.email.notify_on, success):
            self._send_email(subject, body, success)
        if self._should_notify(n.discord.notify_on, success):
            self._send_webhook("discord", n.discord.url, body, success)
        if self._should_notify(n.slack.notify_on, success):
            self._send_webhook("slack", n.slack.url, body, success)

    def send_test_email(self) -> bool:
        if not self.config.notification.enable:
            self.logger.warning("--test-email requested but notification.enable is false")
            return False
        return self._send_email(
            "[Heretic] test email",
            "This is a test email from Heretic Enhanced.",
            success=True,
        )


# --------------------------------------------------------------------------- #
# Single-model pipeline
# --------------------------------------------------------------------------- #

def _append_text(path: Path, text: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    except OSError:
        pass


def _record_result_json(out_dir: Path, result: dict) -> None:
    try:
        (out_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        logging.getLogger("heretic_final").warning(
            "could not write result.json: %s", exc)


def process_single_model(config: Config, model: str,
                         logger: logging.Logger,
                         enable_benchmarks: bool = True) -> dict:
    """Run the full per-model pipeline. Returns the result dict."""

    t0 = time.time()
    out_root = Path(config.output_root)
    out_dir = out_root / safe_name(model)
    result: dict = {
        "model": model,
        "success": False,
        "started_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "output_dir": str(out_dir),
        "dry_run": bool(config.dry_run),
    }

    # resume support: a previously successful final marker skips the model
    marker = out_dir / "SUCCESS.json"
    if config.resume and marker.is_file():
        logger.info("Resume: %s already succeeded; skipping", model)
        result.update({"success": True, "resumed": True})
        _record_result_json(out_dir, result)
        return result

    a = config.abliteration
    model_cfg = copy.deepcopy(config)
    model_cfg.abliteration = copy.deepcopy(a)
    model_cfg.abliteration.model = model

    try:
        if config.dry_run:
            cmd = build_heretic_command(model_cfg, model, logger)
            result["command"] = cmd
            if config.pipeline.convert_to_gguf or config.pipeline.ollama_import \
                    or config.pipeline.upload_to_hf:
                logger.info("DRY RUN: post pipeline stages would run for %s", model)
            result["success"] = True
            result["metrics"] = {"dry_run": True}
            result["duration_s"] = time.time() - t0
            _record_result_json(out_dir, result)
            return result

        # 1) engine (real heretic CLI)
        engine = AbliterationEngine(model_cfg, logger=logger)
        outcome = engine.run(model, out_dir=out_dir)
        result["command"] = outcome.command
        if outcome.command:
            logger.info("engine command: %s", " ".join(outcome.command))
        if not outcome.success:
            result["error"] = outcome.error or "engine failed"
            result["returncode"] = outcome.returncode
            _record_result_json(out_dir, result)
            return result
        result["returncode"] = 0
        result["metrics"] = outcome.metrics
        result["stdout_tail"] = (outcome.stdout or outcome.stderr)[-2000:]
        result["duration_s"] = round(time.time() - t0, 3)
        logger.info("Starting post-pipeline stages for %s", model)

        # 2) GGUF / ollama / HF upload
        try:
            artifacts = _run_post_pipeline(model_cfg, model, out_dir, logger)
        except (PipelineStageError, ConfigError) as exc:
            if config.pipeline.abort_on_error:
                result["success"] = False
                result["error"] = str(exc)
                result["pipeline_error"] = str(exc)
                logger.error("Pipeline error: %s", exc)
                _record_result_json(out_dir, result)
                return result
            logger.warning(
                "post-pipeline stage failed but abort_on_error=false; run "
                "continues as success with degraded output: %s", exc)
            result["pipeline_warning"] = str(exc)

        # 3) benchmarks (informational)
        if enable_benchmarks and not config.dry_run:
            bench_info = run_benchmarks(model_cfg, model, out_dir, logger)
            result.update(bench_info)

        result["success"] = True
        result["duration_s"] = round(time.time() - t0, 3)
        marker.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        _record_result_json(out_dir, result)
        logger.info("Completed %s", model)
        return result

    except ConfigError as exc:
        result["error"] = str(exc)
        result["pipeline_error"] = str(exc)
        logger.error("Configuration error for %s: %s", model, exc)
    except Exception as exc:  # noqa: BLE001 - report, never crash the batch
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["pipeline_error"] = str(exc)
        logger.error("Unexpected error for %s: %s", model, exc)
    result["success"] = False
    result["duration_s"] = round(time.time() - t0, 3)
    _record_result_json(out_dir, result)
    return result


# --------------------------------------------------------------------------- #
# Batch / multiprocessing
# --------------------------------------------------------------------------- #

def _spawn_worker(args: tuple) -> dict:
    """Top-level worker for multiprocessing (spawn-safe on Windows)."""
    config_dict, model, output_root, log_file, enable_benchmarks = args
    logger = make_logger(name=f"heretic-worker-{safe_name(model)}",
                         console=False, logfile=log_file,
                         level=logging.INFO)
    try:
        cfg = Config.from_dict(config_dict, logger=logger)
        cfg.output_root = output_root
    except Exception as exc:  # noqa: BLE001
        logger.error("worker could not rebuild config: %s", exc)
        return {"model": model, "success": False,
                "error": f"config rebuild failed: {exc}"}
    try:
        return process_single_model(cfg, model, logger,
                                    enable_benchmarks=enable_benchmarks)
    finally:
        for h in logger.handlers:
            h.close()


def run_batch(config: Config, models: list,
              logger: logging.Logger,
              enable_benchmarks: bool = True) -> list:
    workers = config.effective_workers()
    logger.info("Processing %d models with %d worker(s)", len(models), workers)
    log_dir = Path(config.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_log = log_dir / f"heretic_batch_{_now_tag()}.log"

    if workers == 1 or len(models) == 1:
        results = [
            process_single_model(config, m, logger,
                                 enable_benchmarks=enable_benchmarks)
            for m in models
        ]
        return results

    cfg_dict = config.to_dict()
    args_list = [
        (cfg_dict, m, config.output_root, str(batch_log), enable_benchmarks)
        for m in models
    ]
    ctx = mp.get_context("spawn")
    results: list = []
    try:
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_spawn_worker, args_list)
    except KeyboardInterrupt:
        logger.error("Interrupted; writing partial results")
        pool.terminate()
        raise
    return results


# --------------------------------------------------------------------------- #
# Summary / exit code
# --------------------------------------------------------------------------- #

def print_summary(results: list) -> int:
    failed = [r for r in results if not r.get("success")]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        tag = "OK" if r.get("success") else "FAIL"
        line = f"  [{tag}] {r['model']}"
        if not r.get("success"):
            err = r.get("error") or r.get("pipeline_error") or "unknown error"
            line += f"  -> {str(err).splitlines()[0][:160]}"
        elif r.get("resumed"):
            line += "  (resumed)"
        print(line)
    print("=" * 60)
    if failed:
        print(f"RESULT: {len(failed)}/{len(results)} model(s) FAILED")
        return 1
    print(f"RESULT: all {len(results)} model(s) OK")
    return 0


def dump_results_json(results: list, config: Config) -> None:
    Path(config.output_root).mkdir(parents=True, exist_ok=True)
    path = Path(config.output_root) / f"results_{_now_tag()}.json"
    payload = {
        "any_success": any(r.get("success") for r in results),
        "all_success": all(r.get("success") for r in results),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logging.getLogger("heretic_final").info("Results JSON: %s", path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="heretic_final.py",
        description="Heretic Enhanced v5 - config-driven wrapper for the real "
                    "heretic-llm 1.4.0 CLI.",
    )
    p.add_argument("-c", "--config", default="config.yaml",
                   help="YAML config path (default: config.yaml)")
    p.add_argument("--models", nargs="*", default=None,
                   help="override model list (space separated)")
    p.add_argument("--model", default=None,
                   help="override a single model id")
    p.add_argument("--output-dir", default=None,
                   help="override config.output_root")
    p.add_argument("--workers", type=int, default=None,
                   help="override hardware.num_workers")
    p.add_argument("--dry-run", action="store_true",
                   help="print the exact commands that would run, execute none")
    p.add_argument("--no-benchmarks", action="store_true",
                   help="skip benchmark stage")
    p.add_argument("--test-email", action="store_true",
                   help="send a test email using the configured notification "
                        "settings, then exit")
    p.add_argument("--write-config", metavar="PATH", default=None,
                   help="write a starter config to PATH and exit")
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)

    if args.write_config:
        target = Path(args.write_config)
        if target.exists() and target.stat().st_size > 0:
            logging.getLogger("heretic_final").error(
                "refusing to overwrite existing %s", target)
            return 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(EXAMPLE_CONFIG_YAML, encoding="utf-8")
        print(f"Wrote starter config to {target}")
        return 0

    logger = make_logger(level=logging.INFO)

    # build config (from file, then apply CLI overrides)
    cfg = load_config(args.config, logger)
    if args.output_dir:
        cfg.output_root = args.output_dir
    if args.workers is not None and args.workers > 0:
        cfg.hardware.num_workers = args.workers
    if args.dry_run:
        cfg.dry_run = True
    cfg.abliteration.quantization = str(cfg.abliteration.quantization or "none").lower()

    if args.test_email:
        ok = Notifier(cfg, logger=logger).send_test_email()
        return 0 if ok else 1

    # decide model list
    models: list = []
    if args.model:
        models.append(args.model)
    if args.models:
        models.extend(args.models)
    if not models:
        models = cfg.model_list()
    if not models:
        logger.error(
            "no models to process - set abliteration.model in %s or pass "
            "--model / --models", args.config,
        )
        return 1

    # final validation of settings the real CLI will reject
    if cfg.abliteration.quantization not in QUANTIZATION_VALUES:
        logger.error(
            "quantization '%s' is not a real heretic value (none|bnb_4bit); "
            "refusing to run", cfg.abliteration.quantization)
        return 1
    if not cfg.abliteration.export_strategy:
        logger.error("abliteration.export_strategy is empty; refusing to run")
        return 1

    logger.info("Heretic Enhanced v%s starting", __version__)
    logger.info("Config: %s", args.config)
    logger.info("Models: %s", ", ".join(models))
    logger.info("Output root: %s", cfg.output_root)
    if cfg.abliteration.device_map:
        logger.info("device_map: %s", cfg.abliteration.device_map)
    if cfg.unresolved_env:
        logger.warning("Unresolved env placeholders present: %s",
                       ", ".join(sorted(set(cfg.unresolved_env))))

    results = run_batch(cfg, models, logger,
                        enable_benchmarks=not args.no_benchmarks)

    dump_results_json(results, cfg)
    code = print_summary(results)

    try:
        notifier = Notifier(cfg, logger=logger)
        notifier.summarize(results, dry_run=cfg.dry_run)
    except Exception as exc:  # noqa: BLE001
        logger.error("Notification step failed: %s", exc)

    return code


if __name__ == "__main__":  # required for Windows multiprocessing (spawn)
    try:
        sys.exit(main())
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
