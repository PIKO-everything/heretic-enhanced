import json
import logging
from pathlib import Path

from heretic import Config, _engine_argv, _record_result_json, build_heretic_command


def test_nested_notification_config_is_typed():
    cfg = Config.from_dict(
        {
            "abliteration": {"model": "org/model"},
            "notification": {
                "enable": True,
                "email": {"to_addrs": ["admin@example.com"]},
                "discord": {"url": "https://discord.example/hook"},
            },
        },
        logger=logging.getLogger("test"),
    )

    assert cfg.notification.email.to_addrs == ["admin@example.com"]
    assert cfg.notification.discord.url == "https://discord.example/hook"
    assert cfg.notify_channels_enabled()


def test_command_places_model_last():
    cfg = Config.from_dict(
        {
            "abliteration": {
                "model": "org/model",
                "n_trials": 3,
                "quantization": "none",
                "export_strategy": "merge",
                "device_map": "auto",
            }
        }
    )

    command = build_heretic_command(cfg, "org/model", logging.getLogger("test"))
    assert command[-1] == "org/model"
    assert "--device-map" in command


def test_windows_python_launcher_uses_active_interpreter():
    command = _engine_argv(r"C:\venv\Scripts\heretic.PY", platform="nt")

    assert command[0].endswith("python") or command[0].endswith("python.exe")
    assert command[1].lower().endswith("heretic.py")
    assert not command[1].startswith(".\\")


def test_result_writer_creates_dry_run_directory(tmp_path: Path):
    output_dir = tmp_path / "nested" / "model"
    _record_result_json(output_dir, {"success": True})

    result = json.loads((output_dir / "result.json").read_text())
    assert result == {"success": True}