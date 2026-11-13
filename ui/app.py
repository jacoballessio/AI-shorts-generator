"""Gradio web UI for the XMLish example and video generation pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import importlib.util
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr


def _load_local_module(module_name: str, file_name: str):
    module_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register first so decorators/type resolution can find module globals during exec.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_auth_module = _load_local_module("ai_shorts_ui_auth", "auth.py")
_billing_module = _load_local_module("ai_shorts_ui_billing", "billing.py")

AuthConfig = _auth_module.AuthConfig
AuthService = _auth_module.AuthService
BillingConfig = _billing_module.BillingConfig
UsageLedger = _billing_module.UsageLedger


ROOT = Path(__file__).resolve().parent.parent


def _load_root_dotenv() -> None:
    """Load ROOT/.env key=value pairs into process env if not already set."""
    dotenv_path = ROOT / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_root_dotenv()

GENERATOR_SCRIPT = ROOT / "generate_example_with_carmind2_lora.py"
VIDEO_SCRIPT = ROOT / "generate_video_from_example.py"
WEB_OUTPUT_DIR = ROOT / "output" / "web_ui"
PYTHON_EXE = Path(sys.executable)
UI_GENERATOR_BASE_MODEL = os.environ.get("AI_SHORTS_UI_BASE_MODEL", "SuperAGI/mistral-7B-PoSE-32k")
_ui_adapter_override = os.environ.get(
    "AI_SHORTS_UI_ADAPTER",
    str(ROOT / "longcontext-20260324_025228" / "emergency_checkpoint"),
).strip()
UI_GENERATOR_ADAPTER = Path(_ui_adapter_override) if _ui_adapter_override else None
UI_FORCE_SPEECH = os.environ.get("AI_SHORTS_UI_FORCE_SPEECH", "1").strip().lower() not in {"0", "false", "no", "off", ""}
UI_FORCE_SPEECH_CLOUD = os.environ.get("AI_SHORTS_UI_FORCE_SPEECH_CLOUD", "0").strip().lower() not in {"0", "false", "no", "off", ""}
UI_CLOUD_MAX_NEW_TOKENS = int(os.environ.get("AI_SHORTS_UI_CLOUD_MAX_NEW_TOKENS", "768"))
UI_CLOUD_MAX_CONTINUATIONS = int(os.environ.get("AI_SHORTS_UI_CLOUD_MAX_CONTINUATIONS", "2"))
UI_CLOUD_TIMEOUT_S = int(os.environ.get("AI_SHORTS_UI_CLOUD_TIMEOUT_S", "120"))
UI_CLOUD_MAX_RETRIES = int(os.environ.get("AI_SHORTS_UI_CLOUD_MAX_RETRIES", "1"))
DEFAULT_CLOUD_ENDPOINT = os.environ.get(
    "AI_SHORTS_DEFAULT_CLOUD_ENDPOINT",
    "https://iv5moa1nbmfdaydv.eu-west-1.aws.endpoints.huggingface.cloud",
).strip()
DEFAULT_CLOUD_TOKEN = (
    os.environ.get("AI_SHORTS_CLOUD_ENDPOINT_TOKEN", "").strip()
    or os.environ.get("HF_TOKEN", "").strip()
    or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
)
UI_BUILD_ID = "2026-04-17-cloud-debug-4"

MODEL_SOURCE_CHOICES = ["Local", "Cloud Endpoint"]

TOPIC_CHOICES = [
    "Comedy",
    "Education",
    "Entertainment",
    "Lifestyle",
    "Technology",
    "Motivation",
    "Finance",
    "Health",
    "Relationships",
    "Storytelling",
]

BATCH_TOPIC_CHOICES = ["Auto"] + TOPIC_CHOICES


def _discover_local_model_choices() -> list[str]:
    choices = ["Auto (default local model)"]
    seen: set[str] = set()

    def _add(kind: str, path: Path) -> None:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        label = f"{kind} | {rel}"
        if label not in seen:
            seen.add(label)
            choices.append(label)

    for p in sorted(ROOT.glob("longcontext-*/merged_model")):
        if p.is_dir():
            _add("Merged", p)
    for p in sorted(ROOT.glob("longcontext-*/emergency_checkpoint")):
        if p.is_dir():
            _add("Adapter", p)

    carmind_base = ROOT / "models" / "carmind2_lora"
    if (carmind_base / "final").is_dir():
        _add("Adapter", carmind_base / "final")
    for p in sorted(carmind_base.glob("checkpoint-*")):
        if p.is_dir():
            _add("Adapter", p)

    return choices


def _discover_cloud_endpoint_choices() -> list[str]:
    endpoints: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        cleaned = (value or "").strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        endpoints.append(cleaned)

    _add(DEFAULT_CLOUD_ENDPOINT)
    raw_list = os.environ.get("AI_SHORTS_CLOUD_ENDPOINTS", "")
    for part in raw_list.replace(";", ",").split(","):
        _add(part)
    return endpoints or ([DEFAULT_CLOUD_ENDPOINT] if DEFAULT_CLOUD_ENDPOINT else [])


LOCAL_MODEL_CHOICES = _discover_local_model_choices()
CLOUD_ENDPOINT_CHOICES = _discover_cloud_endpoint_choices()


def _slugify(value: str, default: str = "video") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in (value or "").strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned.strip("-")
    return cleaned or default


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _run_command(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, output.strip()


def _join_log_lines(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def _run_command_stream(args: list[str]) -> Iterator[tuple[str, bool | None, str]]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        args,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    print(f"[UI] Spawned subprocess pid={proc.pid}: {_format_command_for_log(args)}", flush=True)

    lines: list[str] = []
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            print(f"[UI][subprocess:{proc.pid}] {line}", flush=True)
            lines.append(line)
            yield line, None, _join_log_lines(lines)
    finally:
        proc.stdout.close()

    return_code = proc.wait()
    print(f"[UI] Subprocess pid={proc.pid} exited with code {return_code}", flush=True)
    output = _join_log_lines(lines)
    yield "", return_code == 0, output


def _build_generator_args(
    *,
    output_path: Path,
    creator: str,
    topic: str | None,
    duration: int | float,
    title: str | None,
    model_source: str,
    local_model_choice: str,
    cloud_endpoint: str,
) -> list[str]:
    args = [
        str(PYTHON_EXE),
        "-u",
        str(GENERATOR_SCRIPT),
        "--base-model",
        UI_GENERATOR_BASE_MODEL,
        "--output",
        str(output_path),
        "--duration",
        str(int(duration)),
    ]

    model_source_normalized = (model_source or "").strip().lower()
    is_cloud = model_source_normalized == "cloud endpoint"
    selected_local = (local_model_choice or "").strip()

    selected_local_adapter: Path | None = None
    if selected_local and selected_local != "Auto (default local model)" and " | " in selected_local:
        kind, rel = selected_local.split(" | ", 1)
        selected_path = ROOT / rel
        if kind == "Merged":
            args.extend(["--base-model", str(selected_path), "--no-auto-adapter"])
        elif kind == "Adapter":
            selected_local_adapter = selected_path

    if is_cloud:
        endpoint = (cloud_endpoint or DEFAULT_CLOUD_ENDPOINT).strip()
        if endpoint:
            args.extend(["--cloud-endpoint-url", endpoint])
        if DEFAULT_CLOUD_TOKEN:
            args.extend(["--cloud-endpoint-token", DEFAULT_CLOUD_TOKEN])
        args.extend(["--max-new-tokens", str(max(64, UI_CLOUD_MAX_NEW_TOKENS))])
        args.extend(["--max-continuations", str(max(0, UI_CLOUD_MAX_CONTINUATIONS))])
        args.extend(["--cloud-timeout-s", str(max(15, UI_CLOUD_TIMEOUT_S))])
        args.extend(["--cloud-max-retries", str(max(0, UI_CLOUD_MAX_RETRIES))])
        args.append("--no-auto-adapter")
    else:
        if selected_local_adapter is not None:
            args.extend(["--adapter", str(selected_local_adapter)])
        elif UI_GENERATOR_ADAPTER is not None and selected_local == "Auto (default local model)":
            args.extend(["--adapter", str(UI_GENERATOR_ADAPTER)])

    apply_force_speech = UI_FORCE_SPEECH and (not is_cloud or UI_FORCE_SPEECH_CLOUD)
    if apply_force_speech:
        args.append("--force-speech")
    if creator.strip():
        args.extend(["--creator", creator.strip()])
    if topic and topic.strip():
        args.extend(["--topic", topic.strip()])
    if title and title.strip():
        args.extend(["--title", title.strip()])
    return args


def _build_video_args(json_path: Path, output_path: Path) -> list[str]:
    return [str(PYTHON_EXE), "-u", str(VIDEO_SCRIPT), str(json_path), "-o", str(output_path)]


def _format_command_for_log(args: list[str]) -> str:
    redacted: list[str] = []
    i = 0
    while i < len(args):
        part = args[i]
        redacted.append(part)
        if part == "--cloud-endpoint-token" and i + 1 < len(args):
            redacted.append("[REDACTED]")
            i += 2
            continue
        i += 1
    return " ".join(json.dumps(part) for part in redacted)


def _json_path_for_xml(xml_path: Path) -> Path:
    return xml_path.with_suffix(".json")


def _build_example_paths(prefix: str, title: str | None) -> tuple[Path, Path]:
    stem_title = title.strip() if title and title.strip() else "untitled"
    stem = f"{prefix}_{_slugify(stem_title)}_{_timestamp()}"
    xml_path = WEB_OUTPUT_DIR / f"{stem}.xmlish.txt"
    json_path = _json_path_for_xml(xml_path)
    return xml_path, json_path


def _example_progress_update(log_line: str) -> tuple[float, str] | None:
    line = log_line.lower()
    if "using local merged model" in line or "using base model" in line:
        return 0.12, "Loading example model..."
    if "loading base model from snapshot" in line or "retrying with cleaned local snapshot" in line:
        return 0.18, "Preparing model weights..."
    if "generating title with llm" in line:
        return 0.4, "Generating title..."
    if "generated title:" in line:
        return 0.48, "Generating example..."
    if "generation appears incomplete" in line:
        return 0.52, "Continuing example generation..."
    if "[ok] generated xml-ish example saved to:" in line:
        return 0.58, "Example XML generated"
    if "[ok] structured json saved to:" in line:
        return 0.6, "Example JSON generated"
    return None


def _video_progress_update(log_line: str) -> tuple[float, str] | None:
    line = log_line.lower()
    if "[video] generating video from:" in line:
        return 0.64, "Preparing video render..."
    if "segment " in line:
        return 0.72, "Building video segments..."
    if "[search] searching pexels" in line or "loading stable diffusion model" in line:
        return 0.78, "Generating visuals..."
    if "generating elevenlabs tts" in line or "[ok] generated tts audio" in line:
        return 0.84, "Generating narration..."
    if "[sfx] generating sound effect" in line or "[ok] sfx generated" in line:
        return 0.88, "Generating sound effects..."
    if "rendering final video to" in line:
        return 0.94, "Rendering final video..."
    if "video saved:" in line:
        return 1.0, "Complete"
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _summary_row(item: dict[str, Any]) -> list[str]:
    return [
        item["label"],
        item.get("title", ""),
        item.get("creator", ""),
        item.get("topic", ""),
        str(item.get("duration", "")),
    ]


class PipelineWebUI:
    def __init__(self) -> None:
        WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.billing_config = BillingConfig.from_env()
        self.ledger = UsageLedger(ROOT / "output" / "web_ui" / "billing.sqlite3", config=self.billing_config)
        self.auth = AuthService(ROOT / "output" / "web_ui" / "auth.sqlite3", config=AuthConfig.from_env())

    def _require_authenticated_user(self, session_token: str) -> str:
        user = self.auth.get_user_for_session(session_token)
        if not user:
            raise ValueError("Please sign in first.")
        return user

    def sign_up(self, username: str, password: str) -> tuple[str, str, str, str, str]:
        try:
            created_user = self.auth.sign_up(username, password)
            self.ledger.ensure_user(created_user)
            user, token, expires = self.auth.sign_in(username, password)
            status = f"Signed up and signed in as {user}. Session expires: {expires}"
            billing = self._pricing_summary(user)
            return status, token, user, f"Signed in as: {user}", billing
        except Exception as exc:
            return f"Sign up failed: {exc}", "", "", "Not signed in", self._pricing_summary("")

    def sign_in(self, username: str, password: str) -> tuple[str, str, str, str, str]:
        try:
            user, token, expires = self.auth.sign_in(username, password)
            self.ledger.ensure_user(user)
            status = f"Signed in as {user}. Session expires: {expires}"
            billing = self._pricing_summary(user)
            return status, token, user, f"Signed in as: {user}", billing
        except Exception as exc:
            return f"Sign in failed: {exc}", "", "", "Not signed in", self._pricing_summary("")

    def sign_out(self, session_token: str) -> tuple[str, str, str, str, str]:
        self.auth.sign_out(session_token)
        return "Signed out.", "", "", "Not signed in", self._pricing_summary("")

    def _resolve_cloud_endpoint(self, dropdown_value: str | None, custom_value: str | None) -> str:
        custom = (custom_value or "").strip()
        if custom:
            return custom
        return (dropdown_value or DEFAULT_CLOUD_ENDPOINT).strip()

    def _pricing_summary(self, user_id: str) -> str:
        per_request = self.billing_config.effective_per_request_usd()
        target_rph = self.billing_config.target_requests_per_hour
        balance = self.ledger.get_balance((user_id or "").strip()) if user_id else 0.0
        return (
            f"Cloud billing: ${per_request:.4f} per request "
            f"(derived from ${self.billing_config.cloud_hourly_usd:.2f}/hr at {target_rph:.1f} req/hr). "
            f"Balance: ${balance:.4f}"
        )

    def refresh_billing_status(self, session_token: str) -> str:
        user = self.auth.get_user_for_session(session_token)
        if not user:
            return self._pricing_summary("")
        self.ledger.ensure_user(user)
        return self._pricing_summary(user)

    def add_test_credit(self, session_token: str, amount_usd: float) -> str:
        user = self.auth.get_user_for_session(session_token)
        if not user:
            return "Please sign in first."
        if amount_usd <= 0:
            return "Credit amount must be > 0."
        new_balance = self.ledger.credit_user(user, float(amount_usd))
        return f"Credited ${amount_usd:.2f} to {user}. New balance: ${new_balance:.4f}"

    def _charge_cloud_request(self, session_token: str, request_count: int, source_label: str) -> tuple[bool, str]:
        user = self.auth.get_user_for_session(session_token)
        if not user:
            return False, "Cloud generation requires sign in."

        self.ledger.ensure_user(user)
        estimate = self.ledger.estimate_cloud_cost(request_count=request_count)
        if not self.ledger.can_afford(user, estimate):
            balance = self.ledger.get_balance(user)
            return (
                False,
                f"Insufficient balance for cloud request. Need ${estimate:.4f}, available ${balance:.4f}.",
            )

        charged, remaining = self.ledger.charge_cloud_usage(
            user_id=user,
            source=source_label,
            request_count=request_count,
            metadata={"kind": source_label},
        )
        return True, f"[Billing] Charged ${charged:.4f}. Remaining balance: ${remaining:.4f}"

    def _generate_example(
        self,
        *,
        title: str | None,
        creator: str,
        duration: int | float,
        topic: str | None,
        prefix: str,
        model_source: str,
        local_model_choice: str,
        cloud_endpoint: str,
    ) -> tuple[Path | None, Path | None, str]:
        xml_path, json_path = _build_example_paths(prefix, title)

        ok, output = _run_command(
            _build_generator_args(
                output_path=xml_path,
                creator=creator,
                topic=topic,
                duration=duration,
                title=title,
                model_source=model_source,
                local_model_choice=local_model_choice,
                cloud_endpoint=cloud_endpoint,
            )
        )
        if not ok:
            return None, None, output
        if not json_path.exists():
            return xml_path, None, output + "\n[ERROR] Example JSON was not created."
        return xml_path, json_path, output

    def generate_single(
        self,
        title: str,
        creator: str,
        duration: int,
        topic: str,
        session_token: str,
        model_source: str,
        local_model_choice: str,
        cloud_endpoint_choice: str,
        cloud_endpoint_custom: str,
        progress=gr.Progress(),
    ) -> Iterator[tuple[str | None, dict[str, Any] | None, str]]:
        print(
            f"[UI] generate_single called (build={UI_BUILD_ID}) model_source={model_source!r} title={title!r}",
            flush=True,
        )
        xml_path, json_path = _build_example_paths("single", title)
        video_path = WEB_OUTPUT_DIR / f"{json_path.stem}.mp4"
        cloud_endpoint = self._resolve_cloud_endpoint(cloud_endpoint_choice, cloud_endpoint_custom)

        log_lines = [
            f"Example: {xml_path}",
            f"JSON: {json_path}",
            f"Video: {video_path}",
            f"Model source: {model_source}",
            f"Cloud endpoint: {cloud_endpoint if model_source == 'Cloud Endpoint' else '(not used)'}",
            f"UI build: {UI_BUILD_ID}",
            "",
            "[UI] Starting example generation...",
        ]

        if model_source == "Cloud Endpoint":
            print("[UI] Cloud mode selected; checking billing before generation", flush=True)
            billed_ok, billed_msg = self._charge_cloud_request(session_token=session_token, request_count=1, source_label="single_script")
            if not billed_ok:
                print(f"[UI] Billing gate blocked generation: {billed_msg}", flush=True)
                progress(1.0, desc="Billing required")
                yield None, None, _join_log_lines(log_lines + [billed_msg])
                return
            print(f"[UI] Billing gate passed: {billed_msg}", flush=True)
            log_lines.append(billed_msg)
        else:
            try:
                user = self._require_authenticated_user(session_token)
                print(f"[UI] Local mode authorized user={user}", flush=True)
                log_lines.append(f"[Auth] Using account: {user}")
            except Exception as exc:
                print(f"[UI] Auth gate blocked generation: {exc}", flush=True)
                progress(1.0, desc="Sign in required")
                yield None, None, _join_log_lines(log_lines + [f"[Auth] {exc}"])
                return

        progress(0.02, desc="Starting example generation...")
        yield None, None, _join_log_lines(log_lines)

        generator_args = _build_generator_args(
            output_path=xml_path,
            creator=creator,
            topic=topic,
            duration=duration,
            title=title,
            model_source=model_source,
            local_model_choice=local_model_choice,
            cloud_endpoint=cloud_endpoint,
        )
        log_lines.append(f"[UI] Generator command: {_format_command_for_log(generator_args)}")
        print(f"[UI] Launching generator command: {_format_command_for_log(generator_args)}", flush=True)
        yield None, None, _join_log_lines(log_lines)

        gen_ok = False
        for line, done, combined_output in _run_command_stream(generator_args):
            if line:
                update = _example_progress_update(line)
                if update is not None:
                    progress(*update)
                yield None, None, _join_log_lines(log_lines[:4] + [combined_output])
            if done is not None:
                gen_ok = done
                gen_log = combined_output
                break
        else:
            gen_log = ""

        if not gen_ok:
            progress(1.0, desc="Example generation failed")
            yield None, None, _join_log_lines(log_lines[:4] + [gen_log])
            return

        if not json_path.exists():
            progress(1.0, desc="Example generation failed")
            error_log = _join_log_lines([gen_log, "[ERROR] Example JSON was not created."])
            yield None, None, _join_log_lines(log_lines[:4] + [error_log])
            return

        payload = _load_json(json_path)
        progress(0.62, desc="Starting video generation...")
        yield None, payload, _join_log_lines(log_lines[:4] + [gen_log, "", "[UI] Starting video generation..."])

        video_ok = False
        for line, done, combined_output in _run_command_stream(_build_video_args(json_path, video_path)):
            if line:
                update = _video_progress_update(line)
                if update is not None:
                    progress(*update)
                yield None, payload, _join_log_lines(log_lines[:4] + [gen_log, "", combined_output])
            if done is not None:
                video_ok = done
                video_log = combined_output
                break
        else:
            video_log = ""

        status = _join_log_lines(log_lines[:4] + [gen_log, "", video_log])
        if not video_ok:
            progress(1.0, desc="Video generation failed")
            yield None, payload, status
            return

        progress(1.0, desc="Complete")
        yield str(video_path), payload, status

    def generate_batch_scripts(
        self,
        batch_count: int,
        creator: str,
        topic: str,
        duration: int,
        session_token: str,
        model_source: str,
        local_model_choice: str,
        cloud_endpoint_choice: str,
        cloud_endpoint_custom: str,
        progress=gr.Progress(),
    ) -> tuple[list[list[str]], Any, Any, dict[str, Any], str, Any]:
        items: list[dict[str, Any]] = []
        logs: list[str] = []
        selected_topic = None if topic == "Auto" else topic
        cloud_endpoint = self._resolve_cloud_endpoint(cloud_endpoint_choice, cloud_endpoint_custom)

        try:
            user = self._require_authenticated_user(session_token)
            logs.append(f"[Auth] Using account: {user}")
        except Exception as exc:
            return (
                [],
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=None),
                {},
                f"[Auth] {exc}",
                None,
            )

        for idx in range(batch_count):
            progress((idx + 1) / max(batch_count, 1), desc=f"Generating script {idx + 1}/{batch_count}...")

            if model_source == "Cloud Endpoint":
                billed_ok, billed_msg = self._charge_cloud_request(
                    session_token=session_token,
                    request_count=1,
                    source_label="batch_script",
                )
                if not billed_ok:
                    logs.append(f"[{idx + 1}]\n{billed_msg}")
                    break
                logs.append(f"[{idx + 1}]\n{billed_msg}")

            xml_path, json_path, gen_log = self._generate_example(
                title=None,
                creator=creator,
                duration=duration,
                topic=selected_topic,
                prefix=f"batch_{idx + 1}",
                model_source=model_source,
                local_model_choice=local_model_choice,
                cloud_endpoint=cloud_endpoint,
            )
            logs.append(f"[{idx + 1}]\n{gen_log}")
            if json_path is None:
                continue

            payload = _load_json(json_path)
            input_block = payload.get("input", {}) if isinstance(payload, dict) else {}
            title_value = str(input_block.get("title", "") or "")
            item = {
                "id": idx + 1,
                "label": f"[{idx + 1}] {title_value or json_path.stem}",
                "title": title_value,
                "creator": str(input_block.get("creator", "") or ""),
                "topic": str(input_block.get("topic", "") or ""),
                "duration": input_block.get("duration", duration),
                "xml_path": str(xml_path) if xml_path else "",
                "json_path": str(json_path),
                "video_path": str(WEB_OUTPUT_DIR / f"{json_path.stem}.mp4"),
                "payload": payload,
            }
            items.append(item)

        table = [_summary_row(item) for item in items]
        labels = [item["label"] for item in items]
        state = {item["label"]: item for item in items}
        preview_payload = items[0]["payload"] if items else None
        status = "\n\n".join(logs).strip() if logs else "No scripts were generated."
        return (
            table,
            gr.update(choices=labels, value=[]),
            gr.update(choices=labels, value=labels[0] if labels else None),
            state,
            status,
            preview_payload,
        )

    def preview_batch_script(self, selected_label: str, batch_state: dict[str, Any]) -> dict[str, Any] | None:
        if not selected_label or not batch_state:
            return None
        item = batch_state.get(selected_label)
        if not item:
            return None
        return item.get("payload")

    def create_selected_videos(
        self,
        selected_labels: list[str],
        batch_state: dict[str, Any],
        progress=gr.Progress(),
    ) -> tuple[list[str], str]:
        if not selected_labels:
            return [], "No scripts selected."
        if not batch_state:
            return [], "No batch scripts available."

        created: list[str] = []
        logs: list[str] = []
        total = len(selected_labels)
        for idx, label in enumerate(selected_labels, start=1):
            progress(idx / total, desc=f"Generating video {idx}/{total}...")
            item = batch_state.get(label)
            if not item:
                logs.append(f"{label}: missing batch state entry")
                continue
            json_path = item.get("json_path")
            video_path = item.get("video_path")
            ok, output = _run_command(
                _build_video_args(Path(json_path), Path(video_path))
            )
            logs.append(f"{label}\n{output}")
            if ok:
                created.append(str(video_path))

        return created, "\n\n".join(logs).strip()

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="AI Shorts Web Generator") as app:
            gr.Markdown(
                """
                # AI Shorts Web Generator
                Generate an XMLish example, turn it into a video, and review batch-generated scripts before video creation.
                """
            )

            batch_state = gr.State({})
            session_token_state = gr.State("")
            session_user_state = gr.State("")

            with gr.Row():
                with gr.Column(scale=2):
                    auth_username = gr.Textbox(label="Username", placeholder="Choose username")
                    auth_password = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                    with gr.Row():
                        sign_up_btn = gr.Button("Sign Up", variant="secondary")
                        sign_in_btn = gr.Button("Sign In", variant="primary")
                        sign_out_btn = gr.Button("Sign Out")
                with gr.Column(scale=2):
                    model_source = gr.Dropdown(label="Model Source", choices=MODEL_SOURCE_CHOICES, value="Local")
                with gr.Column(scale=2):
                    local_model_choice = gr.Dropdown(
                        label="Local Model",
                        choices=LOCAL_MODEL_CHOICES,
                        value=LOCAL_MODEL_CHOICES[0] if LOCAL_MODEL_CHOICES else "Auto (default local model)",
                    )
                with gr.Column(scale=2):
                    cloud_endpoint_choice = gr.Dropdown(
                        label="Cloud Endpoint",
                        choices=CLOUD_ENDPOINT_CHOICES,
                        value=(CLOUD_ENDPOINT_CHOICES[0] if CLOUD_ENDPOINT_CHOICES else DEFAULT_CLOUD_ENDPOINT),
                    )
                    cloud_endpoint_custom = gr.Textbox(
                        label="Cloud Endpoint Override",
                        placeholder="Optional custom endpoint URL",
                        value="",
                    )

            with gr.Row():
                auth_status = gr.Textbox(label="Account", lines=2, value="Not signed in")
                signed_in_as = gr.Textbox(label="Session", lines=1, value="Not signed in")

            with gr.Row():
                billing_status = gr.Textbox(label="Billing Status", lines=2, value=self._pricing_summary(""))
            with gr.Row():
                add_credit_amount = gr.Slider(label="Add Test Credit (USD)", minimum=1, maximum=100, step=1, value=10)
                refresh_billing_btn = gr.Button("Refresh Billing")
                add_credit_btn = gr.Button("Add Test Credit")

            with gr.Tabs():
                with gr.Tab("Single Video"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            single_title = gr.Textbox(label="Title", placeholder="Leave blank to let the LLM generate it")
                            single_creator = gr.Textbox(label="Creator", placeholder="Creator/style name")
                            single_duration = gr.Slider(label="Duration", minimum=5, maximum=90, step=1, value=30)
                            single_topic = gr.Dropdown(label="Topic", choices=TOPIC_CHOICES, value=TOPIC_CHOICES[0])
                            single_generate = gr.Button("Generate Video", variant="primary")
                        with gr.Column(scale=1):
                            single_video = gr.Video(label="Generated Video")
                            single_json = gr.JSON(label="Generated Example JSON")
                            single_status = gr.Textbox(label="Logs", lines=16, max_lines=30)

                    single_generate.click(
                        fn=self.generate_single,
                        inputs=[
                            single_title,
                            single_creator,
                            single_duration,
                            single_topic,
                            session_token_state,
                            model_source,
                            local_model_choice,
                            cloud_endpoint_choice,
                            cloud_endpoint_custom,
                        ],
                        outputs=[single_video, single_json, single_status],
                    )

                with gr.Tab("Batch Mode"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_count = gr.Slider(label="Number of Scripts", minimum=1, maximum=20, step=1, value=5)
                            batch_creator = gr.Textbox(label="Creator", placeholder="Optional creator/style")
                            batch_duration = gr.Slider(label="Duration", minimum=5, maximum=90, step=1, value=30)
                            batch_topic = gr.Dropdown(label="Topic", choices=BATCH_TOPIC_CHOICES, value="Auto")
                            batch_generate = gr.Button("Generate Batch Scripts", variant="primary")
                        with gr.Column(scale=2):
                            batch_table = gr.Dataframe(
                                headers=["Select Label", "Title", "Creator", "Topic", "Duration"],
                                datatype=["str", "str", "str", "str", "str"],
                                row_count=0,
                                col_count=(5, "fixed"),
                                label="Generated Scripts",
                            )
                            batch_status = gr.Textbox(label="Batch Logs", lines=14, max_lines=24)

                    with gr.Row():
                        batch_select = gr.CheckboxGroup(label="Select scripts to render into videos", choices=[])
                    with gr.Row():
                        batch_preview_choice = gr.Dropdown(label="Preview script", choices=[], value=None)
                        batch_render = gr.Button("Create Videos From Selected Scripts")
                    with gr.Row():
                        batch_preview = gr.JSON(label="Selected Script JSON")
                        batch_videos = gr.Files(label="Generated Videos")
                    batch_video_logs = gr.Textbox(label="Video Generation Logs", lines=14, max_lines=24)

                    batch_generate.click(
                        fn=self.generate_batch_scripts,
                        inputs=[
                            batch_count,
                            batch_creator,
                            batch_topic,
                            batch_duration,
                            session_token_state,
                            model_source,
                            local_model_choice,
                            cloud_endpoint_choice,
                            cloud_endpoint_custom,
                        ],
                        outputs=[batch_table, batch_select, batch_preview_choice, batch_state, batch_status, batch_preview],
                    )
                    batch_preview_choice.change(
                        fn=self.preview_batch_script,
                        inputs=[batch_preview_choice, batch_state],
                        outputs=[batch_preview],
                    )
                    batch_render.click(
                        fn=self.create_selected_videos,
                        inputs=[batch_select, batch_state],
                        outputs=[batch_videos, batch_video_logs],
                    )

            refresh_billing_btn.click(
                fn=self.refresh_billing_status,
                inputs=[session_token_state],
                outputs=[billing_status],
            )
            add_credit_btn.click(
                fn=self.add_test_credit,
                inputs=[session_token_state, add_credit_amount],
                outputs=[billing_status],
            )

            sign_up_btn.click(
                fn=self.sign_up,
                inputs=[auth_username, auth_password],
                outputs=[auth_status, session_token_state, session_user_state, signed_in_as, billing_status],
            )
            sign_in_btn.click(
                fn=self.sign_in,
                inputs=[auth_username, auth_password],
                outputs=[auth_status, session_token_state, session_user_state, signed_in_as, billing_status],
            )
            sign_out_btn.click(
                fn=self.sign_out,
                inputs=[session_token_state],
                outputs=[auth_status, session_token_state, session_user_state, signed_in_as, billing_status],
            )

        return app


def main() -> None:
    ui = PipelineWebUI()
    app = ui.create_interface()
    app.queue().launch(server_name="0.0.0.0", server_port=7862, share=False)


if __name__ == "__main__":
    main()
