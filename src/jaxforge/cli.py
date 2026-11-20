"""Command-line interface for Jax Forge."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence

import jaxforge
from jaxforge.agent import JaxForge
from jaxforge.codegen.validator import CodeValidator
from jaxforge.config import SUPPORTED_BACKENDS, SUPPORTED_PROVIDERS, JaxForgeConfig
from jaxforge.contrib.sharding import device_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jaxforge",
        description="Model-agnostic natural language JAX engineering workbench.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show version, supported providers, and execution backends.")
    sub.add_parser("devices", help="List available JAX compute devices as JSON.")

    val_p = sub.add_parser("validate", help="Run CodeValidator AST checks on a Python file.")
    val_p.add_argument("file", help="Path to Python source file.")
    val_p.add_argument("--intent", default="build", help="Target intent name (default: build).")

    prev_p = sub.add_parser("preview", help="Generate and preview JAX code without executing it.")
    prev_p.add_argument("prompt", help="Natural-language model or training instruction.")
    prev_p.add_argument("--model", default=None, help="Model or provider:model override.")
    prev_p.add_argument("--export", default=None, help="Optional directory to export code into.")

    run_p = sub.add_parser("run", help="Preview, approve, and run a natural-language prompt.")
    run_p.add_argument("prompt", help="Natural-language instruction.")
    run_p.add_argument("--model", default=None, help="Model or provider:model override.")
    run_p.add_argument("--yes", action="store_true", help="Skip interactive confirmation gate.")

    sub.add_parser("mcp", help="Start the Jax Forge MCP tool server over stdio.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "info":
        payload = {
            "version": jaxforge.__version__,
            "providers": sorted(SUPPORTED_PROVIDERS - {"auto"}),
            "execution_backends": sorted(SUPPORTED_BACKENDS),
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "devices":
        print(json.dumps(device_summary(), indent=2))
        return 0

    if args.command == "validate":
        src_path = pathlib.Path(args.file)
        code = src_path.read_text(encoding="utf-8")
        CodeValidator.validate(code, args.intent)
        print(f"Valid for intent '{args.intent}': {src_path}")
        return 0

    if args.command == "preview":
        cfg = JaxForgeConfig()
        if args.model:
            cfg.set({"model_name": args.model})
        agent = JaxForge(cfg)
        plan = agent.preview(args.prompt)
        print(f"# Intent: {plan.intent} ({plan.provider}:{plan.model_name})\n")
        print(plan.code)
        if args.export:
            from jaxforge.exporter import export_project

            export_project(args.export, model_code=plan.code)
            print(f"\nExported standalone project to: {args.export}")
        return 0

    if args.command == "run":
        cfg = JaxForgeConfig()
        if args.model:
            cfg.set({"model_name": args.model})
        agent = JaxForge(cfg)

        def _confirm(plan: jaxforge.GeneratedPlan) -> bool:
            if args.yes:
                return True
            print(f"\n--- Preview ({plan.intent}) ---\n{plan.code}\n")
            reply = input("Approve execution? [y/N]: ").strip().lower()
            return reply == "y"

        result = agent.run(args.prompt, approve=_confirm)
        print(repr(result))
        return 0

    if args.command == "mcp":
        from jaxforge.mcp_server import serve_stdio

        serve_stdio()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
