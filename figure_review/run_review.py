#!/usr/bin/env python3
"""
Figure Review Orchestrator
==========================

Runs a multi-agent review pipeline on all manuscript figures.

Usage:
    # Review all figures (full pipeline):
    python run_review.py

    # Review specific figures:
    python run_review.py manuscript/figures/figure1a*.png manuscript/figures/figure2*.png

    # Run only specific agents:
    python run_review.py --agents story,composition

    # Dry run (show what would be reviewed):
    python run_review.py --dry-run

    # Use a different model:
    python run_review.py --model claude-sonnet-4-6-20250514

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import argparse
import base64
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from agents import AGENTS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "manuscript", "figures")
DEFAULT_MODEL = "claude-sonnet-4-6"
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_IMAGE_BYTES = 3_500_000  # Stay under 5MB after base64 encoding (+33%)


def load_image_b64(path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for an image file. Downscale if >5MB."""
    media_type = "image/png"
    try:
        from PIL import Image
        import io
        with Image.open(path) as img:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            raw = buf.getvalue()
            # If under limit, use as-is
            if len(raw) <= MAX_IMAGE_BYTES:
                return base64.standard_b64encode(raw).decode("utf-8"), media_type
            # Downscale until under limit
            scale = 0.7
            while len(raw) > MAX_IMAGE_BYTES and scale > 0.1:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized = img.resize(new_size, Image.LANCZOS)
                buf = io.BytesIO()
                resized.save(buf, format='PNG')
                raw = buf.getvalue()
                scale *= 0.7
            return base64.standard_b64encode(raw).decode("utf-8"), media_type
    except ImportError:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, media_type


def image_metadata(path: str) -> dict:
    """Get file size and pixel dimensions."""
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    # Try to get dimensions via PIL if available
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
        return {"file": os.path.basename(path), "size_mb": round(size_mb, 2),
                "width_px": w, "height_px": h}
    except ImportError:
        return {"file": os.path.basename(path), "size_mb": round(size_mb, 2),
                "width_px": "unknown", "height_px": "unknown"}


def build_image_content(path: str) -> list[dict]:
    """Build Anthropic API content blocks for a single image."""
    b64, media = load_image_b64(path)
    meta = image_metadata(path)
    return [
        {"type": "text", "text": f"Figure: {meta['file']} "
                                 f"({meta['width_px']}x{meta['height_px']} px, "
                                 f"{meta['size_mb']} MB)"},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": media, "data": b64}},
    ]


def build_all_images_content(paths: list[str]) -> list[dict]:
    """Build content blocks for ALL images (for cross-figure agents)."""
    blocks = [{"type": "text",
               "text": f"You are reviewing {len(paths)} figures from the same manuscript. "
                       f"Evaluate each one AND their cross-figure consistency."}]
    for p in paths:
        blocks.extend(build_image_content(p))
    return blocks


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------
def run_agent(client: anthropic.Anthropic, agent: dict, fig_paths: list[str],
              model: str, prior_reports: dict | None = None) -> str:
    """Run a single agent and return its text output."""
    system = agent["system_prompt"]

    # Build user message content
    if agent["needs_all_figures"]:
        content = build_all_images_content(fig_paths)
    else:
        # For per-figure agents called one figure at a time
        content = build_image_content(fig_paths[0])

    # For the cross-figure orchestrator, append prior agent reports
    if agent["name"] == "consistency" and prior_reports:
        report_text = "\n\n".join(
            f"=== {name} Agent Report ===\n{text}"
            for name, text in prior_reports.items()
        )
        content.append({
            "type": "text",
            "text": f"Previous agent reports for context:\n\n{report_text}"
        })

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_pipeline(fig_paths: list[str], agent_names: list[str] | None = None,
                 model: str = DEFAULT_MODEL, dry_run: bool = False):
    """Run the full review pipeline."""

    # Filter agents if specific ones requested
    agents = AGENTS
    if agent_names:
        agents = [a for a in AGENTS if a["name"] in agent_names]
        if not agents:
            print(f"ERROR: No agents matched {agent_names}")
            print(f"Available: {[a['name'] for a in AGENTS]}")
            sys.exit(1)

    # Separate per-figure and cross-figure agents
    per_fig_agents = [a for a in agents if a["scope"] == "per_figure"
                      and not a["needs_all_figures"]]
    all_fig_agents = [a for a in agents if a["needs_all_figures"]
                      and a["name"] != "consistency"]
    orchestrator = [a for a in agents if a["name"] == "consistency"]

    print(f"\n{'='*60}")
    print(f"  FIGURE REVIEW PIPELINE")
    print(f"{'='*60}")
    print(f"  Figures: {len(fig_paths)}")
    print(f"  Agents:  {[a['name'] for a in agents]}")
    print(f"  Model:   {model}")
    print(f"{'='*60}\n")

    for p in fig_paths:
        meta = image_metadata(p)
        print(f"  {meta['file']:50s} {meta['width_px']}x{meta['height_px']} px, "
              f"{meta['size_mb']} MB")
    print()

    if dry_run:
        print("DRY RUN — no API calls made.")
        return

    client = anthropic.Anthropic()
    all_reports: dict[str, str] = {}  # agent_name -> combined report text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Phase 1: Per-figure agents (run each agent on each figure) ---
    for agent in per_fig_agents:
        print(f"\n--- Agent: {agent['title']} ---")
        agent_report_parts = []
        for path in fig_paths:
            fname = os.path.basename(path)
            print(f"  Reviewing {fname}...", end=" ", flush=True)
            t0 = time.time()
            result = run_agent(client, agent, [path], model)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            agent_report_parts.append(f"### {fname}\n\n{result}")
        all_reports[agent["name"]] = "\n\n".join(agent_report_parts)

    # --- Phase 2: All-figure agents (receive every figure at once) ---
    for agent in all_fig_agents:
        print(f"\n--- Agent: {agent['title']} (all figures) ---")
        print(f"  Sending {len(fig_paths)} figures...", end=" ", flush=True)
        t0 = time.time()
        result = run_agent(client, agent, fig_paths, model)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        all_reports[agent["name"]] = result

    # --- Phase 3: Cross-figure orchestrator ---
    for agent in orchestrator:
        print(f"\n--- Agent: {agent['title']} (final synthesis) ---")
        print(f"  Synthesizing {len(all_reports)} agent reports...", end=" ", flush=True)
        t0 = time.time()
        result = run_agent(client, agent, fig_paths, model,
                           prior_reports=all_reports)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        all_reports[agent["name"]] = result

    # --- Save reports ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"review_{timestamp}.md")

    with open(report_path, "w") as f:
        f.write(f"# Figure Review Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Figures: {len(fig_paths)}\n\n")

        for agent in agents:
            if agent["name"] in all_reports:
                f.write(f"---\n\n## {agent['title']}\n\n")
                f.write(all_reports[agent["name"]])
                f.write("\n\n")

    print(f"\n{'='*60}")
    print(f"  REPORT SAVED: {report_path}")
    print(f"{'='*60}\n")

    # Also print the orchestrator's final summary to stdout
    if "consistency" in all_reports:
        print("FINAL CONSISTENCY ASSESSMENT:")
        print("-" * 40)
        print(all_reports["consistency"])

    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent figure review pipeline")
    parser.add_argument("figures", nargs="*",
                        help="Figure files to review (default: all in manuscript/figures/)")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent names to run "
                             "(default: all). Options: story, composition, "
                             "color, typography, format, consistency")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be reviewed without making API calls")
    parser.add_argument("--main-only", action="store_true",
                        help="Only review main text figures (figure*), skip efigures")
    args = parser.parse_args()

    # Resolve figure paths
    if args.figures:
        fig_paths = []
        for pattern in args.figures:
            fig_paths.extend(glob.glob(pattern))
    else:
        fig_dir = os.path.abspath(DEFAULT_FIG_DIR)
        fig_paths = sorted(glob.glob(os.path.join(fig_dir, "*.png")))

    if args.main_only:
        fig_paths = [p for p in fig_paths
                     if os.path.basename(p).startswith("figure")]

    if not fig_paths:
        print("ERROR: No figure files found.")
        sys.exit(1)

    fig_paths = sorted(fig_paths)

    agent_names = None
    if args.agents:
        agent_names = [a.strip() for a in args.agents.split(",")]

    run_pipeline(fig_paths, agent_names, args.model, args.dry_run)


if __name__ == "__main__":
    main()
