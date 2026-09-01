#!/usr/bin/env python3
"""Maintain the download history of this repository's release assets.

Every number is computed from three sources and nothing else:

1. The git history of assets/downloads.svg: one committed cumulative total
   per badge refresh, used for the days before any per-asset record exists.
2. assets/download-observations.json: the raw per-release download counts
   seen by every run of the badge workflow. Old runs come from the Actions
   logs (which GitHub keeps for 90 days), new runs from the Releases API.
3. The Releases API, for the current run.

Commands:

  harvest-logs  Copy the per-tag counts of every retained workflow run log
                into the observations file. Skips runs already stored.
  update        Append the current API snapshot to the observations file,
                then rebuild assets/download-history.json.
  bootstrap     Rebuild assets/download-history.json without a new snapshot.
  render        Draw assets/download-history.svg and the assets/downloads.svg
                badge from the state file.

What counts as one download: one fetch of a release TARBALL. A release also
carries a SHA256SUMS, but fetching it is part of one download rather than
another one, and the shipped auto-updater fetches it on every up-to-date
launch without ever touching the tarball, so counting it would count
launches. The snapshot still records every asset, so the raw observations
stay complete and this choice can be revisited without re-collecting.

Rebuilding replays every observation in time order. An asset id is immutable
and its count never falls, because replacing a release asset creates a new id
starting at zero; so each id is measured against the highest count ever seen
for it, and an id missing from one snapshot and back in a later one resumes
from that baseline rather than being counted a second time. Between two
snapshots keyed by release tag (the harvested logs recorded per-tag totals
only), a tag whose count fell had some of its assets replaced; how many of
the remaining downloads are new is unknowable, so nothing is added and the
new count becomes the baseline. The total therefore never overstates
downloads and can only miss those made between a replacement and the next
observation.

The same script produces the local preview and the workflow output.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "assets" / "download-history.json"
OBSERVATIONS_PATH = ROOT / "assets" / "download-observations.json"
BADGE_PATH = ROOT / "assets" / "downloads.svg"
CHART_PATH = ROOT / "assets" / "download-history.svg"
FONT_DIR = ROOT / "assets" / "fonts"
WORKFLOW_FILE = "update-downloads-badge.yml"
BADGE_TITLE = re.compile(r"<title>downloads: (\d+)</title>")
LOG_TAGS = re.compile(r"^\S+ tags discovered: (.*)$", re.M)
LOG_COUNT = re.compile(r"^\S+   (\S+): (\d+)\s*$", re.M)
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------- sources


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def repo_slug() -> str:
    url = git("remote", "get-url", "origin").strip()
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot read a GitHub repository from {url!r}")
    return match.group(1)


def utc_time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_day(stamp: str) -> str:
    return utc_time(stamp).date().isoformat()


def first_commit_date() -> str:
    return utc_day(git("log", "--reverse", "--format=%cI").splitlines()[0])


def badge_history() -> list[tuple[str, int]]:
    """Every committed badge total with its commit time, oldest first."""

    rel = BADGE_PATH.relative_to(ROOT).as_posix()
    history = []
    for line in git("log", "--reverse", "--format=%H %cI", "--", rel).splitlines():
        sha, stamp = line.split()
        match = BADGE_TITLE.search(git("show", f"{sha}:{rel}"))
        if match:
            history.append((stamp, int(match.group(1))))
    if not history:
        raise SystemExit(f"no badge values found in the history of {rel}")
    return history


def api_request(url: str) -> urllib.request.Request:
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def api_json(url: str) -> object:
    with urllib.request.urlopen(api_request(url), timeout=30) as response:
        return json.load(response)


def api_pages(url: str, key: str | None = None) -> list:
    """Every page of a listing endpoint. The url must already carry per_page=100.

    key names the array inside the payload; omit it when the endpoint returns a
    bare array, as /releases and /tags do."""

    items = []
    page = 1
    while True:
        payload = api_json(f"{url}&page={page}")
        chunk = payload[key] if key else payload
        items.extend(chunk)
        if len(chunk) < 100:
            return items
        page += 1


def live_assets(repo: str) -> dict[str, dict]:
    """Every asset of every PUBLISHED release, keyed by its immutable asset id.

    A draft is skipped by name here rather than by accident: a draft has no git
    tag, so reading releases through the tag list used to hide drafts as a side
    effect, and anyone who can see a draft can download its assets. Those are
    not public downloads.

    Read the whole /releases listing, every page. Walking the tag list instead
    took one request per tag and, worse, silently dropped every release past
    the first page of tags."""

    api = f"https://api.github.com/repos/{repo}"
    assets: dict[str, dict] = {}
    for release in api_pages(f"{api}/releases?per_page=100"):
        if release["draft"]:
            continue
        for asset in release.get("assets", []):
            assets[str(asset["id"])] = {
                "tag": release["tag_name"],
                "name": asset["name"],
                "download_count": int(asset["download_count"]),
            }
    return assets


# ----------------------------------------------------------- observations


def load_observations() -> dict[str, dict]:
    if OBSERVATIONS_PATH.exists():
        return json.loads(OBSERVATIONS_PATH.read_text())
    return {}


def save_observations(observations: dict[str, dict]) -> None:
    OBSERVATIONS_PATH.write_text(
        json.dumps(observations, indent=1, sort_keys=True, separators=(",", ":")) + "\n"
    )


def parse_run_log(archive: bytes) -> dict[str, int] | None:
    """Per-tag download counts printed by one run of the badge workflow."""

    text = ""
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in bundle.namelist():
            if name.endswith(".txt") and not name.endswith("system.txt"):
                text += bundle.read(name).decode(errors="replace")
    tags = LOG_TAGS.search(text)
    if not tags:
        return None
    wanted = set(tags.group(1).split())
    counts = {tag: int(count) for tag, count in LOG_COUNT.findall(text) if tag in wanted}
    return counts or None


def harvest_logs(repo: str) -> None:
    observations = load_observations()
    api = f"https://api.github.com/repos/{repo}/actions"
    runs = api_pages(f"{api}/workflows/{WORKFLOW_FILE}/runs?per_page=100", "workflow_runs")
    added = expired = 0
    for run in sorted(runs, key=lambda run: run["created_at"]):
        key = f"run-{run['id']}"
        if key in observations:
            continue
        try:
            with urllib.request.urlopen(api_request(f"{api}/runs/{run['id']}/logs"), timeout=60) as response:
                counts = parse_run_log(response.read())
        except urllib.error.HTTPError as error:
            if error.code in (404, 410):
                expired += 1
                continue
            raise
        if counts:
            observations[key] = {"at": run["created_at"], "tags": counts}
            added += 1
    save_observations(observations)
    print(f"harvested {added} runs, {expired} runs have no log any more, {len(observations)} observations stored")


def record_snapshot(now: datetime, assets: dict[str, dict]) -> None:
    """Store the API snapshot unless nothing changed since the last one."""

    observations = load_observations()
    counts = {asset_id: asset["download_count"] for asset_id, asset in assets.items()}
    latest = max(observations.values(), key=lambda item: item["at"], default={})
    if latest.get("assets") == counts:
        print("no new downloads since the last snapshot")
        return
    key = "api-" + now.strftime("%Y%m%dT%H%M%SZ")
    observations[key] = {
        "at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "assets": counts,
        "releases": {asset_id: [asset["tag"], asset["name"]] for asset_id, asset in assets.items()},
    }
    save_observations(observations)


# ------------------------------------------------------------------ state


def counted_growth(previous: dict[str, int], current: dict[str, int]) -> int:
    """New downloads between two snapshots keyed the same way (tag or asset id).

    A key that grew contributes its growth and a new key its whole count. A
    key whose count fell (only possible per tag, when assets were replaced)
    contributes nothing. A key that vanished was already counted while it
    existed.
    """

    total = 0
    for key, count in current.items():
        before = previous.get(key)
        if before is None:
            total += count
        elif count > before:
            total += count - before
    return total


def absorb(seen: dict[str, int], current: dict[str, int]) -> int:
    """New downloads since the HIGHEST count ever recorded for each asset id.

    An asset id is immutable and its count never falls: replacing a release
    asset creates a new id starting at zero. So the highest count ever seen for
    an id is a safe baseline, and an id that drops out of one snapshot and
    returns in a later one resumes from that baseline instead of being counted
    from zero a second time. Comparing against only the PREVIOUS snapshot would
    add such an id's whole count again, and nothing would report it.

    A fall means the assumption broke; say so instead of hiding it."""

    total = 0
    for asset_id, count in current.items():
        before = seen.get(asset_id, 0)
        if count > before:
            total += count - before
            seen[asset_id] = count
        elif count < before:
            print(f"warning: asset {asset_id} fell from {before} to {count}", file=sys.stderr)
    return total


def tarballs(observation: dict) -> dict[str, int]:
    """Only the release tarballs. Fetching a release's SHA256SUMS is part of one
    download, not a second one, and the shipped auto-updater fetches it on every
    up-to-date check without ever touching the tarball, so counting it counts
    launches rather than downloads."""

    return {
        asset_id: count
        for asset_id, count in observation["assets"].items()
        if observation["releases"][asset_id][1].endswith(".tar.gz")
    }


def by_tag(observation: dict) -> dict[str, int]:
    if "tags" in observation:
        return observation["tags"]
    totals: dict[str, int] = {}
    for asset_id, count in tarballs(observation).items():
        tag = observation["releases"][asset_id][0]
        totals[tag] = totals.get(tag, 0) + count
    return totals


def rebuild_state(repo: str) -> dict:
    """Daily cumulative totals from the badge history and the observations."""

    daily: dict[str, int] = {}
    badges = badge_history()
    observations = sorted(load_observations().values(), key=lambda item: item["at"])
    start = observations[0]["at"] if observations else None

    for stamp, total in badges:
        if start and utc_time(stamp) >= utc_time(start):
            break
        daily[utc_day(stamp)] = total

    if observations:
        # The badge total committed nearest before the first observation seeds
        # the replay; the first observation itself is the baseline.
        cumulative = daily[max(daily)] if daily else sum(by_tag(observations[0]).values())
        previous = observations[0]
        seen: dict[str, int] = dict(tarballs(previous)) if "assets" in previous else {}
        daily[utc_day(previous["at"])] = cumulative
        for current in observations[1:]:
            if "assets" in previous and "assets" in current:
                cumulative += absorb(seen, tarballs(current))
            else:
                # One side is a harvested log, which recorded per-tag totals only.
                cumulative += counted_growth(by_tag(previous), by_tag(current))
                if "assets" in current:
                    seen.update(tarballs(current))
            daily[utc_day(current["at"])] = cumulative
            previous = current

    return {
        "repo": repo,
        "timezone": "UTC",
        "axis_start": {"date": first_commit_date(), "source": "first commit of repository"},
        "first_observation": {"date": min(daily), "source": "git history of assets/downloads.svg"},
        "daily_cumulative": daily,
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text())


def bootstrap(repo: str) -> None:
    state = rebuild_state(repo)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(f"{len(state['daily_cumulative'])} observed days, cumulative {state['daily_cumulative'][max(state['daily_cumulative'])]}")


def update(repo: str, now: datetime) -> None:
    record_snapshot(now, live_assets(repo))
    bootstrap(repo)


# ----------------------------------------------------------------- render


def font_face(name: str, weight: int) -> str:
    payload = base64.b64encode((FONT_DIR / f"ComicNeue-{name}.woff2").read_bytes()).decode()
    return (
        "@font-face{font-family:'Comic Neue';font-weight:"
        f"{weight};src:url(data:font/woff2;base64,{payload}) format('woff2')}}"
    )


def nice_step(span: int, target_ticks: int = 5) -> int:
    raw = max(span / target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return int(nice * magnitude)


def fmt_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def jittered_line(x1, y1, x2, y2, *, seed, wobble=1.8, segments=26) -> str:
    """Deterministic hand-drawn-looking line."""

    rng = random.Random(seed)
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if i not in (0, segments):
            x += rng.uniform(-wobble, wobble) * 0.35
            y += rng.uniform(-wobble, wobble)
        points.append((x, y))
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def smooth_path(points: list[tuple[float, float]]) -> str:
    """Bezier curve with horizontal tangents; it never overshoots a point."""

    chunks = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        dx = x2 - x1
        chunks.append(
            f"C {x1 + dx * 0.38:.2f} {y1:.2f} {x2 - dx * 0.38:.2f} {y2:.2f} {x2:.2f} {y2:.2f}"
        )
    return " ".join(chunks)


def daily_series(daily: dict[str, int]) -> list[tuple[date, int]]:
    """One value per day from the first to the last observation, forward-filled."""

    observed = {date.fromisoformat(day): value for day, value in daily.items()}
    first, last = min(observed), max(observed)
    series = []
    value = observed[first]
    day = first
    while day <= last:
        value = observed.get(day, value)
        series.append((day, value))
        day += timedelta(days=1)
    return series


def month_starts(first: date, last: date) -> list[date]:
    year, month = first.year, first.month
    if first.day != 1:
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    ticks = []
    while date(year, month, 1) <= last:
        ticks.append(date(year, month, 1))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return ticks


def render_chart(state: dict, width: int = 1456, height: int = 1024) -> str:
    repo = state["repo"]
    series = daily_series(state["daily_cumulative"])
    axis_start = date.fromisoformat(state["axis_start"]["date"])
    axis_end = series[-1][0]
    if axis_end <= axis_start:
        raise SystemExit("axis start must precede the last observation")

    left, right, top, bottom = 140, width - 92, 145, height - 145
    span_days = (axis_end - axis_start).days
    step = nice_step(series[-1][1])
    y_max = math.ceil(series[-1][1] / step) * step

    def px(day: date) -> float:
        return left + (day - axis_start).days / span_days * (right - left)

    def py(value: int) -> float:
        return bottom - value / y_max * (bottom - top)

    curve = smooth_path([(px(day), py(value)) for day, value in series])
    hand = "'Comic Neue', 'Comic Sans MS', cursive, sans-serif"
    ink = "#070707"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Download history for {escape(repo)}">',
        f"<style>{font_face('Regular', 400)}{font_face('Bold', 700)}</style>",
        f'<rect width="{width}" height="{height}" fill="#f7f7f5"/>',
        f'<text x="{width / 2:.1f}" y="86" text-anchor="middle" fill="#050505" '
        f'font-family="{hand}" font-size="34" font-weight="700">Download history</text>',
    ]

    axes = [
        (left, top, left, bottom, 11, 2.4, 4.4, 1),
        (left + 3, top + 1, left + 3, bottom, 12, 1.3, 1.7, 0.72),
        (left, bottom, right, bottom, 21, 2.2, 4.1, 1),
        (left, bottom + 3, right, bottom + 3, 22, 1.1, 1.6, 0.7),
    ]
    for x1, y1, x2, y2, seed, wobble, stroke, opacity in axes:
        out.append(
            f'<path d="{jittered_line(x1, y1, x2, y2, seed=seed, wobble=wobble)}" fill="none" '
            f'stroke="#050505" stroke-width="{stroke}" stroke-linecap="round" opacity="{opacity}"/>'
        )

    for value in range(step, y_max + 1, step):
        y = py(value)
        out.append(
            f'<text x="{left - 24}" y="{y + 9:.1f}" text-anchor="end" fill="{ink}" '
            f'font-family="{hand}" font-size="28">{fmt_count(value)}</text>'
        )

    ticks = month_starts(axis_start, axis_end)
    for tick in ticks:
        x = px(tick)
        label = MONTHS[tick.month - 1]
        if tick.month == 1 or tick == ticks[0]:
            label = f"{label} {tick.year}"
        out.append(
            f'<path d="M {x:.1f} {bottom} L {x:.1f} {bottom + 12}" stroke="#050505" '
            f'stroke-width="3" stroke-linecap="round"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{bottom + 47}" text-anchor="middle" fill="{ink}" '
            f'font-family="{hand}" font-size="27">{label}</text>'
        )

    out.append(
        f'<path d="{curve}" fill="none" stroke="#ef3517" stroke-width="5.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    out.append(
        f'<text x="{width / 2:.1f}" y="{height - 55}" text-anchor="middle" fill="{ink}" '
        f'font-family="{hand}" font-size="29">Date</text>'
    )
    mid = (top + bottom) / 2
    out.append(
        f'<text x="51" y="{mid:.1f}" text-anchor="middle" fill="{ink}" font-family="{hand}" '
        f'font-size="29" transform="rotate(-90 51 {mid:.1f})">Downloads</text>'
    )

    legend_w = min(420, max(280, 105 + len(repo) * 12.3))
    lx, ly, lh = left + 15, top + 7, 58
    frame = " ".join([
        jittered_line(lx + 8, ly, lx + legend_w - 10, ly + 2, seed=41, wobble=2.0, segments=16),
        jittered_line(lx + legend_w - 6, ly + 5, lx + legend_w, ly + lh - 6, seed=42, wobble=1.7, segments=8),
        jittered_line(lx + legend_w - 5, ly + lh, lx + 7, ly + lh - 2, seed=43, wobble=1.9, segments=16),
        jittered_line(lx + 2, ly + lh - 5, lx, ly + 6, seed=44, wobble=1.7, segments=8),
    ])
    out.extend([
        f'<path d="{frame}" fill="none" stroke="#090909" stroke-width="3.3" stroke-linecap="round"/>',
        f'<rect x="{lx + 18}" y="{ly + 20}" width="14" height="14" rx="2" fill="#ef3517"/>',
        f'<text x="{lx + 43}" y="{ly + 36}" fill="#111" font-family="{hand}" '
        f'font-size="26">{escape(repo)}</text>',
    ])

    bx, by = right - 230, bottom + 80
    out.extend([
        f'<g transform="translate({bx} {by})" stroke="#06d824" fill="none" '
        f'stroke-width="3.5" stroke-linecap="round">',
        '<path d="M 0 -12 L 0 12 M -11 -6 L 11 6 M 11 -6 L -11 6"/>',
        '<path d="M -8 -9 L 8 9 M 8 -9 L -8 9"/>',
        "</g>",
        f'<text x="{bx + 24}" y="{by + 8}" fill="#666d7a" font-family="{hand}" '
        f'font-size="24" font-weight="700">aibuildai.io</text>',
        "</svg>",
    ])
    return "\n".join(out) + "\n"


def render_badge(total: int) -> str:
    """A flat-square badge with the same geometry shields.io uses."""

    label, value = "downloads", str(total)
    label_w = 69
    value_w = 10 + 7 * len(value)
    width = label_w + value_w
    font = "Verdana,Geneva,DejaVu Sans,sans-serif"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" '
        f'aria-label="{label}: {value}"><title>{label}: {value}</title>'
        f'<g shape-rendering="crispEdges"><rect width="{label_w}" height="20" fill="#555"/>'
        f'<rect x="{label_w}" width="{value_w}" height="20" fill="#4b0"/></g>'
        f'<g fill="#fff" text-anchor="middle" font-family="{font}" '
        f'text-rendering="geometricPrecision" font-size="110">'
        f'<text x="{label_w * 5 + 5}" y="140" textLength="{label_w * 10 - 100}" '
        f'transform="scale(.1)">{label}</text>'
        f'<text x="{label_w * 10 + value_w * 5}" y="140" textLength="{value_w * 10 - 100}" '
        f'transform="scale(.1)">{value}</text></g></svg>\n'
    )


def render(png: Path | None) -> None:
    state = load_state()
    svg = render_chart(state)
    CHART_PATH.write_text(svg)
    total = state["daily_cumulative"][max(state["daily_cumulative"])]
    BADGE_PATH.write_text(render_badge(total))
    print(f"rendered {CHART_PATH.name} and {BADGE_PATH.name} at {total} downloads")
    if png:
        import cairosvg

        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png))
        print(f"wrote {png}")


# ------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("harvest-logs", "update", "bootstrap"):
        sub.add_parser(name)
    render_parser = sub.add_parser("render")
    render_parser.add_argument("--png", type=Path, help="also write a PNG preview")
    args = parser.parse_args()

    if args.command == "harvest-logs":
        harvest_logs(repo_slug())
    elif args.command == "update":
        update(repo_slug(), datetime.now(timezone.utc))
    elif args.command == "bootstrap":
        bootstrap(repo_slug())
    else:
        render(args.png)


if __name__ == "__main__":
    sys.exit(main())
