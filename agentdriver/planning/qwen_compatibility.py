"""Qwen3:14B compatibility helpers for AgentDriver motion planning."""
import ast
import re
import numpy as np


def get_planner_prompt(system_message, model_name):
    if model_name != "qwen3:14b":
        return system_message
    return (
        system_message
        + "\nReturn ONLY six FUTURE (x,y) waypoints at these exact times: "
        "0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s. "
        "Do NOT include the vehicle's current position at t=0s. "
        "Coordinates must be relative to the current ego position. "
        "Use this exact Python format: "
        "[(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6)]. "
        "Replace x1,y1,etc. with actual numerical coordinates. "
        "Do not include explanations, Markdown, or equations."
    )


def parse_qwen_trajectory(result):
    """Read six waypoints from Python lists or timestamped Markdown entries."""
    # Remove LaTeX spacing directives such as \  in coordinate pairs.
    result = re.sub(r"\\\s+", " ", result)
    def validate(coords):
        traj = np.asarray(coords, dtype=float)
        if traj.shape != (6, 2) or not np.isfinite(traj).all():
            raise ValueError("Invalid trajectory shape or coordinates")
        return traj

    # Preferred output: a Python list containing six (x, y) pairs.
    for match in re.finditer(r"\[[^\[\]]*\]", result, re.DOTALL):
        try:
            return validate(ast.literal_eval(match.group()))
        except (ValueError, SyntaxError, TypeError):
            continue

    # Fallback for timestamped numbered/bulleted Markdown.
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pair = re.compile(rf"\(\s*({number})\s*,\s*({number})\s*\)")
    # Fallback for six numbered coordinate entries without individual timestamps.
    # Accept only a contiguous 1..6 list with 0.5s and 1.0s timing context.
    numbered_line = re.compile(
        rf"^\s*([1-6])\.\s*\*\*\s*{pair.pattern}\s*\*\*\s*$"
    )
    numbered = []
    for line in result.splitlines():
        match = numbered_line.match(line)
        if match:
            numbered.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    has_timing_context = all(
        re.search(rf"(?<![\d.]){t}\s*s(?:econds)?\b", result)
        for t in ("0.5", "1.0")
    )
    if len(numbered) == 6 and [n for n, _, _ in numbered] == list(range(1, 7)) and has_timing_context:
        return validate([(x, y) for _, x, y in numbered])

    # Only recognise actual timestamped waypoint headings. In particular,
    # explanatory bullet points mentioning "t = 3.0s" are not waypoints.
    heading = re.compile(
        r"^\s*(?:[1-6]\.\s*|[-*]\s+)"
        r"\*{0,2}\s*(?:\$\s*)?(?:(?:t\s*=\s*)|(?:At\s+))?"
        r"(0\.5|1\.0|1\.5|2\.0|2\.5|3\.0)"
        r"\s*\$?\s*s(?:econds)?\b",
        re.I,
    )
    lines = result.splitlines()
    coords = []
    times = []

    for i, line in enumerate(lines):
        found_time = heading.match(line)
        if not found_time:
            continue

        # Prefer a coordinate on the timestamp line. Otherwise search the
        # whole numbered/bulleted waypoint section, not just two lines:
        # Qwen may insert several calculation lines before **Waypoint**.
        found_pair = pair.search(line)
        found_xy = None
        if not found_pair:
            section = []
            for following in lines[i + 1:]:
                if heading.match(following):
                    break
                section.append(following)

            # Explicit Waypoint/Position label takes precedence over any
            # interim calculations or other numerical values in the section.
            for following in section:
                if re.search(r"\b(?:Waypoint|Position)\b\s*\*{0,2}\s*:", following, re.I):
                    found_pair = pair.search(following)
                    if found_pair:
                        break

            if not found_pair:
                for following in section:
                    standalone = re.fullmatch(
                        rf"\s*\$?\s*(?:\(x\s*,\s*y\)\s*=\s*)?{pair.pattern}\s*\$?\s*",
                        following,
                    )
                    if standalone:
                        found_pair = pair.search(following)
                        break
                    x_match = re.search(rf"\bx\s*\\approx\s*({number})", following)
                    y_match = re.search(rf"\by\s*\\approx\s*({number})", following)
                    if x_match and y_match:
                        found_xy = (float(x_match.group(1)), float(y_match.group(1)))
                        break

        if not found_pair and found_xy is None:
            raise ValueError("Timestamped waypoint missing coordinates")
        times.append(float(found_time.group(1)))
        coords.append(
            (float(found_pair.group(1)), float(found_pair.group(2)))
            if found_pair else found_xy
        )

    if len(times) != 6 or not np.allclose(
        times, np.arange(0.5, 3.1, 0.5), rtol=0, atol=1e-6
    ):
        raise ValueError("Missing or incorrectly ordered waypoint timestamps")
    return validate(coords)
