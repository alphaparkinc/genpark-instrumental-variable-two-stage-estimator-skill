"""
MCP Server for Instrumental Variable Two-Stage Least Squares (2SLS) Estimator Skill
"""

import json
import sys
from client import TwoStageLeastSquaresIV

def handle_call(name: str, args: dict) -> dict:
    if name == "estimate_2sls":
        z = args.get("instrument_z", [1.0, 2.0, 3.0])
        x = args.get("treatment_x", [2.0, 4.0, 6.0])
        y = args.get("outcome_y", [6.0, 12.0, 18.0])
        est = TwoStageLeastSquaresIV()
        res = est.fit(z, x, y)
        return res
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
