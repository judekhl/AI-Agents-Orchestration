"""
economic_analysis.py — API vs on-premises cost comparison.

Reads experiment configuration from .env / config file and computes:
  - API cost per request and per month at various request volumes
  - On-prem monthly cost (hardware amortization + electricity + operator time)
  - Break-even request volume where on-prem becomes cheaper
  - Saves results to results/processed/economic_analysis.json

All numbers are derived from .env assumptions — see .env.example for sources.

Usage:
    python src/economic_analysis.py \
        --output results/processed/economic_analysis.json
"""

import argparse
import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_assumptions() -> dict:
    return {
        "hardware_cost_usd": float(os.getenv("HARDWARE_COST_USD", "800")),
        "amortization_years": float(os.getenv("AMORTIZATION_YEARS", "3")),
        "electricity_rate_usd_per_kwh": float(os.getenv("ELECTRICITY_RATE_USD_PER_KWH", "0.12")),
        "cpu_tdp_watts": float(os.getenv("CPU_TDP_WATTS", "45")),
        "api_input_price_per_1m": float(os.getenv("API_INPUT_PRICE_PER_1M_TOKENS", "3.00")),
        "api_output_price_per_1m": float(os.getenv("API_OUTPUT_PRICE_PER_1M_TOKENS", "15.00")),
        "api_pricing_date": os.getenv("API_PRICING_DATE", "UNKNOWN — set API_PRICING_DATE in .env"),
        "api_pricing_model": os.getenv("API_PRICING_MODEL", "UNKNOWN"),
        # Operator time: hours per month spent on maintenance
        "operator_hours_per_month": float(os.getenv("OPERATOR_HOURS_PER_MONTH", "2")),
        "operator_hourly_rate_usd": float(os.getenv("OPERATOR_HOURLY_RATE_USD", "30")),
    }


def compute_on_prem_monthly(a: dict, inference_hours_per_month: float = 50.0) -> dict:
    """
    Compute monthly on-prem cost.
    inference_hours_per_month: estimated hours the machine runs inference per month.
    """
    hw_monthly = a["hardware_cost_usd"] / (a["amortization_years"] * 12)
    electricity_monthly = (
        (a["cpu_tdp_watts"] / 1000)  # kW
        * inference_hours_per_month
        * a["electricity_rate_usd_per_kwh"]
    )
    operator_monthly = a["operator_hours_per_month"] * a["operator_hourly_rate_usd"]
    total = hw_monthly + electricity_monthly + operator_monthly

    return {
        "hardware_amortization_usd_month": round(hw_monthly, 4),
        "electricity_usd_month": round(electricity_monthly, 4),
        "operator_usd_month": round(operator_monthly, 4),
        "total_fixed_usd_month": round(total, 4),
        "inference_hours_per_month_assumed": inference_hours_per_month,
    }


def compute_api_cost_per_request(a: dict, input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * a["api_input_price_per_1m"]
    output_cost = (output_tokens / 1_000_000) * a["api_output_price_per_1m"]
    return round(input_cost + output_cost, 8)


def compute_breakeven(on_prem_fixed_monthly: float, api_cost_per_request: float) -> float:
    """Break-even requests/month: on_prem_fixed / api_cost_per_request."""
    if api_cost_per_request <= 0:
        return float("inf")
    return round(on_prem_fixed_monthly / api_cost_per_request, 1)


def build_monthly_cost_curve(a: dict, on_prem_monthly: dict,
                              api_cost_per_request: float,
                              request_volumes: list) -> list:
    rows = []
    fixed = on_prem_monthly["total_fixed_usd_month"]
    for n in request_volumes:
        api_total = round(n * api_cost_per_request, 4)
        on_prem_total = round(fixed, 4)
        rows.append({
            "requests_per_month": n,
            "api_cost_usd": api_total,
            "on_prem_cost_usd": on_prem_total,
            "cheaper": "on_prem" if on_prem_total < api_total else "api",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute API vs on-prem economic analysis.")
    parser.add_argument("--output", default="results/processed/economic_analysis.json")
    parser.add_argument("--input-tokens", type=int, default=200,
                        help="Input tokens per request (from your benchmark prompt)")
    parser.add_argument("--output-tokens", type=int, default=200,
                        help="Output tokens per request (from your benchmark max_new_tokens)")
    parser.add_argument("--inference-hours-month", type=float, default=50.0,
                        help="Estimated inference hours per month for electricity calc")
    args = parser.parse_args()

    assumptions = load_assumptions()

    on_prem = compute_on_prem_monthly(assumptions, args.inference_hours_month)
    api_cost_per_req = compute_api_cost_per_request(
        assumptions, args.input_tokens, args.output_tokens
    )
    breakeven = compute_breakeven(on_prem["total_fixed_usd_month"], api_cost_per_req)

    volumes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    curve = build_monthly_cost_curve(assumptions, on_prem, api_cost_per_req, volumes)

    report = {
        "assumptions": assumptions,
        "request_parameters": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
        },
        "api_cost_per_request_usd": api_cost_per_req,
        "on_prem_monthly": on_prem,
        "breakeven_requests_per_month": breakeven,
        "monthly_cost_curve": curve,
        "recommendation": (
            f"Below {breakeven:.0f} requests/month, the API is cheaper. "
            f"Above {breakeven:.0f} requests/month, on-prem is cheaper "
            f"assuming ${on_prem['total_fixed_usd_month']:.2f}/month fixed cost."
        ),
        "pricing_note": (
            f"API prices from {assumptions['api_pricing_date']} for "
            f"{assumptions['api_pricing_model']}. Verify current prices before "
            "using these figures in production decisions."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"API cost per request:  ${api_cost_per_req:.6f}")
    print(f"On-prem fixed/month:   ${on_prem['total_fixed_usd_month']:.2f}")
    print(f"Break-even:            {breakeven:.0f} requests/month")
    print(f"\n{report['recommendation']}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
