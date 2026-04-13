"""
Real Agentic ReAct Implementation
----------------------------------
The model outputs: Thought / Action / Observation steps.
Each Action maps to a REAL Python tool that executes and returns REAL data
from the inventory.csv and suppliers.csv datasets.

Usage:
    python react_agent.py
    python react_agent.py --scenario custom
"""

import re
import csv
import math
import json
import argparse
import time
import httpx
from pathlib import Path

OLLAMA_BASE    = "http://localhost:11434"
MODEL_NAME     = "llama3.2:3b"
INVENTORY_PATH = Path("data/inventory.csv")
SUPPLIER_PATH  = Path("data/suppliers.csv")
MAX_STEPS      = 8
GPU_LAYERS     = 35

SUPPLY_CHAIN_SCENARIO = """You are an expert supply chain analyst with access to real tools.
A logistics company faces these problems:
- 35% stockout rate on critical SKUs
- 28-day average supplier lead time
- $2.1M/yr inventory carrying costs
- 67% on-time delivery rate

You must analyse the real inventory and supplier data using tools,
then produce a data-driven improvement plan.

Available tools:
- check_stockout_risk(14) → lists SKUs with less than 14 days of stock
- analyse_suppliers(0.85) → lists suppliers below 85% on-time delivery
- calculate_safety_stock(SKU-005, 0.95) → computes required safety stock
- calculate_eoq(SKU-002) → computes Economic Order Quantity
- get_carrying_cost(Electronics) → computes annual carrying cost for a category
- get_reorder_alerts() → lists all SKUs currently below reorder point

IMPORTANT: Call tools using positional arguments only, like this:
  Action: check_stockout_risk(14)
  Action: analyse_suppliers(0.85)
  Action: calculate_safety_stock(SKU-005, 0.95)
  Action: calculate_eoq(SKU-002)
  Action: get_carrying_cost(Electronics)
  Action: get_reorder_alerts()

Respond in this exact format for each step:
Thought: <your reasoning>
Action: tool_name(argument)

Wait for the Observation before your next Thought.
Continue until you have enough data, then write:
Final Answer: <your complete data-driven improvement plan>"""


def load_inventory():
    rows = []
    with open(INVENTORY_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from all values to prevent type conversion issues
            rows.append({k: v.strip() for k, v in row.items()})
    return rows


def load_suppliers():
    rows = []
    with open(SUPPLIER_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})
    return rows


# ── REAL TOOLS ─────────────────────────────────────────────────────────────────

def check_stockout_risk(threshold_days) -> str:
    """Returns SKUs with on_hand_days below threshold — reads real CSV data."""
    threshold_days = float(threshold_days)
    inventory = load_inventory()
    at_risk = []
    for item in inventory:
        days = float(item["on_hand_days"])
        lead = int(item["lead_time_days"])
        if days < threshold_days:
            at_risk.append({
                "sku":          item["sku_id"],
                "name":         item["product_name"],
                "on_hand_days": days,
                "lead_time_days": lead,
                "gap_days":     round(lead - days, 1),
                "supplier":     item["supplier"],
            })
    at_risk.sort(key=lambda x: x["on_hand_days"])
    if not at_risk:
        return f"No SKUs found with on-hand stock below {threshold_days} days."
    lines = [f"SKUs with < {threshold_days:.0f} days of stock (sorted by urgency):"]
    for item in at_risk[:8]:
        gap     = item["gap_days"]
        urgency = "CRITICAL" if item["on_hand_days"] < 5 else "WARNING"
        lines.append(
            f"  [{urgency}] {item['sku']} | {item['name'][:30]:30s} | "
            f"Stock: {item['on_hand_days']}d | Lead: {item['lead_time_days']}d | "
            f"Gap: +{gap}d | Supplier: {item['supplier']}"
        )
    lines.append(f"Total at-risk SKUs: {len(at_risk)} of {len(inventory)}")
    lines.append(f"Implied stockout rate: {round(len(at_risk)/len(inventory)*100,1)}%")
    return "\n".join(lines)


def analyse_suppliers(min_otd_rate) -> str:
    """Returns suppliers below on-time delivery threshold."""
    min_otd_rate = float(min_otd_rate)
    suppliers = load_suppliers()
    underperforming = []
    for s in suppliers:
        otd = float(s["on_time_delivery_rate"])
        if otd < min_otd_rate:
            underperforming.append(s)
    underperforming.sort(key=lambda x: float(x["on_time_delivery_rate"]))
    if not underperforming:
        return f"All suppliers meet the {min_otd_rate*100:.0f}% OTD threshold."
    total_spend = sum(int(s["annual_spend_usd"]) for s in underperforming)
    lines = [f"Suppliers below {min_otd_rate*100:.0f}% on-time delivery:"]
    for s in underperforming[:8]:
        otd_pct = round(float(s["on_time_delivery_rate"]) * 100, 1)
        reject  = round(float(s["quality_rejection_rate"]) * 100, 2)
        spend   = int(s["annual_spend_usd"])
        lines.append(
            f"  {s['supplier_name']:25s} | OTD: {otd_pct}% | "
            f"Lead: {s['avg_lead_time_days']}d | Rejection: {reject}% | "
            f"Audit: {s['last_audit_score']}/100 | Spend: ${spend:,}"
        )
    lines.append(f"\nTotal underperforming suppliers: {len(underperforming)}")
    lines.append(f"Total annual spend at risk: ${total_spend:,}")
    avg_lead = round(
        sum(float(s["avg_lead_time_days"]) for s in underperforming) / len(underperforming), 1
    )
    lines.append(f"Average lead time (underperformers): {avg_lead} days")
    return "\n".join(lines)


def calculate_safety_stock(sku_id, service_level=0.95) -> str:
    """
    Computes safety stock: SS = Z * sqrt(LT * sigma_demand^2 + D^2 * sigma_LT^2)
    """
    sku_id       = str(sku_id).strip().upper()
    service_level = float(service_level)
    inventory    = load_inventory()
    item = next((i for i in inventory if i["sku_id"].upper() == sku_id), None)
    if not item:
        avail = [i["sku_id"] for i in inventory[:5]]
        return f"SKU '{sku_id}' not found. Available examples: {avail}"

    service_to_z = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
    closest_sl   = min(service_to_z.keys(), key=lambda k: abs(k - service_level))
    z            = service_to_z[closest_sl]

    lead_time     = float(item["lead_time_days"])
    annual_demand = float(item["annual_demand"])
    daily_demand  = annual_demand / 365

    suppliers_data = load_suppliers()
    supplier_info  = next(
        (s for s in suppliers_data if s["supplier_name"] == item["supplier"]), None
    )
    lead_variance = float(supplier_info["lead_time_variance_days"]) if supplier_info else lead_time * 0.15

    sigma_demand = daily_demand * 0.20
    sigma_lt     = lead_variance
    safety_stock = round(
        math.sqrt(lead_time * sigma_demand**2 + daily_demand**2 * sigma_lt**2) * z, 0
    )
    reorder_point = round(daily_demand * lead_time + safety_stock, 0)
    current_ss    = int(item["safety_stock"])
    gap           = int(safety_stock) - current_ss

    lines = [
        f"Safety Stock Analysis — {sku_id} ({item['product_name']})",
        f"  Formula: SS = Z x sqrt(LT x sigma_d^2 + D^2 x sigma_LT^2)",
        f"  Service level: {closest_sl*100:.0f}%  |  Z-score: {z}",
        f"  Daily demand: {daily_demand:.2f} units/day",
        f"  Lead time: {lead_time:.0f} days (variance={sigma_lt:.1f} days)",
        f"  Calculated safety stock : {int(safety_stock)} units",
        f"  Current safety stock    : {current_ss} units",
        f"  Gap (units to add)      : {gap:+d} units",
        f"  Recommended reorder pt  : {int(reorder_point)} units",
        f"  Current reorder point   : {item['reorder_point']} units",
    ]
    if gap > 0:
        extra_cost = gap * float(item["unit_cost"]) * 0.25
        lines.append(f"  Extra carrying cost if gap filled: ${extra_cost:,.0f}/yr")
    return "\n".join(lines)


def calculate_eoq(sku_id) -> str:
    """EOQ = sqrt(2 * D * S / H)"""
    sku_id    = str(sku_id).strip().upper()
    inventory = load_inventory()
    item = next((i for i in inventory if i["sku_id"].upper() == sku_id), None)
    if not item:
        return f"SKU '{sku_id}' not found."

    D         = float(item["annual_demand"])
    unit_cost = float(item["unit_cost"])
    S         = 50.0
    H         = unit_cost * 0.25
    eoq       = math.sqrt((2 * D * S) / H)

    current_rq           = float(item["reorder_qty"])
    annual_orders_current = D / current_rq
    annual_orders_eoq     = D / eoq
    cost_current          = (current_rq / 2) * H + annual_orders_current * S
    cost_eoq              = (eoq / 2) * H + annual_orders_eoq * S
    savings               = cost_current - cost_eoq

    lines = [
        f"EOQ Analysis — {sku_id} ({item['product_name']})",
        f"  Formula: EOQ = sqrt(2 x D x S / H)",
        f"  Annual demand (D): {D:,.0f} units  |  Order cost (S): ${S}  |  Holding rate: 25%",
        f"  Optimal EOQ       : {eoq:.0f} units/order",
        f"  Current order qty : {current_rq:.0f} units/order",
        f"  Orders/year (EOQ) : {annual_orders_eoq:.1f}",
        f"  Annual cost (EOQ) : ${cost_eoq:,.0f}",
        f"  Annual cost now   : ${cost_current:,.0f}",
        f"  Potential savings : ${savings:,.0f}/yr",
    ]
    return "\n".join(lines)


def get_carrying_cost(category) -> str:
    """Computes total annual carrying cost (25% rule) for a category."""
    category  = str(category).strip()
    inventory = load_inventory()
    cat_items = [i for i in inventory if i["category"].lower() == category.lower()]
    if not cat_items:
        categories = sorted(set(i["category"] for i in inventory))
        return f"Category '{category}' not found. Available: {categories}"

    total_carrying = 0
    lines_data     = []
    for item in cat_items:
        stock    = float(item["current_stock"])
        cost     = float(item["unit_cost"])
        inv_val  = stock * cost
        carrying = inv_val * 0.25
        total_carrying += carrying
        lines_data.append((item["sku_id"], item["product_name"][:28], inv_val, carrying))

    lines = [
        f"Annual Carrying Cost — {category} (25% of inventory value/yr)",
        f"  {'SKU':10s} {'Product':28s} {'Inv Value':>12s} {'Carry/yr':>10s}",
        f"  {'─'*66}",
    ]
    for sku, name, val, carry in sorted(lines_data, key=lambda x: -x[3]):
        lines.append(f"  {sku:10s} {name:28s} ${val:>10,.0f}  ${carry:>9,.0f}")
    lines.append(f"  {'─'*66}")
    lines.append(f"  Total carrying cost ({category}): ${total_carrying:,.0f}/yr")
    return "\n".join(lines)


def get_reorder_alerts() -> str:
    """Lists all SKUs where current_stock < reorder_point."""
    inventory = load_inventory()
    alerts    = []
    for item in inventory:
        curr = int(item["current_stock"])
        rop  = int(item["reorder_point"])
        if curr < rop:
            alerts.append({
                "sku":         item["sku_id"],
                "name":        item["product_name"],
                "current":     curr,
                "reorder_pt":  rop,
                "deficit":     rop - curr,
                "lead_days":   int(item["lead_time_days"]),
                "supplier":    item["supplier"],
                "last_stockout": item["last_stockout_date"],
            })
    alerts.sort(key=lambda x: x["current"])
    if not alerts:
        return "All SKUs are above their reorder points. No immediate action needed."
    lines = [f"REORDER ALERTS — {len(alerts)} SKUs below reorder point:"]
    for a in alerts:
        lines.append(
            f"  {a['sku']:10s} {a['name'][:28]:28s} | "
            f"Stock: {a['current']:4d} | ROP: {a['reorder_pt']:4d} | "
            f"Deficit: {a['deficit']:4d} | Lead: {a['lead_days']}d | "
            f"Last stockout: {a['last_stockout']}"
        )
    lines.append(f"\nImmediate action required: Place orders for {len(alerts)} SKUs")
    return "\n".join(lines)


TOOLS = {
    "check_stockout_risk":    check_stockout_risk,
    "analyse_suppliers":      analyse_suppliers,
    "calculate_safety_stock": calculate_safety_stock,
    "calculate_eoq":          calculate_eoq,
    "get_carrying_cost":      get_carrying_cost,
    "get_reorder_alerts":     get_reorder_alerts,
}


def parse_action(text: str):
    """
    Extract tool name and argument(s) from model output.
    Handles:
      - positional:  check_stockout_risk(14)
      - keyword:     check_stockout_risk(threshold_days=14)
      - two args:    calculate_safety_stock(SKU-005, 0.95)
      - no args:     get_reorder_alerts()
    """
    match = re.search(r'Action:\s*(\w+)\(([^)]*)\)', text)
    if not match:
        return None, None

    tool_name = match.group(1).strip()
    raw_args  = match.group(2).strip()

    if not raw_args:
        return tool_name, None

    # Split on comma for multi-arg tools
    parts = [p.strip() for p in raw_args.split(',')]

    parsed = []
    for part in parts:
        # Strip keyword name if present: threshold_days=14 -> 14
        if '=' in part:
            part = part.split('=', 1)[1].strip()
        part = part.strip('"').strip("'")
        # Convert to number if possible
        try:
            if '.' in part:
                parsed.append(float(part))
            else:
                parsed.append(int(part))
        except ValueError:
            parsed.append(part)

    if len(parsed) == 1:
        return tool_name, parsed[0]
    return tool_name, parsed   # return list for multi-arg


def call_tool(tool_name: str, arg) -> str:
    if tool_name not in TOOLS:
        avail = list(TOOLS.keys())
        return f"Unknown tool '{tool_name}'. Available tools: {avail}"
    try:
        if arg is None:
            return TOOLS[tool_name]()
        elif isinstance(arg, list):
            return TOOLS[tool_name](*arg)
        else:
            return TOOLS[tool_name](arg)
    except Exception as e:
        return f"Tool execution error in '{tool_name}': {e}"


def query_ollama(messages: list) -> str:
    prompt = "\n".join(messages)
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p":       0.9,
            "num_predict": 350,
            "num_gpu":     GPU_LAYERS,
            "main_gpu":    0,
            "f16_kv":      True,
            "stop":        ["Observation:", "\n\nThought:", "\n\nStep"],
        },
    }
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


def run_react_agent(scenario: str = SUPPLY_CHAIN_SCENARIO, verbose: bool = True):
    context      = [scenario]
    final_answer = None

    if verbose:
        print("\n" + "=" * 70)
        print("AGENTIC ReAct — Supply Chain Analysis")
        print("=" * 70)

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n{'─'*70}")
            print(f"STEP {step}/{MAX_STEPS}")
            print(f"{'─'*70}")

        model_output = query_ollama(context)

        if not model_output:
            if verbose:
                print("[No output from model]")
            break

        if verbose:
            print(model_output)

        if "Final Answer:" in model_output:
            fa_idx       = model_output.index("Final Answer:")
            final_answer = model_output[fa_idx + len("Final Answer:"):].strip()
            if verbose:
                print(f"\n{'='*70}")
                print("FINAL ANSWER:")
                print(f"{'='*70}")
                print(final_answer)
            break

        tool_name, arg = parse_action(model_output)
        if tool_name:
            if verbose:
                print(f"\nObservation: [Executing {tool_name}({arg})]")
            observation = call_tool(tool_name, arg)
            if verbose:
                print(observation)
            context.append(model_output)
            context.append(f"Observation:\n{observation}\n")
        else:
            context.append(model_output)
            if "Thought:" not in model_output and "Action:" not in model_output:
                if verbose:
                    print("[Model did not produce a valid Action. Stopping.]")
                break

    if not final_answer and verbose:
        print(f"\n[Reached step limit ({MAX_STEPS}) without Final Answer]")

    return final_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="default")
    args = parser.parse_args()

    if args.scenario == "custom":
        print("Enter your supply chain scenario (blank line to submit):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        scenario = SUPPLY_CHAIN_SCENARIO + "\n\nAdditional context: " + " ".join(lines)
    else:
        scenario = SUPPLY_CHAIN_SCENARIO

    run_react_agent(scenario)
