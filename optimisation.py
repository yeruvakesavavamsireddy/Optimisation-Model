
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import pulp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

SECTION_LINE = "="*70

def section(title: str):
    print(f"\n{SECTION_LINE}")
    print(f"  {title}")
    print(SECTION_LINE)


def print_status(prob: pulp.LpProblem):
    status = pulp.LpStatus[prob.status]
    symbol = "✅" if status == "Optimal" else "⚠️"
    print(f"\n  {symbol} Solver Status : {status}")
    print(f"  Objective Value  : {pulp.value(prob.objective):,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM A — Production Planning
# ══════════════════════════════════════════════════════════════════════════════
# A manufacturer makes 5 products (P1–P5).
# Each requires machine time, raw material, and labour.
# Goal: Maximise profit subject to resource constraints and min/max demand.
# ══════════════════════════════════════════════════════════════════════════════

def solve_production_planning() -> dict:
    section("PROBLEM A: Production Planning Optimisation")

    # ── Data ──────────────────────────────────────────────────────────────────
    products = ["P1", "P2", "P3", "P4", "P5"]

    profit_per_unit = {"P1": 25, "P2": 30, "P3": 15, "P4": 40, "P5": 20}

    # Resource usage per unit
    machine_hours   = {"P1": 2.0, "P2": 3.5, "P3": 1.0, "P4": 4.5, "P5": 1.5}
    raw_material_kg = {"P1": 1.5, "P2": 2.0, "P3": 0.8, "P4": 3.0, "P5": 1.0}
    labour_hours    = {"P1": 1.0, "P2": 1.5, "P3": 0.5, "P4": 2.0, "P5": 0.8}

    # Total available resources
    MAX_MACHINE_HOURS   = 800
    MAX_RAW_MATERIAL_KG = 500
    MAX_LABOUR_HOURS    = 400

    # Demand bounds
    min_demand = {"P1": 20, "P2": 10, "P3": 30, "P4":  5, "P5": 25}
    max_demand = {"P1":200, "P2":150, "P3":300, "P4": 80, "P5":250}

    # ── Model ─────────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

    x = {p: pulp.LpVariable(f"x_{p}", lowBound=min_demand[p],
                              upBound=max_demand[p], cat="Integer")
         for p in products}

    # Objective: Maximise total profit
    prob += pulp.lpSum(profit_per_unit[p] * x[p] for p in products), "Total_Profit"

    # Resource constraints
    prob += pulp.lpSum(machine_hours[p]   * x[p] for p in products) <= MAX_MACHINE_HOURS,   "Machine_Hours"
    prob += pulp.lpSum(raw_material_kg[p] * x[p] for p in products) <= MAX_RAW_MATERIAL_KG, "Raw_Material"
    prob += pulp.lpSum(labour_hours[p]    * x[p] for p in products) <= MAX_LABOUR_HOURS,    "Labour_Hours"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print_status(prob)

    # ── Results ───────────────────────────────────────────────────────────────
    results = {}
    print(f"\n  {'Product':<10} {'Units':>8} {'Profit ($)':>12} {'Machine h':>10} {'Material':>10} {'Labour h':>10}")
    print("  " + "-"*65)
    for p in products:
        units   = int(pulp.value(x[p]))
        profit  = units * profit_per_unit[p]
        results[p] = {"units": units, "profit": profit}
        print(f"  {p:<10} {units:>8} {profit:>12,.0f} "
              f"{units*machine_hours[p]:>10.1f} "
              f"{units*raw_material_kg[p]:>10.1f} "
              f"{units*labour_hours[p]:>10.1f}")

    total_profit = sum(v["profit"] for v in results.values())
    print(f"\n  {'TOTAL':>10} {'':>8} {total_profit:>12,.0f}")

    # Resource utilisation
    used_machine  = sum(machine_hours[p]   * int(pulp.value(x[p])) for p in products)
    used_material = sum(raw_material_kg[p] * int(pulp.value(x[p])) for p in products)
    used_labour   = sum(labour_hours[p]    * int(pulp.value(x[p])) for p in products)
    print(f"\n  Resource Utilisation:")
    print(f"    Machine  : {used_machine:.1f} / {MAX_MACHINE_HOURS}   ({used_machine/MAX_MACHINE_HOURS*100:.1f}%)")
    print(f"    Material : {used_material:.1f} / {MAX_RAW_MATERIAL_KG}   ({used_material/MAX_RAW_MATERIAL_KG*100:.1f}%)")
    print(f"    Labour   : {used_labour:.1f} / {MAX_LABOUR_HOURS}   ({used_labour/MAX_LABOUR_HOURS*100:.1f}%)")

    return {
        "results"      : results,
        "total_profit" : total_profit,
        "utilisation"  : {
            "machine" : used_machine/MAX_MACHINE_HOURS,
            "material": used_material/MAX_RAW_MATERIAL_KG,
            "labour"  : used_labour/MAX_LABOUR_HOURS
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM B — Staff Scheduling
# ══════════════════════════════════════════════════════════════════════════════
# A call centre needs to staff 7 shifts over a week.
# Each employee works 5 consecutive days; Goal: Minimise total staff needed
# while meeting minimum daily headcount requirements.
# ══════════════════════════════════════════════════════════════════════════════

def solve_staff_scheduling() -> dict:
    section("PROBLEM B: Staff Scheduling Optimisation")

    days     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    min_staff = [15,  18,   20,   20,   18,  12,    8]   # min staff per day

    # x[i] = employees starting on day i (work days i..i+4 cyclically)
    prob = pulp.LpProblem("Staff_Scheduling", pulp.LpMinimize)
    x    = {d: pulp.LpVariable(f"start_{d}", lowBound=0, cat="Integer") for d in days}

    # Objective: Minimise total employees
    prob += pulp.lpSum(x[d] for d in days), "Total_Employees"

    # Each day's coverage constraint (each employee covers 5 consecutive days)
    for i, day in enumerate(days):
        covering = [days[(i - j) % 7] for j in range(5)]
        prob += pulp.lpSum(x[d] for d in covering) >= min_staff[i], f"Coverage_{day}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print_status(prob)

    schedule = {d: int(pulp.value(x[d])) for d in days}
    total    = sum(schedule.values())

    # Compute actual coverage per day
    actual_coverage = {}
    for i, day in enumerate(days):
        actual_coverage[day] = sum(schedule[days[(i-j) % 7]] for j in range(5))

    print(f"\n  {'Day':<8} {'Starting':>10} {'Actual':>10} {'Required':>10} {'Slack':>8}")
    print("  " + "-"*50)
    for i, day in enumerate(days):
        slack = actual_coverage[day] - min_staff[i]
        print(f"  {day:<8} {schedule[day]:>10} {actual_coverage[day]:>10} {min_staff[i]:>10} {slack:>8}")
    print(f"\n  Total Employees Required: {total}")

    return {"schedule": schedule, "coverage": actual_coverage,
            "required": dict(zip(days, min_staff)), "total_employees": total}


# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM C — Transportation / Supply Chain
# ══════════════════════════════════════════════════════════════════════════════
# 3 warehouses supply 4 retail stores.
# Goal: Minimise total shipping cost while meeting all supply/demand constraints.
# ══════════════════════════════════════════════════════════════════════════════

def solve_transportation() -> dict:
    section("PROBLEM C: Supply Chain Transportation Optimisation")

    warehouses = ["WH-A", "WH-B", "WH-C"]
    stores     = ["S1",   "S2",   "S3",   "S4"]

    supply     = {"WH-A": 300, "WH-B": 400, "WH-C": 250}
    demand     = {"S1": 200,   "S2": 300,   "S3": 250,   "S4": 150}

    # Shipping cost per unit ($)
    cost = {
        "WH-A": {"S1":  4, "S2":  8, "S3":  6, "S4": 10},
        "WH-B": {"S1":  7, "S2":  3, "S3":  5, "S4":  9},
        "WH-C": {"S1":  9, "S2":  6, "S3":  4, "S4":  7},
    }

    prob = pulp.LpProblem("Transportation", pulp.LpMinimize)
    x    = {(w, s): pulp.LpVariable(f"ship_{w}_{s}", lowBound=0, cat="Integer")
            for w in warehouses for s in stores}

    # Objective: Minimise total shipping cost
    prob += pulp.lpSum(cost[w][s] * x[(w, s)] for w in warehouses for s in stores)

    # Supply constraints
    for w in warehouses:
        prob += pulp.lpSum(x[(w, s)] for s in stores) <= supply[w], f"Supply_{w}"

    # Demand constraints
    for s in stores:
        prob += pulp.lpSum(x[(w, s)] for w in warehouses) >= demand[s], f"Demand_{s}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print_status(prob)

    # ── Results Table ─────────────────────────────────────────────────────────
    print(f"\n  Shipment Plan (units):")
    header = f"  {'':>8}" + "".join(f"{s:>8}" for s in stores) + f"  {'Supply':>8} {'Used':>6}"
    print(header)
    print("  " + "-" * (8 + 8*len(stores) + 18))

    total_cost = 0
    shipments  = {}
    for w in warehouses:
        row     = ""
        w_total = 0
        for s in stores:
            val    = int(pulp.value(x[(w, s)]))
            total_cost += val * cost[w][s]
            shipments[(w, s)] = val
            row    += f"{val:>8}"
            w_total += val
        print(f"  {w:>8}{row}  {supply[w]:>8} {w_total:>6}")

    print("  " + "-" * (8 + 8*len(stores) + 18))
    demand_row = "".join(f"{demand[s]:>8}" for s in stores)
    print(f"  {'Demand':>8}{demand_row}")
    print(f"\n  ✔  Minimum Total Shipping Cost: ${total_cost:,.0f}")

    return {"shipments": shipments, "total_cost": total_cost,
            "supply": supply, "demand": demand, "cost": cost}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION — 3-panel summary dashboard
# ══════════════════════════════════════════════════════════════════════════════

def visualise_all(res_a: dict, res_b: dict, res_c: dict):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("TASK 4 — Business Optimisation Model Results",
                 fontsize=16, fontweight="bold", y=1.01)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── A1: Units produced ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    prods  = list(res_a["results"].keys())
    units  = [res_a["results"][p]["units"]  for p in prods]
    profit = [res_a["results"][p]["profit"] for p in prods]
    colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]
    bars = ax1.bar(prods, units, color=colors)
    ax1.set_title("A: Units Produced per Product")
    ax1.set_ylabel("Units")
    for bar, u in zip(bars, units):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(u), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # ── A2: Resource utilisation ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    resources = list(res_a["utilisation"].keys())
    util_vals = [v * 100 for v in res_a["utilisation"].values()]
    bars2 = ax2.barh(resources, util_vals, color=["#F44336","#FF9800","#4CAF50"])
    ax2.set_xlim(0, 110); ax2.axvline(100, color="red", linestyle="--", lw=1.5)
    ax2.set_title("A: Resource Utilisation (%)")
    ax2.set_xlabel("Utilisation (%)")
    for bar, val in zip(bars2, util_vals):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    # ── A3: Profit per product ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(prods, [p/1000 for p in profit], color=colors)
    ax3.set_title(f"A: Profit per Product\n(Total: ${res_a['total_profit']:,.0f})")
    ax3.set_ylabel("Profit ($k)")
    ax3.grid(axis="y", alpha=0.3)

    # ── B: Staff scheduling ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    days     = list(res_b["coverage"].keys())
    actual   = [res_b["coverage"][d]  for d in days]
    required = [res_b["required"][d]  for d in days]
    x_pos    = np.arange(len(days))
    ax4.bar(x_pos, actual,   0.4, label="Actual Coverage", color="#2196F3", alpha=0.8)
    ax4.plot(x_pos, required, "ro--", lw=2, label="Minimum Required", markersize=7)
    ax4.set_xticks(x_pos); ax4.set_xticklabels(days)
    ax4.set_title(f"B: Staff Schedule\n(Total employees: {res_b['total_employees']})")
    ax4.set_ylabel("Staff Count")
    ax4.legend(); ax4.grid(axis="y", alpha=0.3)

    # ── C1: Shipment heatmap ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    warehouses = ["WH-A", "WH-B", "WH-C"]
    stores     = ["S1","S2","S3","S4"]
    matrix = np.array([[res_c["shipments"][(w, s)] for s in stores]
                        for w in warehouses])
    im = ax5.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax5.set_xticks(range(len(stores))); ax5.set_yticks(range(len(warehouses)))
    ax5.set_xticklabels(stores); ax5.set_yticklabels(warehouses)
    ax5.set_title("C: Shipment Plan (units)")
    for i in range(len(warehouses)):
        for j in range(len(stores)):
            ax5.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    plt.colorbar(im, ax=ax5, label="Units shipped")

    # ── C2: Cost breakdown per route ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    route_costs = {f"{w}→{s}": res_c["shipments"][(w,s)] * res_c["cost"][w][s]
                   for w in warehouses for s in stores
                   if res_c["shipments"][(w,s)] > 0}
    top_routes  = sorted(route_costs.items(), key=lambda x: x[1], reverse=True)[:8]
    rnames, rvals = zip(*top_routes)
    ax6.barh(list(rnames)[::-1], [v/1000 for v in rvals][::-1], color="#FF5722")
    ax6.set_title(f"C: Top Route Costs\n(Total: ${res_c['total_cost']:,.0f})")
    ax6.set_xlabel("Cost ($k)")
    ax6.grid(axis="x", alpha=0.3)

    plt.savefig("/mnt/user-data/outputs/task4_optimization_results.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  ✔  Optimisation dashboard saved → task4_optimization_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHTS REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_insights(res_a, res_b, res_c):
    section("BUSINESS INSIGHTS & RECOMMENDATIONS")

    print("\n  📊 Problem A — Production Planning:")
    best_product = max(res_a["results"], key=lambda p: res_a["results"][p]["profit"])
    print(f"    • Maximum profit: ${res_a['total_profit']:,.0f}")
    print(f"    • Best performing product: {best_product} "
          f"(${res_a['results'][best_product]['profit']:,.0f} profit)")
    bottleneck = max(res_a["utilisation"], key=res_a["utilisation"].get)
    print(f"    • Bottleneck resource: {bottleneck.title()} "
          f"({res_a['utilisation'][bottleneck]*100:.1f}% used)")
    print(f"    • Recommendation: Expand {bottleneck} capacity to increase output.")

    print("\n  📊 Problem B — Staff Scheduling:")
    slack_days = [(d, res_b["coverage"][d] - res_b["required"][d])
                  for d in res_b["coverage"]]
    max_slack_day = max(slack_days, key=lambda x: x[1])
    print(f"    • Minimum staff needed: {res_b['total_employees']} employees")
    print(f"    • Day with most slack: {max_slack_day[0]} (+{max_slack_day[1]} extra staff)")
    print(f"    • Recommendation: Consider part-time staff on {max_slack_day[0]} to reduce cost.")

    print("\n  📊 Problem C — Transportation:")
    cheapest_route = min(res_c["shipments"], key=lambda r: res_c["cost"][r[0]][r[1]]
                         if res_c["shipments"][r] > 0 else 999)
    print(f"    • Minimum shipping cost: ${res_c['total_cost']:,.0f}")
    print(f"    • Most cost-effective route: {cheapest_route[0]} → {cheapest_route[1]} "
          f"(${res_c['cost'][cheapest_route[0]][cheapest_route[1]]}/unit)")
    print(f"    • Recommendation: Prioritise {cheapest_route[0]} for high-demand stores.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*70)
    print("  CODTECH INTERNSHIP — TASK 4: OPTIMISATION MODEL (PuLP)")
    print("█"*70)

    # Solve all three problems
    res_a = solve_production_planning()
    res_b = solve_staff_scheduling()
    res_c = solve_transportation()

    # Visualise
    visualise_all(res_a, res_b, res_c)

    # Business insights
    print_insights(res_a, res_b, res_c)

    print("\n" + "="*70)
    print("  ✅  All optimisation problems solved successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
