# utils/analysis_utils.py

import csv

def build_stay_rows(log, agent_type, sigma, seed):
    rows = []
    for t in range(1, len(log)):
        prev = log[t - 1]
        curr = log[t]

        pa, ca = prev["first_stage_action"], curr["first_stage_action"]
        if pa is None or ca is None:
            continue

        rows.append({
            "agent_type": agent_type,
            "volatility": sigma,
            "seed": seed,
            "episode": curr["episode"],
            "prev_reward": float(prev["reward"]),
            "prev_transition": prev["transition_type"],
            "stay": int(pa == ca),
        })

    return rows


def save_stay_rows_to_csv(rows, out_file="stay_dataset.csv"):
    if not rows:
        print("No rows to save.")
        return

    fieldnames = rows[0].keys()
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {out_file}")
