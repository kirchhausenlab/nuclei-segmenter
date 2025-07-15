import os
import csv
import re
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Aggregate mask_report.csv files under the specified input directory.")
parser.add_argument("--input_dir", required=True, help="Path to the root directory containing subfolders.")
args = parser.parse_args()

# Use the --input_dir argument instead of a hardcoded path
ROOT_DIR = args.input_dir

# Output CSV paths
OUTPUT_CSV = os.path.join(ROOT_DIR, "total_masked_results.csv")
SUMMARY_CSV = os.path.join(ROOT_DIR, "condition_summary.csv")

# Regex to match folder names like "Something-Position1", "Something-Position2", etc.
# Group(1): condition name, Group(2): position number

FOLDER_REGEX = re.compile(r"(.+)-Position(\d+)$")
#FOLDER_REGEX = re.compile(r"^(.+)_(\d+)$")

# Prepare a list to hold all rows for the final CSV
all_rows = []

# Traverse the directory
for root, dirs, files in os.walk(ROOT_DIR):
    for d in dirs:
        match = FOLDER_REGEX.match(d)
        if match:
            condition_name = match.group(1)
            position_num = match.group(2)

            # Construct the full path to this subdirectory
            subdir_path = os.path.join(root, d)
            # The mask_report.csv should be inside this subdirectory
            csv_path = os.path.join(subdir_path, "mask_report.csv")
            if os.path.isfile(csv_path):
                # Read the CSV and append data to all_rows
                with open(csv_path, mode="r", newline="") as f:
                    reader = csv.DictReader(f)

                    # Expecting columns: id, summed_intensity, average_intensity
                    for row in reader:
                        new_row = {
                            "condition": condition_name,
                            "position": position_num,
                            "id": row["id"],
                            "summed_intensity": row["summed_intensity"],
                            "average_intensity": row["average_intensity"]
                        }
                        all_rows.append(new_row)

# Sort the rows:
# 1) By condition (alphabetically),
# 2) Then by position (numerically ascending).
all_rows.sort(key=lambda x: (x["condition"], int(x["position"])))

# Write out the aggregated CSV (total_masked_results.csv)
fieldnames = ["condition", "position", "id", "summed_intensity", "average_intensity"]
with open(OUTPUT_CSV, mode="w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

### Create the condition_summary.csv
# We'll aggregate (sum and count) intensities by condition, then compute means.

condition_stats = {}  # { condition_name: {"sum_summed": float, "sum_avg": float, "count": int} }

for row in all_rows:
    c = row["condition"]
    if c not in condition_stats:
        condition_stats[c] = {"sum_summed": 0.0, "sum_avg": 0.0, "count": 0}
    # Convert from string to float to sum
    condition_stats[c]["sum_summed"] += float(row["summed_intensity"])
    condition_stats[c]["sum_avg"] += float(row["average_intensity"])
    condition_stats[c]["count"] += 1

# Now compute means for each condition
summary_rows = []
for c, stats in condition_stats.items():
    count = stats["count"]
    mean_summed = stats["sum_summed"] / count
    mean_avg = stats["sum_avg"] / count
    
    # Format these however you prefer; we'll do 3-decimal-place scientific notation
    summary_rows.append({
        "condition": c,
        "summed_intensity": f"{mean_summed:.3e}",
        "average_intensity": f"{mean_avg:.3e}"
    })

# Sort summary rows by condition name alphabetically
summary_rows.sort(key=lambda x: x["condition"])

# Write condition_summary.csv
summary_fieldnames = ["condition", "summed_intensity", "average_intensity"]
with open(SUMMARY_CSV, mode="w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Aggregation complete! Files saved at:\n{OUTPUT_CSV}\n{SUMMARY_CSV}")
