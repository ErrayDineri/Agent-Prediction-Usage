import csv
import json
def csv_to_json(csv_file_path, output_path):
    with open(csv_file_path, "r", encoding="utf-8") as csv_f:
        reader=csv.DictReader(csv_f)
        data=list(reader)
    with open(output_path, "w", encoding="utf-8") as o_f:
        json.dump(data, o_f, indent=4)
csv_to_json(r"metric_monthly.csv", r"metric_monthly_generated.json")