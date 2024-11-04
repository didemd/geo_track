import sys
import os
from helpers import standardize_headers

input_csv = sys.argv[1]
output_csv = sys.argv[2]

output_dir = os.path.dirname(output_csv)
    
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

df = standardize_headers(input_csv)
if df is not None and not df.empty:
    df.to_csv(output_csv, index=False)
else:
    print(f"Failed to process {input_csv}")
    