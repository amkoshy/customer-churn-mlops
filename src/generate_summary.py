import os

# Set directory path relative to where this script is run (src/)
report_dir = os.path.join(os.path.dirname(__file__), "../results/classification_reports")
output_file = os.path.join(report_dir, "combined_report.txt")

# Open the output file in write mode
with open(output_file, "w") as outfile:
    # Loop through all classification report text files
    for filename in os.listdir(report_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(report_dir, filename)
            with open(file_path, "r") as infile:
                outfile.write(f"--- {filename} ---\n")  # Add a header for each file
                outfile.write(infile.read())
                outfile.write("\n\n")  # Add spacing between files

print("✅ Combined report generated at:", output_file)