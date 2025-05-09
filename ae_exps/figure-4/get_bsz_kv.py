import re
import argparse
import json
from statistics import mean, StatisticsError # Using statistics.mean for simplicity

def parse_log_file(filename):
    """
    Parses a log file to extract specific metrics information.

    Args:
        filename (str): The path to the log file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the extracted metrics from a matching log line.
              Returns an empty list if the file cannot be read or no
              matching lines are found.
    """
    log_pattern = re.compile(
        r"Avg prompt throughput:\s*(\d+\.?\d*)\s*tokens/s, "
        r"Avg generation throughput:\s*(\d+\.?\d*)\s*tokens/s, "
        r"Running:\s*(\d+)\s*reqs, "
        r"Swapped:\s*(\d+)\s*reqs, "
        r"Pending:\s*(\d+)\s*reqs, "
        r"GPU KV cache usage:\s*(\d+\.?\d*)%, "
        r"CPU KV cache usage:\s*(\d+\.?\d*)%\."
    )

    extracted_data = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    groups = match.groups()
                    try:
                        data = {
                            "prompt_throughput": float(groups[0]),
                            "generation_throughput": float(groups[1]),
                            "running_reqs": int(groups[2]),
                            "swapped_reqs": int(groups[3]),
                            "pending_reqs": int(groups[4]),
                            "gpu_cache_usage_percent": float(groups[5]),
                            "cpu_cache_usage_percent": float(groups[6]),
                            "raw_line": line.strip() # Optional: Keep the original line
                        }
                        extracted_data.append(data)
                    except (ValueError, IndexError) as e:
                         print(f"Warning: Could not parse data from line: {line.strip()}. Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return [] # Return empty list on error
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        return [] # Return empty list on error

    return extracted_data

def calculate_averages(data_list):
    """
    Calculates the average for each numerical key in a list of dictionaries.

    Args:
        data_list (list): A list of dictionaries (like the ones from parse_log_file).

    Returns:
        dict: A dictionary containing the average value for each key.
              Returns an empty dictionary if the input list is empty.
    """
    if not data_list:
        return {}

    averages = {}
    # Define the keys we want to average
    keys_to_average = [
        "prompt_throughput", "generation_throughput",
        "running_reqs", "swapped_reqs", "pending_reqs",
        "gpu_cache_usage_percent", "cpu_cache_usage_percent"
    ]

    for key in keys_to_average:
        try:
            # Extract all values for the current key and calculate the mean
            values = [d[key] for d in data_list]
            averages[key] = mean(values)
        except StatisticsError:
            averages[key] = None # Handle case where mean cannot be calculated (e.g., empty list, though checked above)
        except KeyError:
            print(f"Warning: Key '{key}' not found in all data points for averaging.")
            averages[key] = None # Or handle as appropriate

    return averages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract performance metrics from log files, filter by index, and calculate averages."
    )
    parser.add_argument("filename", help="Path to the log file to parse.")

    args = parser.parse_args()

    all_results = parse_log_file(args.filename)

    if not all_results:
        print("No matching log entries found or error reading file.")
    else:
        print(f"Found {len(all_results)} total matching log entries.")

        # --- Filtering Step ---
        # Define the slice range (0-based index)
        # Indices 60 up to (but not including) 70 -> Elements 61 through 70
        start_index = int(0.6 * len(all_results)) # 60% of the total length
        end_index = int(0.7 * len(all_results)) # 70% of the total length

        if len(all_results) < start_index:
            print(f"\nNot enough entries ({len(all_results)}) to start filtering at index {start_index}.")
            filtered_results = []
        else:
            # Apply the filter (slice based on order of appearance)
            filtered_results = all_results[start_index:end_index]
            print(f"\n--- Filtered Entries (Index {start_index} to {end_index - 1}) ---")
            if not filtered_results:
                print(f"No entries found within the specified index range [{start_index}:{end_index}).")
            else:
                 print(f"Showing {len(filtered_results)} entries from the specified range:")
                 for i, entry in enumerate(filtered_results):
                     # Adding original index for context
                     original_index = start_index + i
                     print(f"Original Index: {original_index}")
                     print(json.dumps(entry, indent=2)) # Pretty print each entry
                     print("-" * 10)

        # --- Averaging Step ---
        if filtered_results:
            print(f"\n--- Average Values for Filtered Entries (Indices {start_index}-{end_index - 1}) ---")
            average_values = calculate_averages(filtered_results)

            # Print averages nicely formatted
            print(json.dumps(average_values, indent=4))
            # Example of printing with specific formatting:
            # print("Calculated Averages:")
            # for key, avg_value in average_values.items():
            #     if avg_value is not None:
            #          # Format floats to 2 decimal places, keep ints as ints
            #         if isinstance(avg_value, float):
            #             print(f"  Average {key}: {avg_value:.2f}")
            #         else:
            #             print(f"  Average {key}: {avg_value}")
            #     else:
            #         print(f"  Average {key}: N/A")

            print("-" * 20)
        # else: # Message already printed above if filtered_results is empty
        #     pass