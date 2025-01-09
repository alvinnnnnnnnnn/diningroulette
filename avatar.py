import os
import sys

def main():
    try:
        # Prompt the user for an integer input between 1 and 49
        train_num = int(input("Please enter a train number (1 to 49): "))

        # Validate the input
        if train_num < 1 or train_num > 49:
            print("Error: Invalid input. Please enter a number between 1 and 49.")
            sys.exit(1)

        # Determine which script to run based on the train_num
        if 1 <= train_num <= 25:
            # Run object_detection_ar.py
            print(f"Running object_detection_ar_a.py with train_num {train_num}")
            os.system(f"python object_detection_ar_a.py {train_num}")
        # elif 26 <= train_num <= 43:
        #     # Run object_detection_c.py
        #     print(f"Running object_detection_c.py with train_num {train_num}")
        #     os.system(f"python object_detection_c.py {train_num}")
        # elif 44 <= train_num <= 49:
        #     # Run object_detection_e.py
        #     print(f"Running object_detection_e.py with train_num {train_num}")
        #     os.system(f"python object_detection_e.py {train_num}")

    except ValueError:
        print("Error: Please enter a valid integer.")
        sys.exit(1)

if __name__ == "__main__":
    main()
