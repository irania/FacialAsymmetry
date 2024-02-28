import os
import json

def generate_json_from_folder(base_path):
    data = {
        'finger_tapping': {
            'left_finger_tapping': {},
            'right_finger_tapping': {},
        }
    }
    
    # Find booth_id directories
    for booth_id in os.listdir(base_path):
        booth_path = os.path.join(base_path, booth_id)
        if os.path.isdir(booth_path):
            # Find date directories within each booth_id directory
            for date in os.listdir(booth_path):
                date_path = os.path.join(booth_path, date)
                if os.path.isdir(date_path):
                    finger_tapping_path = os.path.join(date_path, "finger_tapping")
                    # Check if the finger_tapping directory exists
                    if os.path.exists(finger_tapping_path) and os.path.isdir(finger_tapping_path):
                        # Check for left and right finger tapping files
                        left_path = os.path.join(finger_tapping_path, "left_finger_tapping.mp4")
                        right_path = os.path.join(finger_tapping_path, "right_finger_tapping.mp4")
                        if os.path.exists(left_path):
                            data['finger_tapping']['left_finger_tapping'].setdefault(booth_id, date)
                        if os.path.exists(right_path):
                            data['finger_tapping']['right_finger_tapping'].setdefault(booth_id, date)

    return json.dumps(data, indent=4)

# Specify the base path to start exploring
base_path = "Z:\\CAMERA Booth Data\\Booth\\"
json_output = generate_json_from_folder(base_path)
print(json_output)

# Specify the filename for the output JSON
output_filename = "output.json"

# Write the JSON data to the file
with open(output_filename, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)