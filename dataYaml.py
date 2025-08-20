# Python function to automatically create data.yaml config file
# 1. Uses a predefined list of class names
# 2. Creates data dictionary with correct paths to folders, number of classes, and names of classes
# 3. Writes data in YAML format to data.yaml

import yaml
import os

# --- Custom classes and representers for specific YAML formatting ---
class QuotedString(str):
    pass

def quoted_string_representer(dumper, data):
    """Custom representer to always add single quotes to strings."""
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

yaml.add_representer(QuotedString, quoted_string_representer)

class FlowList(list):
    pass

def flow_list_representer(dumper, data):
    """Custom representer to format lists in flow style (e.g., [a, b, c])"""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowList, flow_list_representer)


def create_data_yaml(path_to_data_yaml):
    """
    Creates a data.yaml file for YOLO training using a predefined class list.
    """
    # --- Define the class names directly in the script ---
    # MODIFIED: Wrap each class name with QuotedString to force quotes
    classes = [QuotedString("Rice weevil"), QuotedString("paddy"), QuotedString("worm")]
    
    number_of_classes = len(classes)
    print(f"Using {number_of_classes} classes: {[str(c) for c in classes]}") # Print without the class wrapper

    # --- Generate absolute paths for the dataset ---
    cwd = os.getcwd()
    data_root = os.path.abspath(os.path.join(cwd, 'data'))
    train_path = os.path.join(data_root, 'train', 'images')
    val_path = os.path.join(data_root, 'validation', 'images')

    # --- Create data dictionary ---
    data = {
        'path': data_root,
        'train': train_path,
        'val': val_path,
        'nc': number_of_classes,
        'names': FlowList(classes) # Use the custom FlowList for correct formatting
    }

    # --- Write data to YAML file ---
    try:
        with open(path_to_data_yaml, 'w') as f:
            # Use default_flow_style=False to keep the main structure in block style
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        
        print(f'\nSuccessfully created config file at: {os.path.abspath(path_to_data_yaml)}')
        print('--- File Contents ---')
        with open(path_to_data_yaml, 'r') as f:
            print(f.read())
        print('---------------------')

    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

# --- Define path for data.yaml and run function ---
path_to_data_yaml = 'data.yaml'

create_data_yaml(path_to_data_yaml)
