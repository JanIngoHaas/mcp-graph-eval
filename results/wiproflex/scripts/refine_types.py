#!/usr/bin/env python3
import sys
import re

def refine_instance_types(content):
    mapping = {
        "DRESSING_WATER_STATUS": "cmpo:DressingWaterStatus",
        "USAGE_OF_DRESSER": "cmpo:DresserUsage",
        "HEAD_ROTATION": "cmpo:HeadRotation",
        "SLURRY_FLOW_LINE_A": "cmpo:SlurryFlowLineA",
        "SLURRY_FLOW_LINE_B": "cmpo:SlurryFlowLineB",
        "SLURRY_FLOW_LINE_C": "cmpo:SlurryFlowLineC",
        "WAFER_ROTATION": "cmpo:WaferRotation",
        "STAGE_ROTATION": "cmpo:StageRotation",
        "PRESSURIZED_CHAMBER_PRESSURE": "cmpo:PressurizedChamberPressure",
        "MAIN_OUTER_AIR_BAG_PRESSURE": "cmpo:MainOuterAirBagPressure",
        "CENTER_AIR_BAG_PRESSURE": "cmpo:CenterAirBagPressure",
        "RIPPLE_AIR_BAG_PRESSURE": "cmpo:RippleAirBagPressure",
        "EDGE_AIR_BAG_PRESSURE": "cmpo:EdgeAirBagPressure",
        "USAGE_OF_DRESSER_TABLE": "cmpo:DresserTableUsage",
        "USAGE_OF_POLISHING_TABLE": "cmpo:PolishingTableUsage",
        "USAGE_OF_BACKING_FILM": "cmpo:BackingFilmUsage",
        "USAGE_OF_MEMBRANE": "cmpo:MembraneUsage",
        "USAGE_OF_PRESSURIZED_SHEET": "cmpo:PressurizedSheetUsage",
        "AVG_REMOVAL_RATE": "cmpo:AverageRemovalRate"
    }
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.startswith('ex:') and ' a ' in line:
            # Use regex to find the type after ' a '
            match = re.search(r'^(ex:[^ ]+) a ([^ ;]+)', line)
            if match:
                subject_uri = match.group(1)
                old_type = match.group(2)
                for key, new_type in mapping.items():
                    if key in subject_uri:
                        line = line.replace(f" a {old_type}", f" a {new_type}")
                        break
        new_lines.append(line)
        
    return '\n'.join(new_lines)

if __name__ == "__main__":
    file_path = "datasets/wiproflex/instances_v2.ttl"
    with open(file_path, "r") as f:
        content = f.read()
    
    refined = refine_instance_types(content)
    
    with open(file_path, "w") as f:
        f.write(refined)
    print("Successfully refined instance types using robust regex matching.")
