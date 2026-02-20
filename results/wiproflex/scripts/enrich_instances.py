#!/usr/bin/env python3
"""
Enrich instance data with missing properties for entity types that currently
have zero (or very few) datatype/object properties beyond rdfs:label and rdf:type.

This version:
1. Refines Parameter types to match CSV headers (Specific subclasses).
2. Augments all Parameters with SensorID and Timestamp.
3. Adds metadata for Lots, Projects, Tools, Wafers, and other domain entities.
4. Ensures every entity type has ≥3 outgoing properties.

Usage:
    python scripts/enrich_instances.py datasets/wiproflex/instances_v2.ttl --ontology datasets/wiproflex/ontology_v2.ttl
"""

import re
import sys
import random

def build_enrichment_blocks() -> str:
    """Returns Turtle blocks to APPEND to the instances file."""
    lines: list[str] = []
    lines.append("")
    lines.append("#################################################################")
    lines.append("### Enrichment: Mandatory Properties for All Leaf Types")
    lines.append("#################################################################")
    lines.append("")

    # 1. Projects
    projects = ["Project_Alpha", "Project_Beta", "Project_Gamma"]
    for p in projects:
        name = p.split("_")[1]
        lines.append(f'ex:{p} a cmpo:Project ;')
        lines.append(f'    rdfs:label "{name} Research"@en ;')
        lines.append(f'    cmpo:projectName "{name} Research"^^xsd:string ;')
        lines.append(f'    cmpo:budgetCode "BC-{random.randint(1000,9999)}"^^xsd:string ;')
        lines.append(f'    cmpo:startDate "2025-01-01"^^xsd:date .')
        lines.append("")

    # 2. WaferMaterial
    wafer_materials = {
        "WaferMaterial_Copper":   {"hardness": "3.0", "density": "8.96",  "meltingPoint": "1085", "category": "Metal"},
        "WaferMaterial_Nitride":  {"hardness": "9.0", "density": "3.17",  "meltingPoint": "1900", "category": "Dielectric"},
        "WaferMaterial_Oxide":    {"hardness": "7.0", "density": "2.20",  "meltingPoint": "1713", "category": "Dielectric"},
        "WaferMaterial_Silicon":  {"hardness": "6.5", "density": "2.33",  "meltingPoint": "1414", "category": "Substrate"},
        "WaferMaterial_Tungsten": {"hardness": "7.5", "density": "19.25", "meltingPoint": "3422", "category": "Metal"},
    }
    for inst, props in wafer_materials.items():
        lines.append(f'ex:{inst} a cmpo:WaferMaterial ;')
        lines.append(f'    cmpo:mohsHardness "{props["hardness"]}"^^xsd:float ;')
        lines.append(f'    cmpo:density "{props["density"]}"^^xsd:float ;')
        lines.append(f'    cmpo:materialCategory "{props["category"]}"^^xsd:string ;')
        lines.append(f'    cmpo:meltingPoint "{props["meltingPoint"]}"^^xsd:float .')
        lines.append("")

    # 3. RetainingRing
    retaining_rings = {
        "RetainingRing_1": {"tool": "CMPTool_MIRRA_2", "chamber": "Chamber_4", "lifetime": "1500", "material": "PPS", "status": "Good"},
        "RetainingRing_2": {"tool": "CMPTool_MIRRA_2", "chamber": "Chamber_5", "lifetime": "1200", "material": "PEEK", "status": "Fair"},
        "RetainingRing_3": {"tool": "CMPTool_MIRRA_2", "chamber": "Chamber_6", "lifetime": "900",  "material": "PPS", "status": "Good"},
    }
    for inst, props in retaining_rings.items():
        lines.append(f'ex:{inst} a cmpo:RetainingRing ;')
        lines.append(f'    cmpo:belongsToTool ex:{props["tool"]} ;')
        lines.append(f'    cmpo:installedInChamber ex:{props["chamber"]} ;')
        lines.append(f'    cmpo:ringLifetimeWafers "{props["lifetime"]}"^^xsd:integer ;')
        lines.append(f'    cmpo:ringStatus "{props["status"]}"^^xsd:string ;')
        lines.append(f'    cmpo:ringMaterial "{props["material"]}"^^xsd:string .')
        lines.append("")

    # 4. ConditionerPad
    conditioner_pads = {
        "ConditionerPad_1": {"diskType": "Diamond Matrix", "gritSize": "100", "diameter": "108.0", "manufacturer": "3M"},
        "ConditionerPad_2": {"diskType": "Diamond Matrix", "gritSize": "60",  "diameter": "108.0", "manufacturer": "Kinik"},
        "ConditionerPad_3": {"diskType": "Brazed Diamond",  "gritSize": "150", "diameter": "100.0", "manufacturer": "Entegris"},
    }
    for inst, props in conditioner_pads.items():
        lines.append(f'ex:{inst} a cmpo:ConditionerPad ;')
        lines.append(f'    cmpo:diskType "{props["diskType"]}"^^xsd:string ;')
        lines.append(f'    cmpo:gritSize "{props["gritSize"]}"^^xsd:integer ;')
        lines.append(f'    cmpo:hasManufacturer "{props["manufacturer"]}"^^xsd:string ;')
        lines.append(f'    cmpo:diskDiameter "{props["diameter"]}"^^xsd:float .')
        lines.append("")

    # 5. SlurryType
    slurry_types = {
        "SlurryType_Alumina": {"particleSize": "200.0", "selectivity": "High oxide selectivity", "chemicalFamily": "Aluminum Oxide"},
        "SlurryType_Ceria":   {"particleSize": "120.0", "selectivity": "Very high oxide selectivity", "chemicalFamily": "Cerium Oxide"},
        "SlurryType_Silica":  {"particleSize": "75.0",  "selectivity": "Moderate selectivity", "chemicalFamily": "Silicon Dioxide"},
    }
    for inst, props in slurry_types.items():
        lines.append(f'ex:{inst} a cmpo:SlurryType ;')
        lines.append(f'    cmpo:meanParticleSize "{props["particleSize"]}"^^xsd:float ;')
        lines.append(f'    cmpo:selectivityNote "{props["selectivity"]}"^^xsd:string ;')
        lines.append(f'    cmpo:chemicalFamily "{props["chemicalFamily"]}"^^xsd:string .')
        lines.append("")

    # 7. Spatial
    lines.append('ex:Facility_WiproFlex a cmpo:Facility ;')
    lines.append('    cmpo:facilityLocation "Bangalore, India"^^xsd:string ;')
    lines.append('    cmpo:establishedDate "2010-05-20"^^xsd:date ;')
    lines.append('    cmpo:totalArea "50000.0"^^xsd:float .')
    lines.append("")
    lines.append('ex:Cleanroom_Alpha a cmpo:Cleanroom ;')
    lines.append('    cmpo:isoClass "ISO 5"^^xsd:string ;')
    lines.append('    cmpo:cleanroomArea "2500.0"^^xsd:float ;')
    lines.append('    cmpo:airExchangeRate "300"^^xsd:integer .')
    lines.append("")
    lines.append('ex:Bay_3 a cmpo:Bay ;')
    lines.append('    cmpo:bayNumber "3"^^xsd:integer ;')
    lines.append('    cmpo:toolCapacity "8"^^xsd:integer ;')
    lines.append('    cmpo:maintenanceTeam "Team Gamma"^^xsd:string .')
    lines.append("")

    # 8. WaferDiameter
    wafer_diams = {
        "WaferDiameter_150mm": {"wafersPerLot": "25", "standard": "SEMI-150-M1"},
        "WaferDiameter_200mm": {"wafersPerLot": "25", "standard": "SEMI-200-M2"},
        "WaferDiameter_300mm": {"wafersPerLot": "25", "standard": "SEMI-300-M3"},
    }
    for inst, props in wafer_diams.items():
        lines.append(f'ex:{inst} a cmpo:WaferDiameter ;')
        lines.append(f'    cmpo:wafersPerLot "{props["wafersPerLot"]}"^^xsd:integer ;')
        lines.append(f'    cmpo:standardCode "{props["standard"]}"^^xsd:string .')
        lines.append("")

    # 9. Conditioner
    conditioners = {
        "Conditioner_1": {"sn": "SN-COND-001", "status": "Ready", "lastService": "2026-01-15", "type": "Swing Arm"},
        "Conditioner_2": {"sn": "SN-COND-002", "status": "Ready", "lastService": "2026-01-20", "type": "Swing Arm"},
        "Conditioner_3": {"sn": "SN-COND-003", "status": "Calibration Required", "lastService": "2025-12-10", "type": "Swing Arm"},
    }
    for inst, props in conditioners.items():
        lines.append(f'ex:{inst} a cmpo:Conditioner ;')
        lines.append(f'    cmpo:serialNumber "{props["sn"]}"^^xsd:string ;')
        lines.append(f'    cmpo:conditionerStatus "{props["status"]}"^^xsd:string ;')
        lines.append(f'    cmpo:conditionerType "{props["type"]}"^^xsd:string ;')
        lines.append(f'    cmpo:lastServiceDate "{props["lastService"]}"^^xsd:date .')
        lines.append("")

    # 10. Tools
    tools = {
        "CMPTool_MIRRA_2": {"machineID": "2", "status": "Online", "location": "ex:Cleanroom_Alpha"},
        "CMPTool_IPEC_1":  {"machineID": "1", "status": "Offline", "location": "ex:Cleanroom_Alpha"},
    }
    for inst, props in tools.items():
        lines.append(f'ex:{inst} a cmpo:MIRRA ;') # or appropriate type
        lines.append(f'    cmpo:machineID "{props["machineID"]}"^^xsd:string ;')
        lines.append(f'    cmpo:toolStatus "{props["status"]}"^^xsd:string ;')
        lines.append(f'    cmpo:locatedIn {props["location"]} ;')
        lines.append(f'    cmpo:lastMaintenanceDate "2026-02-10 08:00:00"^^xsd:dateTime .')
        lines.append("")

    return "\n".join(lines) + "\n"

def transform_content(content: str) -> str:
    """
    1. Refines types based on CSV headers.
    2. Augments parameters with SensorID and Timestamp.
    3. Adds Lot/Project metadata.
    """
    type_mapping = {
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
    
    param_types = [
        "cmpo:Pressure", "cmpo:Velocity", "cmpo:SlurryFlow", "cmpo:HeadSpeed", 
        "cmpo:PlatenSpeed", "cmpo:Conditioning", "cmpo:PadLifetime", 
        "cmpo:RetainingRingPressure", "cmpo:WaferMaterialRemovalRate", "cmpo:Parameter",
        "cmpo:DressingWaterStatus", "cmpo:DresserUsage", "cmpo:HeadRotation",
        "cmpo:SlurryFlowLineA", "cmpo:SlurryFlowLineB", "cmpo:SlurryFlowLineC",
        "cmpo:WaferRotation", "cmpo:StageRotation", "cmpo:PressurizedChamberPressure",
        "cmpo:MainOuterAirBagPressure", "cmpo:CenterAirBagPressure", "cmpo:RippleAirBagPressure",
        "cmpo:EdgeAirBagPressure", "cmpo:DresserTableUsage", "cmpo:PolishingTableUsage",
        "cmpo:BackingFilmUsage", "cmpo:MembraneUsage", "cmpo:PressurizedSheetUsage",
        "cmpo:AverageRemovalRate"
    ]
    
    lines = content.split("\n")
    new_lines = []
    
    is_in_param_block = False
    proj_idx = 0
    projects = ["ex:Project_Alpha", "ex:Project_Beta", "ex:Project_Gamma"]

    for line in lines:
        if line.startswith('ex:') and ' a ' in line:
            match = re.search(r'^(ex:[^ ]+) a ([^ ;]+)', line)
            if match:
                subject_uri = match.group(1)
                old_type = match.group(2)
                
                # 1. Refine Type
                for key, new_type in type_mapping.items():
                    if key in subject_uri:
                        line = line.replace(f" a {old_type}", f" a {new_type}")
                        old_type = new_type
                        break
                
                # 2. Add properties to Lot
                if "a cmpo:Lot" in line:
                    sep = " ;" if line.strip().endswith(";") else " ;"
                    line = line.replace(" .", "").rstrip()
                    if not line.endswith(";"): line += " ;"
                    lot_num = subject_uri.split("_")[1]
                    proj = projects[proj_idx % len(projects)]
                    proj_idx += 1
                    line += f'\n    cmpo:lotID "L-{lot_num}"^^xsd:string ;\n    cmpo:lotSize "25"^^xsd:integer ;\n    cmpo:belongsToProject {proj} .'
                
                # 3. Mark for parameter augmentation
                if any(pt in old_type for pt in param_types):
                    is_in_param_block = True

        if is_in_param_block and line.strip().endswith("."):
            head = line.rstrip().rstrip(".")
            seed = sum(ord(c) for c in (line[:10] + str(len(new_lines))))
            sensor_id = f"SENS-{(seed % 1000):03}"
            timestamp = f"2026-02-18T{((seed % 24)):02}:{(seed % 60):02}:00"
            
            sep = " ;" if not head.strip().endswith(";") else ""
            new_lines.append(f'{head}{sep}')
            new_lines.append(f'    cmpo:hasSensorId "{sensor_id}"^^xsd:string ;')
            new_lines.append(f'    cmpo:hasMeasurementTimestamp "{timestamp}"^^xsd:dateTime .')
            is_in_param_block = False
            continue

        new_lines.append(line)
        if not line.strip():
            is_in_param_block = False

    # Filter out duplicate labels or types if they were accidentally added
    return "\n".join(new_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/enrich_instances.py <instances.ttl>")
        sys.exit(1)

    instance_path = sys.argv[1]
    
    with open(instance_path, "r") as f:
        content = f.read()

    print("Cleaning up previous enrichments...")
    # Remove previous enrichment blocks
    content = re.sub(r'#################################################################\n### Enrichment: Mandatory Properties.*', '', content, flags=re.DOTALL)

    print("Transforming types and augmenting parameters...")
    content = transform_content(content)
    
    print("Adding metadata enrichment blocks...")
    enrichment = build_enrichment_blocks()
    content = content.rstrip() + "\n" + enrichment

    with open(instance_path, "w") as f:
        f.write(content)

    print(f"✅ Enriched and refined instance data written to {instance_path}")

if __name__ == "__main__":
    main()
