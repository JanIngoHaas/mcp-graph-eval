#!/usr/bin/env python3
import re

def sanity_fix(content):
    # 1. Fix the dot inside Lot blocks
    # Looking for: cmpo:belongsToProject ex:Project_XYZ .
    # Followed by: rdfs:label
    content = re.sub(r'(cmpo:belongsToProject ex:Project_[A-Za-z]+) \.\n', r'\1 ;\n', content)
    
    # 2. Remove duplicate lotID/lotSize lines within the same block
    lines = content.split('\n')
    new_lines = []
    seen_props = set()
    current_subject = None
    
    # Also fix potential missing space before semicolon in some lines
    for line in lines:
        if line.startswith('ex:'):
            current_subject = line.split(' ')[0]
            seen_props = set()
        
        strip_line = line.strip()
        if current_subject and ('cmpo:lotID' in strip_line or 'cmpo:lotSize' in strip_line or 'cmpo:belongsToProject' in strip_line):
            prop = strip_line.split(' ')[0]
            if prop in seen_props:
                continue # Skip duplicate
            seen_props.add(prop)
            
        new_lines.append(line)
    
    return '\n'.join(new_lines)

if __name__ == "__main__":
    file_path = "datasets/wiproflex/instances_v2.ttl"
    with open(file_path, "r") as f:
        content = f.read()
    fixed = sanity_fix(content)
    with open(file_path, "w") as f:
        f.write(fixed)
    print("Sanity fix applied to Lot blocks.")
