---
description: How to enrich data, index with QLever, and generate questions
---

# Workflow: CMP Graph Question Generation

Follow these steps command-by-command to process the data and generate question samples.

### 1. Enrich Instance Data
This script aligns the instance data with CSV headers and adds mandatory properties (SensorID, Timestamps, Project metadata) to ensure every entity has at least 3 outgoing properties.
```bash
python scripts/enrich_instances.py datasets/wiproflex/instances_v2.ttl
```

### 2. Navigate to QLever Dataset Directory
QLever commands must be run from the directory containing the `Qleverfile` and `.ttl` files.
```bash
cd datasets/wiproflex
```

### 3. Index the Data
This builds the QLever index from the ontology and instance files.
// turbo
```bash
uv run qlever index --overwrite-existing
```

### 4. Start the QLever Server
This starts the SPARQL endpoint on port 7001.
// turbo
```bash
uv run qlever start
```

### 5. Generate Question Samples
Navigate back to the project root and run the generation script.
```bash
cd ../..
uv run python -m src.runner
```

### 6. (Optional) Stop the QLever Server
When finished, you can stop the QLever container.
```bash
cd datasets/wiproflex
uv run qlever stop
```
