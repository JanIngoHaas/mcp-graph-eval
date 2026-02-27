# Found=False Audit

- Computed automatically from all loaded `eval_results_*.toml` rows.
- `found=false` is expected for correctly handled `impossible` questions, but suspicious for non-impossible rows.
- Models with at least one `found=false`: 10/10.
- Total rows audited: 3020.

## Combined (all datasets)

| model | found=false | found=true | found_missing | found=false (impossible) | found=false (non-impossible) |
| --- | --- | --- | --- | --- | --- |
| ministral-3:3b-cloud | 51/302 (16.9%) | 173/302 (57.3%) | 78/302 (25.8%) | 11/60 (18.3%) | 40/242 (16.5%) |
| ministral-3:8b-cloud | 74/302 (24.5%) | 55/302 (18.2%) | 173/302 (57.3%) | 9/60 (15.0%) | 65/242 (26.9%) |
| ministral-3:14b-cloud | 74/302 (24.5%) | 133/302 (44.0%) | 95/302 (31.5%) | 16/60 (26.7%) | 58/242 (24.0%) |
| nemotron-3-nano:30b-cloud | 116/302 (38.4%) | 151/302 (50.0%) | 35/302 (11.6%) | 26/60 (43.3%) | 90/242 (37.2%) |
| qwen3-coder-next:cloud | 45/302 (14.9%) | 161/302 (53.3%) | 96/302 (31.8%) | 27/60 (45.0%) | 18/242 (7.4%) |
| devstral-2:123b-cloud | 100/302 (33.1%) | 199/302 (65.9%) | 3/302 (1.0%) | 23/60 (38.3%) | 77/242 (31.8%) |
| gpt-oss:120b-cloud | 14/302 (4.6%) | 205/302 (67.9%) | 83/302 (27.5%) | 7/60 (11.7%) | 7/242 (2.9%) |
| glm-4.7:cloud | 31/302 (10.3%) | 258/302 (85.4%) | 13/302 (4.3%) | 29/60 (48.3%) | 2/242 (0.8%) |
| kimi-k2.5:cloud | 25/302 (8.3%) | 275/302 (91.1%) | 2/302 (0.7%) | 25/60 (41.7%) | 0/242 (0.0%) |
| gemini-3-flash-preview:cloud | 11/302 (3.6%) | 254/302 (84.1%) | 37/302 (12.3%) | 10/60 (16.7%) | 1/242 (0.4%) |

## Dataset: hypmol

| model | found=false | found=true | found_missing | found=false (impossible) | found=false (non-impossible) |
| --- | --- | --- | --- | --- | --- |
| ministral-3:3b-cloud | 16/151 (10.6%) | 93/151 (61.6%) | 42/151 (27.8%) | 5/30 (16.7%) | 11/121 (9.1%) |
| ministral-3:8b-cloud | 43/151 (28.5%) | 31/151 (20.5%) | 77/151 (51.0%) | 6/30 (20.0%) | 37/121 (30.6%) |
| ministral-3:14b-cloud | 44/151 (29.1%) | 59/151 (39.1%) | 48/151 (31.8%) | 7/30 (23.3%) | 37/121 (30.6%) |
| nemotron-3-nano:30b-cloud | 61/151 (40.4%) | 78/151 (51.7%) | 12/151 (7.9%) | 13/30 (43.3%) | 48/121 (39.7%) |
| qwen3-coder-next:cloud | 20/151 (13.2%) | 71/151 (47.0%) | 60/151 (39.7%) | 10/30 (33.3%) | 10/121 (8.3%) |
| devstral-2:123b-cloud | 42/151 (27.8%) | 106/151 (70.2%) | 3/151 (2.0%) | 5/30 (16.7%) | 37/121 (30.6%) |
| gpt-oss:120b-cloud | 2/151 (1.3%) | 101/151 (66.9%) | 48/151 (31.8%) | 1/30 (3.3%) | 1/121 (0.8%) |
| glm-4.7:cloud | 6/151 (4.0%) | 136/151 (90.1%) | 9/151 (6.0%) | 6/30 (20.0%) | 0/121 (0.0%) |
| kimi-k2.5:cloud | 5/151 (3.3%) | 144/151 (95.4%) | 2/151 (1.3%) | 5/30 (16.7%) | 0/121 (0.0%) |
| gemini-3-flash-preview:cloud | 0/151 (0.0%) | 135/151 (89.4%) | 16/151 (10.6%) | 0/30 (0.0%) | 0/121 (0.0%) |

### Examples where `nemotron-3-nano:30b-cloud` fails but `ministral-3:3b-cloud` does better

1. `id=0` `qtype=query_builder`
   Question: I am currently checking the database for chemical substance records and have a task for you. Please list chemical substances where the molecular formula contains "C3H5Cl3O2". What are their molar masses and chemical names?
   Nemotron: F1=0.000, exp=2,rec=0,matched=0, found=False, cite=True
   3B: F1=1.000, exp=2,rec=2,matched=2, found=True, cite=True
   Nemotron answer: **Answer:** No chemical substances were found with the molecular formula C3H5Cl3O2. **Explanation:** - I searched the knowledge graph for entities whose molecular formula contains “C3H5Cl3O2”. - An inspection of the matching URI (https://hypmol.net/Chemical...
   3B answer: The chemical substance with the molecular formula **C3H5Cl3O2** is **[Trichloroethane-4-carboxylic acid](http://localhost:3000/citation/flap-released-earnestly-norfolk)**. Its **molar mass is [155.44 g/mol](http://localhost:3000/citation/remuneration-wrappi...

2. `id=1` `qtype=query_builder`
   Question: I'm curious about the publication entries. Please list publications where the publication title contains "Coherence and Decoherence in". What are their primary authors and authors?
   Nemotron: F1=0.000, exp=21,rec=2,matched=0, found=True, cite=True
   3B: F1=0.923, exp=21,rec=18,matched=18, found=True, cite=True
   Nemotron answer: **Answer:** **Quantum Coherence and Decoherence in Nanoscale Silicon Devices** – Primary Author: [Amara Efe](), Authors: [Amara Efe](), Sarah Chen, **Quantum Coherence and Decoherence Dynamics in Molecular Spintronics** – Primary Author: [Sarah Chen](), Aut...
   3B answer: Here is the list of publications with titles containing **"Coherence and Decoherence in"** and their authors: 1. **[Quantum Coherence and Decoherence in Nanoscale Silicon Devices](http://localhost:3000/citation/curricular-textures-justifiable-exhibited)** -...

3. `id=16` `qtype=query_builder`
   Question: I am currently checking the database for chemical substance records and have a task for you. Show me chemical substances where the molecular formula contains "C6H15NO3P". What are their chemical names and molar masses?
   Nemotron: F1=0.000, exp=2,rec=0,matched=0, found=False, cite=True
   3B: F1=1.000, exp=2,rec=2,matched=2, found=True, cite=True
   Nemotron answer: The database search did not locate any chemical substances with the molecular formula containing **C6H15NO3P**, so no chemical names or molar masses could be retrieved. This may be due to query limitations or the absence of such records in the current datas...
   3B answer: The chemical substance with the molecular formula **C6H15NO3P** is **[Trimethyl-phosphazene oxide](http://localhost:3000/explain/livelihood-settings-analogue-nuisance)** with a molar mass of **[163.14 g/mol](http://localhost:3000/citation/vernacular-prop-te...

## Dataset: wiproflex

| model | found=false | found=true | found_missing | found=false (impossible) | found=false (non-impossible) |
| --- | --- | --- | --- | --- | --- |
| ministral-3:3b-cloud | 35/151 (23.2%) | 80/151 (53.0%) | 36/151 (23.8%) | 6/30 (20.0%) | 29/121 (24.0%) |
| ministral-3:8b-cloud | 31/151 (20.5%) | 24/151 (15.9%) | 96/151 (63.6%) | 3/30 (10.0%) | 28/121 (23.1%) |
| ministral-3:14b-cloud | 30/151 (19.9%) | 74/151 (49.0%) | 47/151 (31.1%) | 9/30 (30.0%) | 21/121 (17.4%) |
| nemotron-3-nano:30b-cloud | 55/151 (36.4%) | 73/151 (48.3%) | 23/151 (15.2%) | 13/30 (43.3%) | 42/121 (34.7%) |
| qwen3-coder-next:cloud | 25/151 (16.6%) | 90/151 (59.6%) | 36/151 (23.8%) | 17/30 (56.7%) | 8/121 (6.6%) |
| devstral-2:123b-cloud | 58/151 (38.4%) | 93/151 (61.6%) | 0/151 (0.0%) | 18/30 (60.0%) | 40/121 (33.1%) |
| gpt-oss:120b-cloud | 12/151 (7.9%) | 104/151 (68.9%) | 35/151 (23.2%) | 6/30 (20.0%) | 6/121 (5.0%) |
| glm-4.7:cloud | 25/151 (16.6%) | 122/151 (80.8%) | 4/151 (2.6%) | 23/30 (76.7%) | 2/121 (1.7%) |
| kimi-k2.5:cloud | 20/151 (13.2%) | 131/151 (86.8%) | 0/151 (0.0%) | 20/30 (66.7%) | 0/121 (0.0%) |
| gemini-3-flash-preview:cloud | 11/151 (7.3%) | 119/151 (78.8%) | 21/151 (13.9%) | 10/30 (33.3%) | 1/121 (0.8%) |

### Examples where `nemotron-3-nano:30b-cloud` fails but `ministral-3:3b-cloud` does better

1. `id=4` `qtype=query_builder`
   Question: I'm looking into the available information for metrologies. Please list metrologies where they measure wafer "wafer 1290253720". What are their measurement values?
   Nemotron: F1=0.000, exp=2,rec=0,matched=0, found=False, cite=True
   3B: F1=0.667, exp=2,rec=4,matched=2, found=True, cite=True
   Nemotron answer: I was unable to find any measurement values for wafer "wafer 1290253720". The knowledge graph exploration did not yield any associated measurement data for this specific wafer. <finalAnswer> No measurement values found for wafer "wafer 1290253720". </finalA...
   3B answer: **Answer**: [View Metrologies and Measurements for Wafer 1290253720](http://localhost:3000/explain/shanghai-secrecy-proposes-documentary) The metrologies and measurement values for **Wafer 1290253720** are as follows: 1. **Pre-Process Measurement**: - Metho...

2. `id=33` `qtype=query_builder`
   Question: I'm looking into the available information for conditioner pads. I'm looking for conditioner pads where the disk type contains "Brazed Diamond" and the disk diameter is 100.0. What are their grit sizes and manufacturers?
   Nemotron: F1=0.000, exp=2,rec=0,matched=0, found=None, cite=False
   3B: F1=0.667, exp=2,rec=1,matched=1, found=True, cite=True
   Nemotron answer: I’m unable to locate any conditioner‑pad entities in the knowledge graph that include a “Brazed Diamond” disk type or specify a 100 mm disk diameter. Consequently, there are no results I can cite for the requested grit‑size and manufacturer information. If ...
   3B answer: The conditioner pad with a **Brazed Diamond** disk type and a **100.0 diameter** is the **Conditioner Pad CP-150**. Its **grit size** is **150**, and the manufacturer is **[Enttegris](http://localhost:3000/citation/flooding-momentary-clutched-becomes)**.

3. `id=44` `qtype=query_builder`
   Question: I'm curious about the conditioner pad entries. Can you find conditioner pads where the disk type contains "Brazed Diamond" and the manufacturer contains "Entegris"? What are their disk diameters?
   Nemotron: F1=0.000, exp=1,rec=0,matched=0, found=True, cite=True
   3B: F1=0.500, exp=1,rec=3,matched=1, found=True, cite=True
   Nemotron answer: The conditioner pad meeting the specified criteria—disk type containing "Brazed Diamond" and manufacturer containing "Entegris"—is the **Conditioner Pad CP-150**. Its disk diameter is **100.0**. [Conditioner Pad](https://tucid/cpmo/instance/ConditionerPad_3)
   3B answer: There is a **Conditioner Pad** with a **Brazed Diamond disk type** and manufactured by **Entegris**, whose disk diameter is **[100.0 mm](http://localhost:3000/explain/rolled-indirect-trustworthy-doctor)**.
