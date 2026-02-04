"""
HypMol Synthetic Knowledge Graph Generator

Generates a synthetic knowledge graph for organic chemistry synthesis using LLMs
and grammar-based decoding to ensure referential integrity.

The generation process has three phases:
1. Context Generation: Creates research projects and people
2. Entity Registry: Generates chemical substances
3. Dataset Generation: Creates experimental datasets with constrained references

Usage:
    python -m datasets.hypmol.generator
    
Environment Variables:
    LLM_MODEL: Model name (default: gpt-oss-120b)
    LLM_API_KEY: API key for the LLM service
    LLM_BASE_URL: Base URL for the LLM API
    LLM_TEMPERATURE: Temperature for generation (default: 1.0)
"""

import os
import sys
import uuid
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from openai import OpenAI
from rdflib import Graph, URIRef, Literal, RDF, Namespace, RDFS
from rdflib.namespace import XSD
from pynput import keyboard

from .models import (
    ProjectList, PersonList, ChemicalList, PublicationList,
    create_constrained_dataset_models, create_constrained_publication_models
)
from .prompts import (
    get_projects_prompt,
    get_people_prompt,
    get_chemicals_prompt,
    get_experiments_prompt,
    get_publications_prompt
)

# Load environment variables
load_dotenv()

# Namespaces
HYPMOL = Namespace("https://hypmol.net/")
SCHEMA = Namespace("http://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# Configuration
MODEL_NAME = os.getenv("LLM_MODEL", "zai-org/glm-4.7-flash")
API_KEY = os.getenv("LLM_API_KEY", "dummy")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))

# ==============================
# Domain Configuration
# ==============================
TOPIC = "Spin Chemistry and Chemistry Synthesis"
N_PROJECTS = 50          # Multiple research groups across institution
N_PEOPLE = 150           # Large research institution
N_PUBLICATIONS = 100     # Number of publications to generate
N_CHEMICALS = 350        # Large variety of substances for diverse experiments

# ==============================
# Batch Settings
# ==============================
N_PROJECTS_PER_BATCH = 3   # How many projects to ask for in one LLM call
N_PEOPLE_PER_BATCH = 3    # How many people to ask for in one LLM call
N_PUBLICATIONS_PER_BATCH = 3  # How many publications to ask for in one LLM call
N_CHEMICALS_PER_BATCH = 3  # How many chemicals to ask for in one LLM call
N_EXPERIMENTS_PER_BATCH = 3 # How many datasets to ask for in one LLM call

# ==============================
# Generation Settings
# ==============================
N_TOTAL_EXPERIMENTS = 350

if not API_KEY:
    print("Warning: LLM_API_KEY not found in environment variables.")

print("ENV: ", MODEL_NAME, API_KEY[0:5] + "...", BASE_URL)



client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def generate_uri(base: str) -> URIRef:
    """Generate a unique URI for a resource."""
    return URIRef(f"https://hypmol.net/{base}/{uuid.uuid4()}")

def clean_key(key: str) -> str:
    """Normalize string for dictionary lookup."""
    return key.strip().lower()

def _list_checkpoints(output_dir: Path) -> List[Path]:
    """Return checkpoint files sorted by timestamp desc (most recent first)."""
    pattern = re.compile(r"synthetic_graph_checkpoint_(\d{8}_\d{6})\.ttl$")
    checkpoints = []
    for path in output_dir.glob("synthetic_graph_checkpoint_*.ttl"):
        match = pattern.search(path.name)
        ts = None
        if match:
            try:
                ts = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                ts = None
        checkpoints.append((ts, path))
    checkpoints.sort(key=lambda item: item[0] or datetime.min, reverse=True)
    return [path for _, path in checkpoints]

def _select_checkpoint(output_dir: Path) -> Optional[Path]:
    """Prompt for a checkpoint to resume from (default: most recent)."""
    checkpoints = _list_checkpoints(output_dir)
    if not checkpoints:
        return None
    print("Found checkpoints:")
    for idx, path in enumerate(checkpoints, start=1):
        print(f"  {idx}) {path.name}")
    while True:
        choice = input("Resume from checkpoint [default: most recent (1), 'n' for new]: ").strip()
        if choice == "":
            return checkpoints[0]
        if choice.lower() in ("n", "new", "no"):
            return None
        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(checkpoints):
                return checkpoints[selected - 1]
        print("Invalid selection. Enter a number from the list or 'n'.")

class GraphGenerator:
    """
    Generates a synthetic knowledge graph for organic chemistry synthesis.
    
    The generator uses a three-phase approach:
    1. Generate context (projects, people)
    2. Generate entity registry (chemicals)
    3. Generate datasets with grammar-based decoding for referential integrity
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the graph generator.
        
        Args:
            output_dir: Directory to save the generated graph (default: current directory)
        """
        self.g = Graph()
        self.g.bind("hypmol", HYPMOL)
        self.g.bind("schema", SCHEMA)
        self.g.bind("dcterms", DCTERMS)
        
        self.output_dir = output_dir or Path.cwd()
        
        # Registries: Key (Name/Email/Code) -> URI
        self.project_registry: Dict[str, URIRef] = {}
        self.person_registry: Dict[str, URIRef] = {}
        self.chemical_registry: Dict[str, URIRef] = {}
        self.publication_count = 0
        self.dataset_count = 0
        
        # Save functionality
        self.save_requested = False
        self.retry_requested = False
        self.request_in_flight = False
        self._pressed_keys = set()
        self.listener = None
        self._start_keyboard_listener()

    def load_checkpoint(self, checkpoint_path: Path):
        """Load a checkpoint graph and rebuild registries."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        self.g.parse(checkpoint_path, format="turtle")

        self.project_registry = {}
        for subj in self.g.subjects(RDF.type, HYPMOL.Project):
            code = self.g.value(subj, HYPMOL.projectCode)
            if code:
                self.project_registry[clean_key(str(code))] = subj

        self.person_registry = {}
        for subj in self.g.subjects(RDF.type, SCHEMA.Person):
            email = self.g.value(subj, SCHEMA.email)
            if email:
                self.person_registry[clean_key(str(email))] = subj

        self.chemical_registry = {}
        for subj in self.g.subjects(RDF.type, HYPMOL.ChemicalSubstance):
            name = self.g.value(subj, HYPMOL.chemicalName)
            if name:
                self.chemical_registry[clean_key(str(name))] = subj

        self.publication_count = sum(1 for _ in self.g.subjects(RDF.type, HYPMOL.Publication))
        self.dataset_count = sum(1 for _ in self.g.subjects(RDF.type, SCHEMA.Dataset))

        print("Checkpoint loaded:")
        print(f"  - Projects: {len(self.project_registry)}")
        print(f"  - People: {len(self.person_registry)}")
        print(f"  - Chemicals: {len(self.chemical_registry)}")
        print(f"  - Publications: {self.publication_count}")
        print(f"  - Datasets: {self.dataset_count}")

    def _start_keyboard_listener(self):
        """Start a background keyboard listener for save/retry requests."""
        def _combo_active():
            return (
                any(k in self._pressed_keys for k in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r))
                and any(k in self._pressed_keys for k in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r))
            )

        def on_press(key):
            self._pressed_keys.add(key)
            try:
                if hasattr(key, 'char') and key.char in ('S', 's') and _combo_active():
                    self.save_requested = True
                    print("\n[Save requested - will save at next checkpoint]")
                elif hasattr(key, 'char') and key.char in ('R', 'r') and _combo_active():
                    self.retry_requested = True
                    print("\n[Retry requested - press Ctrl-C to interrupt and retry]")
            except AttributeError:
                pass

        def on_release(key):
            self._pressed_keys.discard(key)

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.daemon = True
        self.listener.start()
    
    def _stop_keyboard_listener(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()
    
    def save_current_state(self, suffix: str = ""):
        """
        Save the current state of the graph.
        
        Args:
            suffix: Optional suffix for the filename (default: timestamp)
        """
        if not suffix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"checkpoint_{timestamp}"
        
        output_path = self.output_dir / f"synthetic_graph_{suffix}.ttl"
        print(f"\n{'='*60}")
        print(f"SAVING CURRENT STATE")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Statistics:")
        print(f"  - Total triples: {len(self.g)}")
        print(f"  - Projects: {len(self.project_registry)}")
        print(f"  - People: {len(self.person_registry)}")
        print(f"  - Chemicals: {len(self.chemical_registry)}")
        
        self.g.serialize(output_path, format="turtle")
        print(f"✓ Saved successfully!")
        print(f"{'='*60}\n")
        
        # Reset the save flag
        self.save_requested = False
    
    def _check_and_save(self):
        """Check if save was requested and save if needed."""
        if self.save_requested:
            self.save_current_state()

    def _request_with_retry(self, label: str, request_fn):
        """Run a request with retry support via 'r' + Ctrl-C."""
        while True:
            try:
                self.request_in_flight = True
                result = request_fn()
            except KeyboardInterrupt:
                if self.retry_requested:
                    print(f"\nInterrupted: retrying {label} batch...")
                    continue
                raise
            finally:
                self.request_in_flight = False
            if self.retry_requested:
                self.retry_requested = False
                print(f"\nRetry requested - rerunning {label} batch...")
                continue
            return result

    def phase_1_context(self):
        """
        Phase 1: Generate the world context (projects and people).
        
        Creates research projects and researchers, linking them together.
        """
        print("--- PHASE 1: CONTEXT GENERATION ---")
        
        # 1. Projects
        print(f"Requesting Projects...")
        projects_generated = len(self.project_registry)
        if projects_generated:
            print(f"Found {projects_generated} existing projects.")
        projects_batch_num = 1

        while projects_generated < N_PROJECTS:
            current_batch_size = min(N_PROJECTS_PER_BATCH, N_PROJECTS - projects_generated)
            prompt_projects = get_projects_prompt(current_batch_size, TOPIC)
            print(f"Projects batch {projects_batch_num}: requesting {current_batch_size} projects...")

            batch_projects = self._request_with_retry(
                f"projects {projects_batch_num}",
                lambda: client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_projects}],
                    response_format=ProjectList,
                    temperature=TEMPERATURE
                ).choices[0].message.parsed.items
            )

            added = 0
            for p in batch_projects:
                key = clean_key(p.code)
                if key in self.project_registry:
                    continue
                uri = generate_uri("Project")
                self.project_registry[key] = uri
                self.g.add((uri, RDF.type, HYPMOL.Project))
                self.g.add((uri, HYPMOL.projectTitle, Literal(p.title)))
                self.g.add((uri, RDFS.label, Literal(p.title)))
                self.g.add((uri, HYPMOL.projectCode, Literal(p.code)))
                added += 1

            projects_generated += added
            print(f"Projects batch {projects_batch_num} done. Added {added}. Total: {projects_generated}/{N_PROJECTS}")
            self._check_and_save()
            projects_batch_num += 1

        print(f"Registered {projects_generated} projects.")
        self._check_and_save()

        # 2. People
        project_codes = list(self.project_registry.keys())
        print(f"Requesting People...")
        people_generated = len(self.person_registry)
        if people_generated:
            print(f"Found {people_generated} existing people.")
        people_batch_num = 1

        while people_generated < N_PEOPLE:
            current_batch_size = min(N_PEOPLE_PER_BATCH, N_PEOPLE - people_generated)
            prompt_people = get_people_prompt(current_batch_size, project_codes)
            print(f"People batch {people_batch_num}: requesting {current_batch_size} people...")

            batch_people = self._request_with_retry(
                f"people {people_batch_num}",
                lambda: client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_people}],
                    response_format=PersonList,
                    temperature=TEMPERATURE
                ).choices[0].message.parsed.items
            )

            added = 0
            for p in batch_people:
                key = clean_key(p.email)
                if key in self.person_registry:
                    continue
                uri = generate_uri("Person")
                self.person_registry[key] = uri
                self.g.add((uri, RDF.type, SCHEMA.Person))
                self.g.add((uri, SCHEMA.givenName, Literal(p.given_name)))
                self.g.add((uri, SCHEMA.familyName, Literal(p.family_name)))
                self.g.add((uri, RDFS.label, Literal(f"{p.given_name} {p.family_name}")))
                self.g.add((uri, SCHEMA.email, Literal(p.email)))
                self.g.add((uri, HYPMOL.position, Literal(p.position)))
                
                # Link to Project
                proj_key = clean_key(p.project_code_ref)
                if proj_key in self.project_registry:
                    self.g.add((uri, HYPMOL.involvedInProject, self.project_registry[proj_key]))
                added += 1

            people_generated += added
            print(f"People batch {people_batch_num} done. Added {added}. Total: {people_generated}/{N_PEOPLE}")
            self._check_and_save()
            people_batch_num += 1

        print(f"Registered {people_generated} people.")
        self._check_and_save()

        # 3. Publications
        available_authors = list(self.person_registry.keys())
        if not available_authors:
            print("Warning: No authors available for publications.")
            return

        print(f"Requesting Publications...")
        publications_generated = self.publication_count
        if publications_generated:
            print(f"Found {publications_generated} existing publications.")
        publications_batch_num = 1

        ConstrainedPublicationList, _ = create_constrained_publication_models(available_authors)

        while publications_generated < N_PUBLICATIONS:
            current_batch_size = min(N_PUBLICATIONS_PER_BATCH, N_PUBLICATIONS - publications_generated)
            prompt_publications = get_publications_prompt(current_batch_size, available_authors)
            print(f"Publications batch {publications_batch_num}: requesting {current_batch_size} publications...")

            batch_publications = self._request_with_retry(
                f"publications {publications_batch_num}",
                lambda: client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_publications}],
                    response_format=ConstrainedPublicationList,
                    temperature=TEMPERATURE
                ).choices[0].message.parsed.items
            )

            for pub in batch_publications:
                pub_uri = generate_uri("Publication")
                self.g.add((pub_uri, RDF.type, HYPMOL.Publication))
                self.g.add((pub_uri, HYPMOL.publicationTitle, Literal(pub.title)))
                self.g.add((pub_uri, RDFS.label, Literal(pub.title)))

                # Authors
                for author_email in pub.author_email_refs:
                    auth_key = clean_key(author_email)
                    if auth_key in self.person_registry:
                        self.g.add((pub_uri, HYPMOL.hasAuthor, self.person_registry[auth_key]))

                primary_email = pub.primary_author_email_ref or (pub.author_email_refs[0] if pub.author_email_refs else None)
                if primary_email:
                    primary_key = clean_key(primary_email)
                    if primary_key in self.person_registry:
                        self.g.add((pub_uri, HYPMOL.hasPrimaryAuthor, self.person_registry[primary_key]))

            publications_generated += len(batch_publications)
            print(f"Publications batch {publications_batch_num} done. Total: {publications_generated}/{N_PUBLICATIONS}")
            self._check_and_save()
            publications_batch_num += 1

        print(f"Registered {publications_generated} publications.")
        self._check_and_save()

    def phase_2_registry(self):
        """
        Phase 2: Generate the entity registry (chemical substances).
        
        Creates a registry of chemical substances that can be referenced
        in experiments.
        """
        print("\n--- PHASE 2: ENTITY REGISTRY ---")
        print(f"Requesting Chemicals...")
        chemicals_generated = len(self.chemical_registry)
        if chemicals_generated:
            print(f"Found {chemicals_generated} existing chemicals.")
        chemicals_batch_num = 1

        while chemicals_generated < N_CHEMICALS:
            current_batch_size = min(N_CHEMICALS_PER_BATCH, N_CHEMICALS - chemicals_generated)
            prompt_chem = get_chemicals_prompt(current_batch_size, TOPIC)
            print(f"Chemicals batch {chemicals_batch_num}: requesting {current_batch_size} chemicals...")

            batch_chemicals = self._request_with_retry(
                f"chemicals {chemicals_batch_num}",
                lambda: client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_chem}],
                    response_format=ChemicalList,
                    temperature=TEMPERATURE
                ).choices[0].message.parsed.items
            )

            added = 0
            for c in batch_chemicals:
                key = clean_key(c.chemical_name)
                if key in self.chemical_registry:
                    continue
                uri = generate_uri("ChemicalSubstance")
                self.chemical_registry[key] = uri
                self.g.add((uri, RDF.type, HYPMOL.ChemicalSubstance))
                self.g.add((uri, HYPMOL.chemicalName, Literal(c.chemical_name)))
                self.g.add((uri, RDFS.label, Literal(c.chemical_name)))
                self.g.add((uri, HYPMOL.molecularFormula, Literal(c.molecular_formula)))
                self.g.add((uri, HYPMOL.molarMass, Literal(c.molar_mass)))
                added += 1

            chemicals_generated += added
            print(f"Chemicals batch {chemicals_batch_num} done. Added {added}. Total: {chemicals_generated}/{N_CHEMICALS}")
            self._check_and_save()
            chemicals_batch_num += 1

        print(f"Registered {chemicals_generated} chemicals.")
        self._check_and_save()

    def phase_3_datasets(self):
        """
        Phase 3: Generate datasets with experiments using grammar-based decoding.
        
        Uses constrained Pydantic models to ensure the LLM only generates
        valid references to existing chemicals and authors. This eliminates
        the need for post-generation validation.
        """
        print("\n--- PHASE 3: DATASET GENERATION ---")
        
        available_chems = list(self.chemical_registry.keys())
        available_authors = list(self.person_registry.keys())
        
        if not available_chems or not available_authors:
            print("Error: Missing registry data.")
            return

        # Create constrained models with grammar-based decoding
        # This ensures the LLM can ONLY output valid chemical names and author emails
        print("Creating constrained models for grammar-based decoding...")
        ConstrainedDatasetList, _, _, _ = create_constrained_dataset_models(
            available_chemicals=available_chems,
            available_authors=available_authors
        )
        
        generated_count = self.dataset_count
        if generated_count:
            print(f"Found {generated_count} existing datasets.")
        batch_num = 1

        while generated_count < N_TOTAL_EXPERIMENTS:
            # Calculate remaining needed for this batch
            current_batch_size = min(N_EXPERIMENTS_PER_BATCH, N_TOTAL_EXPERIMENTS - generated_count)
            
            print(f"Batch {batch_num}: Requesting {current_batch_size} Datasets...")

            prompt_data = get_experiments_prompt(
                current_batch_size,
                TOPIC,
                available_chems,
                available_authors
            )

            try:
                datasets = self._request_with_retry(
                    f"datasets {batch_num}",
                    lambda: client.beta.chat.completions.parse(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt_data}],
                        response_format=ConstrainedDatasetList,  # Using constrained model!
                        temperature=TEMPERATURE
                    ).choices[0].message.parsed.items
                )
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                continue

            # All references are guaranteed to be valid due to grammar-based decoding
            for ds in datasets:
                ds_uri = generate_uri("Dataset")
                self.g.add((ds_uri, RDF.type, SCHEMA.Dataset))
                self.g.add((ds_uri, SCHEMA.name, Literal(ds.name)))
                self.g.add((ds_uri, RDFS.label, Literal(ds.name)))
                self.g.add((ds_uri, SCHEMA.description, Literal(ds.description)))
                if ds.identifier:
                    self.g.add((ds_uri, SCHEMA.identifier, Literal(ds.identifier)))
                if ds.license:
                    if ds.license.startswith("http"):
                        self.g.add((ds_uri, SCHEMA.license, Literal(ds.license, datatype=XSD.anyURI)))
                    else:
                        self.g.add((ds_uri, SCHEMA.license, Literal(ds.license)))
                if ds.date_published:
                    self.g.add((ds_uri, SCHEMA.datePublished, Literal(ds.date_published)))
                if ds.date_created:
                    self.g.add((ds_uri, SCHEMA.dateCreated, Literal(ds.date_created)))
                if ds.date_modified:
                    self.g.add((ds_uri, SCHEMA.dateModified, Literal(ds.date_modified)))
                if ds.encoding_format:
                    self.g.add((ds_uri, SCHEMA.encodingFormat, Literal(ds.encoding_format)))
                if ds.genre:
                    self.g.add((ds_uri, SCHEMA.genre, Literal(ds.genre)))
                if ds.keywords:
                    for kw in ds.keywords:
                        self.g.add((ds_uri, SCHEMA.keywords, Literal(kw)))
                if ds.text:
                    self.g.add((ds_uri, SCHEMA.text, Literal(ds.text)))
                if ds.url:
                    if ds.url.startswith("http"):
                        self.g.add((ds_uri, SCHEMA.url, Literal(ds.url, datatype=XSD.anyURI)))
                    else:
                        self.g.add((ds_uri, SCHEMA.url, Literal(ds.url)))
                
                # Link Author - guaranteed to exist in registry
                auth_key = clean_key(ds.author_email_ref)
                self.g.add((ds_uri, SCHEMA.author, self.person_registry[auth_key]))
                
                # Process the single experiment
                exp = ds.experiment
                exp_uri = generate_uri("Experiment")
                self.g.add((ds_uri, HYPMOL.hasExperiment, exp_uri))
                self.g.add((exp_uri, RDF.type, HYPMOL.Experiment))
                self.g.add((exp_uri, HYPMOL.experimentName, Literal(exp.name)))
                self.g.add((exp_uri, RDFS.label, Literal(exp.name)))
                self.g.add((exp_uri, HYPMOL.equipment, Literal(exp.equipment)))
                if exp.experiment_designation:
                    self.g.add((exp_uri, HYPMOL.experimentDesignation, Literal(exp.experiment_designation)))
                if exp.substitution_check:
                    self.g.add((exp_uri, HYPMOL.substitutionCheck, Literal(exp.substitution_check)))
                if exp.experiment_notes:
                    self.g.add((exp_uri, HYPMOL.experimentNotes, Literal(exp.experiment_notes)))
                if exp.reaction_equation:
                    self.g.add((exp_uri, HYPMOL.reactionEquation, Literal(exp.reaction_equation)))

                # Procedures
                for proc in exp.procedures:
                    proc_uri = generate_uri("ProcedureStep")
                    self.g.add((exp_uri, HYPMOL.hasProcedureStep, proc_uri))
                    self.g.add((proc_uri, RDF.type, HYPMOL.ProcedureStep))
                    self.g.add((proc_uri, HYPMOL.stepDescription, Literal(proc.step_description)))
                    self.g.add((proc_uri, RDFS.label, Literal(proc.step_description)))
                    self.g.add((proc_uri, HYPMOL.stepOrder, Literal(proc.step_order)))
                    if proc.observation:
                        self.g.add((proc_uri, HYPMOL.observation, Literal(proc.observation)))
                    if proc.procedure_date:
                        self.g.add((proc_uri, HYPMOL.procedureDate, Literal(proc.procedure_date)))

                # Measurements
                for meas in exp.measurements:
                    meas_uri = generate_uri("Measurement")
                    self.g.add((exp_uri, HYPMOL.hasMeasurement, meas_uri))
                    self.g.add((meas_uri, RDF.type, HYPMOL.Measurement))
                    self.g.add((meas_uri, HYPMOL.measurementMethod, Literal(meas.measurement_method)))
                    self.g.add((meas_uri, HYPMOL.measurementResult, Literal(meas.measurement_result)))
                    self.g.add((meas_uri, RDFS.label, Literal(f"{meas.measurement_method} measurement")))

                # Substance Usages - guaranteed to exist in registry
                for use in exp.substance_usages:
                    usage_key = clean_key(use.chemical_name_ref)
                    # No need to check - grammar-based decoding ensures it exists
                    usg_uri = generate_uri("SubstanceUsageInformation")
                    self.g.add((exp_uri, HYPMOL.hasUsageInformation, usg_uri))
                    self.g.add((usg_uri, RDF.type, HYPMOL.SubstanceUsageInformation))
                    self.g.add((usg_uri, HYPMOL.hasChemicalSubstance, self.chemical_registry[usage_key]))
                    self.g.add((usg_uri, RDFS.label, Literal(f"Usage of {use.chemical_name_ref}")))
                    
                    if use.used_mass: self.g.add((usg_uri, HYPMOL.usedMass, Literal(use.used_mass)))
                    if use.used_volume: self.g.add((usg_uri, HYPMOL.usedVolume, Literal(use.used_volume)))
                    if use.concentration: self.g.add((usg_uri, HYPMOL.concentration, Literal(use.concentration)))
                    if use.amount_of_substance: self.g.add((usg_uri, HYPMOL.amountOfSubstance, Literal(use.amount_of_substance)))
            
            count = len(datasets)
            generated_count += count
            print(f"Batch {batch_num} done. Generated {count} experiments. Total: {generated_count}/{N_TOTAL_EXPERIMENTS}")
            self._check_and_save()
            batch_num += 1

    def run(self, output_filename: str = "synthetic_graph.ttl"):
        """
        Run the complete generation pipeline and save the result.
        
        Args:
            output_filename: Name of the output file (default: synthetic_graph.ttl)
        """
        print("="*60)
        print("💡 TIP: Press Ctrl+Alt+S to save; Ctrl+Alt+R then Ctrl-C to interrupt + retry; Ctrl-C alone stops the run")
        print("="*60)
        print()
        
        try:
            self.phase_1_context()
            self.phase_2_registry()
            self.phase_3_datasets()
            
            output_path = self.output_dir / output_filename
            print(f"\nSerializing to '{output_path}'...")
            self.g.serialize(output_path, format="turtle")
            print("Done.")
            print(f"\nGenerated graph statistics:")
            print(f"  - Total triples: {len(self.g)}")
            print(f"  - Projects: {len(self.project_registry)}")
            print(f"  - People: {len(self.person_registry)}")
            print(f"  - Chemicals: {len(self.chemical_registry)}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: stopping run.")
        finally:
            # Stop the keyboard listener when done
            self._stop_keyboard_listener()

def main():
    """Main entry point for the generator."""
    # Determine output directory (datasets/hypmol/)
    output_dir = Path(__file__).parent

    checkpoint = _select_checkpoint(output_dir)
    gen = GraphGenerator(output_dir=output_dir)
    if checkpoint:
        gen.load_checkpoint(checkpoint)
    gen.run()

if __name__ == "__main__":
    main()
