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
from pathlib import Path
from typing import Dict
from datetime import datetime
from dotenv import load_dotenv
import threading

from openai import OpenAI
from rdflib import Graph, URIRef, Literal, RDF, Namespace, RDFS
from pynput import keyboard

from .models import (
    ProjectList, PersonList, ChemicalList,
    create_constrained_dataset_models
)
from .prompts import (
    get_projects_prompt,
    get_people_prompt,
    get_chemicals_prompt,
    get_experiments_prompt
)

# Load environment variables
load_dotenv()

# Namespaces
HYPMOL = Namespace("https://hypmol.net/")
SCHEMA = Namespace("http://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# Configuration
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-oss-120b")
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))

# Domain Configuration
TOPIC = "Organic Chemistry Synthesis"
N_PROJECTS = 15          # Multiple research groups across institution
N_PEOPLE = 80           # Large research institution
N_CHEMICALS = 100        # Large variety of substances for diverse experiments

# Generation Settings
N_TOTAL_EXPERIMENTS = 200   # ~5 experiments per person on average
N_EXPERIMENTS_PER_BATCH = 10   # How many to ask for in one LLM call (keep small for stability)

if not API_KEY:
    print("Warning: LLM_API_KEY not found in environment variables.")

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
        
        # Save functionality
        self.save_requested = False
        self.listener = None
        self._start_keyboard_listener()

    def _start_keyboard_listener(self):
        """Start a background keyboard listener for save requests."""
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == 'S':
                    self.save_requested = True
                    print("\n[Save requested - will save at next checkpoint]")
            except AttributeError:
                pass
        
        self.listener = keyboard.Listener(on_press=on_press)
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

    def phase_1_context(self):
        """
        Phase 1: Generate the world context (projects and people).
        
        Creates research projects and researchers, linking them together.
        """
        print("--- PHASE 1: CONTEXT GENERATION ---")
        
        # 1. Projects
        prompt_projects = get_projects_prompt(N_PROJECTS, TOPIC)
        print(f"Requesting Projects...")
        projects = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_projects}],
            response_format=ProjectList,
            temperature=TEMPERATURE
        ).choices[0].message.parsed.items

        for p in projects:
            uri = generate_uri("Project")
            self.project_registry[clean_key(p.code)] = uri
            self.g.add((uri, RDF.type, HYPMOL.Project))
            self.g.add((uri, HYPMOL.projectTitle, Literal(p.title)))
            self.g.add((uri, RDFS.label, Literal(p.title)))
            self.g.add((uri, HYPMOL.projectCode, Literal(p.code)))
        
        print(f"Registered {len(projects)} projects.")
        self._check_and_save()

        # 2. People
        project_codes = list(self.project_registry.keys())
        prompt_people = get_people_prompt(N_PEOPLE, project_codes)
        print(f"Requesting People...")
        people = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_people}],
            response_format=PersonList,
            temperature=TEMPERATURE
        ).choices[0].message.parsed.items

        for p in people:
            uri = generate_uri("Person")
            self.person_registry[clean_key(p.email)] = uri
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
        
        print(f"Registered {len(people)} people.")
        self._check_and_save()

    def phase_2_registry(self):
        """
        Phase 2: Generate the entity registry (chemical substances).
        
        Creates a registry of chemical substances that can be referenced
        in experiments.
        """
        print("\n--- PHASE 2: ENTITY REGISTRY ---")
        prompt_chem = get_chemicals_prompt(N_CHEMICALS, TOPIC)
        print(f"Requesting Chemicals...")
        chemicals = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_chem}],
            response_format=ChemicalList,
            temperature=TEMPERATURE
        ).choices[0].message.parsed.items

        for c in chemicals:
            uri = generate_uri("ChemicalSubstance")
            self.chemical_registry[clean_key(c.chemical_name)] = uri
            self.g.add((uri, RDF.type, HYPMOL.ChemicalSubstance))
            self.g.add((uri, HYPMOL.chemicalName, Literal(c.chemical_name)))
            self.g.add((uri, RDFS.label, Literal(c.chemical_name)))
            self.g.add((uri, HYPMOL.molecularFormula, Literal(c.molecular_formula)))
            self.g.add((uri, HYPMOL.molarMass, Literal(c.molar_mass)))
        
        print(f"Registered {len(chemicals)} chemicals.")
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
        
        generated_count = 0
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
                response = client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_data}],
                    response_format=ConstrainedDatasetList,  # Using constrained model!
                    temperature=TEMPERATURE
                )
                datasets = response.choices[0].message.parsed.items
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
                if exp.reaction_equation:
                    self.g.add((exp_uri, HYPMOL.reactionEquation, Literal(exp.reaction_equation)))

                # Procedures
                for proc in exp.procedures:
                    proc_uri = generate_uri("ProcedureStep")
                    self.g.add((exp_uri, HYPMOL.hasProcedureStep, proc_uri))
                    self.g.add((proc_uri, RDF.type, HYPMOL.ProcedureStep))
                    self.g.add((proc_uri, HYPMOL.stepDescription, Literal(proc.step_description)))
                    self.g.add((proc_uri, HYPMOL.stepOrder, Literal(proc.step_order)))
                    if proc.observation:
                        self.g.add((proc_uri, HYPMOL.observation, Literal(proc.observation)))
                    if proc.procedure_date:
                        self.g.add((proc_uri, HYPMOL.procedureDate, Literal(proc.procedure_date)))

                # Substance Usages - guaranteed to exist in registry
                for use in exp.substance_usages:
                    usage_key = clean_key(use.chemical_name_ref)
                    # No need to check - grammar-based decoding ensures it exists
                    usg_uri = generate_uri("SubstanceUsageInformation")
                    self.g.add((exp_uri, HYPMOL.hasUsageInformation, usg_uri))
                    self.g.add((usg_uri, RDF.type, HYPMOL.SubstanceUsageInformation))
                    self.g.add((usg_uri, HYPMOL.hasChemicalSubstance, self.chemical_registry[usage_key]))
                    
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
        print("💡 TIP: Press 's' at any time to save the current progress")
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
        finally:
            # Stop the keyboard listener when done
            self._stop_keyboard_listener()

def main():
    """Main entry point for the generator."""
    # Determine output directory (datasets/hypmol/)
    output_dir = Path(__file__).parent
    
    gen = GraphGenerator(output_dir=output_dir)
    gen.run()

if __name__ == "__main__":
    main()
