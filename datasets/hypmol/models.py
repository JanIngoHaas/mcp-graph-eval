"""
Pydantic models for HypMol synthetic data generation.

This module defines the data models used to generate synthetic knowledge graphs
for organic chemistry synthesis. It includes:
- Context models (Projects, People)
- Registry models (Chemical Substances)
- Dataset models (Experiments, Procedures, Measurements)
- Constrained model factory for grammar-based decoding
"""

from typing import List, Optional, Literal, get_args
from datetime import date, datetime
from pydantic import BaseModel, Field, create_model

# ==========================================
# PHASE 1: THE WORLD (Context)
# ==========================================

class Project(BaseModel):
    """Research project in the knowledge graph."""
    title: str = Field(..., description="Full title of the research project")
    code: str = Field(..., description="Short identifier code (e.g., A1, B2)")

class Person(BaseModel):
    """Researcher or academic person."""
    given_name: str = Field(..., description="First name")
    family_name: str = Field(..., description="Last name")
    email: str = Field(..., description="Academic email address")
    position: str = Field(..., description="Academic position (e.g., PhD Student, PI)")
    # We reference the project by its code to link them
    project_code_ref: str = Field(..., description="The project code this person belongs to")

# ==========================================
# PHASE 2: THE Ingredients
# ==========================================

class ChemicalDefinition(BaseModel):
    """Chemical substance with its properties."""
    chemical_name: str = Field(..., description="Unique common name (e.g., 'Ethanol')")
    molecular_formula: str = Field(..., description="Empirical formula")
    molar_mass: float = Field(..., description="Molar mass in g/mol")

# ==========================================
# PHASE 3: THE Usage
# ==========================================

class SubstanceUsage(BaseModel):
    """Usage of a chemical substance in an experiment."""
    # Reference to an existing chemical in the pantry
    chemical_name_ref: str = Field(..., description="Must match a name from the available chemicals list")
    used_mass: Optional[float] = Field(None, description="Mass used in grams")
    used_volume: Optional[float] = Field(None, description="Volume used in ml")
    concentration: Optional[float] = Field(None, description="Concentration in mol/L if solution")
    amount_of_substance: Optional[float] = Field(None, description="Amount in moles")

class ProcedureStep(BaseModel):
    """A single step in an experimental procedure."""
    step_description: str = Field(..., description="Detailed text description of the action")
    step_order: int = Field(..., description="Sequential order: 1, 2, 3...")
    observation: Optional[str] = Field(None, description="Any observations made during this step")
    procedure_date: Optional[date] = Field(None, description="Date the step was performed")

class Measurement(BaseModel):
    """Measurement or analytical result from an experiment."""
    measurement_method: str = Field(..., description="Method used (e.g., NMR, IR)")
    measurement_result: str = Field(..., description="Result or value obtained")

class Experiment(BaseModel):
    """Complete experimental procedure with materials and measurements."""
    name: str = Field(..., description="Title of the experiment")
    equipment: str = Field(..., description="List of equipment used")
    experiment_designation: Optional[str] = Field(None, description="Designation or number identifying the experiment")
    substitution_check: Optional[str] = Field(None, description="Documentation of chemical substitution assessment")
    experiment_notes: Optional[str] = Field(None, description="Additional notes or observations about the experiment")
    procedures: List[ProcedureStep]
    substance_usages: List[SubstanceUsage]
    measurements: List[Measurement] = Field(default_factory=list)
    reaction_equation: Optional[str] = Field(None, description="Chemical reaction equation")

class Dataset(BaseModel):
    """Dataset containing experimental data."""
    name: str = Field(..., description="Name of the dataset")
    description: str = Field(..., description="Description of the dataset content")
    identifier: Optional[str] = Field(None, description="Dataset identifier")
    license: Optional[str] = Field(None, description="License URL for the dataset")
    date_published: Optional[datetime] = Field(None, description="Publication date (ISO 8601)")
    date_created: Optional[datetime] = Field(None, description="Creation date (ISO 8601)")
    date_modified: Optional[datetime] = Field(None, description="Last modified date (ISO 8601)")
    encoding_format: Optional[str] = Field(None, description="Media type (MIME type)")
    genre: Optional[str] = Field(None, description="Genre of the dataset")
    keywords: Optional[List[str]] = Field(None, description="Keywords for the dataset")
    text: Optional[str] = Field(None, description="Textual content/abstract")
    url: Optional[str] = Field(None, description="URL for the dataset")
    # Reference to an existing Person in the World
    author_email_ref: str = Field(..., description="Email of the author (must match a generated person)")
    experiment: Experiment  # Exactly one experiment per dataset

class Publication(BaseModel):
    """Scientific publication metadata."""
    title: str = Field(..., description="Title of the publication")
    author_email_refs: List[str] = Field(..., description="List of author emails")
    primary_author_email_ref: Optional[str] = Field(None, description="Primary author email")

# ==========================================
# WRAPPERS FOR GENERATION
# ==========================================

class ProjectList(BaseModel):
    """List of projects for batch generation."""
    items: List[Project]

class PersonList(BaseModel):
    """List of people for batch generation."""
    items: List[Person]

class ChemicalList(BaseModel):
    """List of chemicals for batch generation."""
    items: List[ChemicalDefinition]

class DatasetList(BaseModel):
    """List of datasets for batch generation."""
    items: List[Dataset]

class PublicationList(BaseModel):
    """List of publications for batch generation."""
    items: List[Publication]

# ==========================================
# CONSTRAINED MODEL FACTORY
# ==========================================

def create_constrained_dataset_models(
    available_chemicals: List[str],
    available_authors: List[str]
):
    """
    Creates Pydantic models with Literal type constraints for references.
    
    This enables grammar-based decoding where the LLM can ONLY output
    valid chemical names and author emails from the provided lists.
    The OpenAI structured outputs API will convert these Literal types
    to JSON Schema enums, forcing the model to choose only from valid values.
    
    Args:
        available_chemicals: List of valid chemical names (normalized keys)
        available_authors: List of valid author emails (normalized keys)
    
    Returns:
        Tuple of (ConstrainedDatasetList, ConstrainedDataset, 
                  ConstrainedExperiment, ConstrainedSubstanceUsage)
    
    Raises:
        ValueError: If either list is empty
    
    Example:
        >>> chemicals = ["ethanol", "benzene", "water"]
        >>> authors = ["alice@uni.edu", "bob@uni.edu"]
        >>> DatasetList, Dataset, Experiment, Usage = create_constrained_dataset_models(
        ...     chemicals, authors
        ... )
        >>> # Now the LLM can only generate datasets with these exact values
    """
    if not available_chemicals:
        raise ValueError("available_chemicals cannot be empty")
    if not available_authors:
        raise ValueError("available_authors cannot be empty")
    
    # Create Literal types from the available values
    # Note: Literal requires unpacking the tuple, not passing a list
    ChemicalNameLiteral = Literal[tuple(available_chemicals)]
    AuthorEmailLiteral = Literal[tuple(available_authors)]
    
    # Create constrained SubstanceUsage with enum for chemical_name_ref
    ConstrainedSubstanceUsage = create_model(
        'ConstrainedSubstanceUsage',
        chemical_name_ref=(ChemicalNameLiteral, Field(..., description="Must match a name from the available chemicals list")),
        used_mass=(Optional[float], Field(None, description="Mass used in grams")),
        used_volume=(Optional[float], Field(None, description="Volume used in ml")),
        concentration=(Optional[float], Field(None, description="Concentration in mol/L if solution")),
        amount_of_substance=(Optional[float], Field(None, description="Amount in moles")),
        __base__=BaseModel
    )
    
    # Create constrained Experiment using the constrained SubstanceUsage
    ConstrainedExperiment = create_model(
        'ConstrainedExperiment',
        name=(str, Field(..., description="Title of the experiment")),
        equipment=(str, Field(..., description="List of equipment used")),
        experiment_designation=(Optional[str], Field(None, description="Designation or number identifying the experiment")),
        substitution_check=(Optional[str], Field(None, description="Documentation of chemical substitution assessment")),
        experiment_notes=(Optional[str], Field(None, description="Additional notes or observations about the experiment")),
        procedures=(List[ProcedureStep], Field(...)),
        substance_usages=(List[ConstrainedSubstanceUsage], Field(...)),
        measurements=(List[Measurement], Field(default_factory=list)),
        reaction_equation=(Optional[str], Field(None, description="Chemical reaction equation")),
        __base__=BaseModel
    )
    
    # Create constrained Dataset with enum for author_email_ref
    ConstrainedDataset = create_model(
        'ConstrainedDataset',
        name=(str, Field(..., description="Name of the dataset")),
        description=(str, Field(..., description="Description of the dataset content")),
        identifier=(Optional[str], Field(None, description="Dataset identifier")),
        license=(Optional[str], Field(None, description="License URL for the dataset")),
        date_published=(Optional[datetime], Field(None, description="Publication date (ISO 8601)")),
        date_created=(Optional[datetime], Field(None, description="Creation date (ISO 8601)")),
        date_modified=(Optional[datetime], Field(None, description="Last modified date (ISO 8601)")),
        encoding_format=(Optional[str], Field(None, description="Media type (MIME type)")),
        genre=(Optional[str], Field(None, description="Genre of the dataset")),
        keywords=(Optional[List[str]], Field(None, description="Keywords for the dataset")),
        text=(Optional[str], Field(None, description="Textual content/abstract")),
        url=(Optional[str], Field(None, description="URL for the dataset")),
        author_email_ref=(AuthorEmailLiteral, Field(..., description="Email of the author (must match a generated person)")),
        experiment=(ConstrainedExperiment, Field(..., description="Exactly one experiment per dataset")),
        __base__=BaseModel
    )
    
    # Create constrained DatasetList
    ConstrainedDatasetList = create_model(
        'ConstrainedDatasetList',
        items=(List[ConstrainedDataset], Field(...)),
        __base__=BaseModel
    )
    
    return ConstrainedDatasetList, ConstrainedDataset, ConstrainedExperiment, ConstrainedSubstanceUsage

def create_constrained_publication_models(available_authors: List[str]):
    """
    Creates Pydantic models with Literal constraints for publication authors.

    Args:
        available_authors: List of valid author emails (normalized keys)

    Returns:
        Tuple of (ConstrainedPublicationList, ConstrainedPublication)
    """
    if not available_authors:
        raise ValueError("available_authors cannot be empty")

    AuthorEmailLiteral = Literal[tuple(available_authors)]

    ConstrainedPublication = create_model(
        'ConstrainedPublication',
        title=(str, Field(..., description="Title of the publication")),
        author_email_refs=(List[AuthorEmailLiteral], Field(..., description="List of author emails")),
        primary_author_email_ref=(Optional[AuthorEmailLiteral], Field(None, description="Primary author email")),
        __base__=BaseModel
    )

    ConstrainedPublicationList = create_model(
        'ConstrainedPublicationList',
        items=(List[ConstrainedPublication], Field(...)),
        __base__=BaseModel
    )

    return ConstrainedPublicationList, ConstrainedPublication
