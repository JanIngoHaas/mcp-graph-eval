"""
Prompt templates for HypMol synthetic knowledge graph generation.

This module centralizes all LLM prompts used in the generation process,
making them easier to maintain, version, and improve.
"""

from typing import List


class PromptTemplates:
    """Collection of prompt templates for synthetic data generation."""
    
    @staticmethod
    def generate_projects(n_projects: int, topic: str) -> str:
        """
        Generate prompt for creating research projects.
        
        Args:
            n_projects: Number of projects to generate
            topic: Research domain/topic
            
        Returns:
            Formatted prompt string
        """
        return f"""Generate {n_projects} realistic and diverse research projects in the domain of {topic}.

Each project should:
- Have a unique, descriptive title that reflects cutting-edge research
- Include a short, memorable project code (e.g., "SYNTH-2024", "CATAL-X")
- Represent different sub-areas within {topic}
- Be plausible for a research institution or university

Focus on creating variety in research approaches and methodologies."""

    @staticmethod
    def generate_people(n_people: int, project_codes: List[str]) -> str:
        """
        Generate prompt for creating researchers.
        
        Args:
            n_people: Number of people to generate
            project_codes: List of available project codes to assign people to
            
        Returns:
            Formatted prompt string
        """
        codes_str = ", ".join(project_codes)
        return f"""Generate {n_people} realistic researchers with diverse backgrounds.

Each researcher should:
- Have a realistic given name and family name from various cultural backgrounds
- Have a unique, valid email address (use realistic domains like university.edu, research-institute.org)
- Have a position appropriate for research (e.g., PhD Student, Postdoc, Research Scientist, Professor, Lab Technician)
- Be assigned to ONE of these project codes: {codes_str}

Ensure diversity in:
- Names and cultural backgrounds
- Career stages (mix of students, early-career, and senior researchers)
- Distribution across projects (some projects may have more people than others)"""

    @staticmethod
    def generate_chemicals(n_chemicals: int, topic: str) -> str:
        """
        Generate prompt for creating FICTIONAL chemical substances.
        
        Args:
            n_chemicals: Number of chemicals to generate
            topic: Research domain/topic
            
        Returns:
            Formatted prompt string
        """
        return f"""Generate {n_chemicals} FICTIONAL but scientifically plausible chemical substances for {topic}.

IMPORTANT: These chemicals might not be real chemicals. Invent new chemicals that SOUND plausible but might not exist in reality.

Each fictional chemical should:
- Have a plausible-sounding IUPAC-style or common chemical name (invent new names that follow chemical naming conventions)
- Include a realistic-looking molecular formula (e.g., C8H15NO3, C12H22O5Cl2, etc.)
- Include a plausible molar mass in g/mol (calculate based on your invented molecular formula)

Requirements:
- Cover different sub-areas within {topic} (e.g., different functional groups, compound classes, applications)
- Include a mix of roles: solvents, reagents, catalysts, products, intermediates
- Ensure chemical plausibility (formulas should follow valence rules, molar masses should be calculated correctly)
- Include both simple molecules (5-10 atoms) and complex molecules (20-50 atoms)
- Make names sound professional and scientific, but ensure they are INVENTED

Examples of fictional chemical types to create:
- Novel solvents with invented names (e.g., "Hexyloxane", "Propanethiol-4-yl acetate")
- Synthetic catalysts (e.g., "Trimethyl-phosphazene oxide", "Cyclobutyl-palladium complex")
- Fictional reagents and starting materials
- Imaginary synthesis intermediates
- Fantasy products with plausible structures

REMEMBER: The goal is to create chemicals that sound real and follow chemical naming conventions, but are completely fictional. This prevents reliance on memorized chemical knowledge."""

    @staticmethod
    def generate_experiments(
        n_experiments: int,
        topic: str,
        available_chemicals: List[str],
        available_authors: List[str]
    ) -> str:
        """
        Generate prompt for creating experimental datasets.
        
        Args:
            n_experiments: Number of experiments to generate
            topic: Research domain/topic
            available_chemicals: List of chemical names that can be referenced
            available_authors: List of author emails that can be referenced
            
        Returns:
            Formatted prompt string
        """
        # Truncate lists if they're too long for the prompt
        max_display = 50
        chem_display = available_chemicals[:max_display]
        auth_display = available_authors[:max_display]
        
        chem_str = ", ".join(chem_display)
        if len(available_chemicals) > max_display:
            chem_str += f" ... and {len(available_chemicals) - max_display} more"
            
        auth_str = ", ".join(auth_display)
        if len(available_authors) > max_display:
            auth_str += f" ... and {len(available_authors) - max_display} more"
        
        return f"""Generate {n_experiments} realistic experimental datasets for {topic}.

NOTE: The chemicals you'll reference are FICTIONAL but plausible substances. Treat them as real chemicals in your experimental descriptions.

Each dataset should contain EXACTLY ONE experiment with:

1. DATASET METADATA:
   - A descriptive name for the dataset
   - A detailed description of what the experiment investigates
   - Author email (MUST be from this list): {auth_str}

2. EXPERIMENT DETAILS:
   - A specific experiment name
   - Equipment used (be specific: e.g., "Round-bottom flask, reflux condenser, magnetic stirrer")
   - Optional: A balanced chemical reaction equation if applicable (using the fictional chemicals)

3. PROCEDURE STEPS (3-7 steps):
   - Clear, sequential steps describing the experimental procedure
   - Each step should have:
     * step_order: Sequential number (1, 2, 3, ...)
     * step_description: Detailed description of what to do
     * observation: (optional - but recommended) What was observed during this step
     * procedure_date: (optional - but recommended) ISO format date (YYYY-MM-DD)

4. SUBSTANCE USAGES (2-10 substances):
   - Chemical name (MUST be from this list of fictional chemicals): {chem_str}
   - At least ONE of these measurements:
     * used_mass: Mass in grams (e.g., "5.2 g")
     * used_volume: Volume in mL or L (e.g., "100 mL")
     * concentration: Concentration (e.g., "0.5 M", "10% w/v")
     * amount_of_substance: Moles (e.g., "0.1 mol")

IMPORTANT CONSTRAINTS (enforced by schema):
- Author email MUST be exactly one from the provided list
- Chemical names MUST be exactly from the provided list (case-sensitive)
- Each dataset has exactly ONE experiment (not a list)

Make the experiments realistic, detailed, and scientifically plausible, treating the fictional chemicals as if they were real."""


# Convenience functions for backward compatibility
def get_projects_prompt(n_projects: int, topic: str) -> str:
    """Get the projects generation prompt."""
    return PromptTemplates.generate_projects(n_projects, topic)


def get_people_prompt(n_people: int, project_codes: List[str]) -> str:
    """Get the people generation prompt."""
    return PromptTemplates.generate_people(n_people, project_codes)


def get_chemicals_prompt(n_chemicals: int, topic: str) -> str:
    """Get the chemicals generation prompt."""
    return PromptTemplates.generate_chemicals(n_chemicals, topic)


def get_experiments_prompt(
    n_experiments: int,
    topic: str,
    available_chemicals: List[str],
    available_authors: List[str]
) -> str:
    """Get the experiments generation prompt."""
    return PromptTemplates.generate_experiments(
        n_experiments, topic, available_chemicals, available_authors
    )
