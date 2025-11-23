from pathlib import Path

import yaml

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

OPENEVOLVE_ROOT = Path(__file__).parent / "eplb"
OPENEVOLVE_CONFIG = yaml.safe_load(open(OPENEVOLVE_ROOT / "config.yaml", "r"))

def main(strategy="weighted"):
    """Run the evolution process."""
    # Configure local job execution
    job_config = LocalJobConfig(eval_program_path="evaluate.py")

    # Parent selection strategy
    if strategy == "uniform":
        # 1. Uniform from correct programs
        parent_config = dict(
            parent_selection_strategy="power_law",
            exploitation_alpha=0.0,
            exploitation_ratio=1.0,
        )
    elif strategy == "hill_climbing":
        # 2. Hill Climbing (Always from the Best)
        parent_config = dict(
            parent_selection_strategy="power_law",
            exploitation_alpha=100.0,
            exploitation_ratio=1.0,
        )
    elif strategy == "weighted":
        # 3. Weighted Prioritization
        parent_config = dict(
            parent_selection_strategy="weighted",
            parent_selection_lambda=10.0,
        )
    elif strategy == "power_law":
        # 4. Power-Law Prioritization
        parent_config = dict(
            parent_selection_strategy="power_law",
            exploitation_alpha=1.0,
            exploitation_ratio=0.2,
        )
    elif strategy == "power_law_high":
        # 5. Power-Law Prioritization (High Alpha)
        parent_config = dict(
            parent_selection_strategy="power_law",
            exploitation_alpha=2.0,
            exploitation_ratio=0.2,
        )
    elif strategy == "beam_search":
        # 6. Beam Search
        parent_config = dict(
            parent_selection_strategy="beam_search",
            num_beams=10,
        )

    # Database configuration
    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=40,
        # Inspiration parameters
        elite_selection_ratio=0.3,
        num_archive_inspirations=4,
        num_top_k_inspirations=2,
        # Island migration parameters
        migration_interval=10,
        migration_rate=0.1,  # Chance to migrate program to random island
        island_elitism=True,  # Island elite is protected from migration
        **parent_config,
    )

    # Task system message for LLM evolution
    search_task_sys_msg = OPENEVOLVE_CONFIG["prompt"]["system_message"]
    evo_config = EvolutionConfig(
        task_sys_msg=search_task_sys_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=[0.6, 0.3, 0.1],
        num_generations=400,
        max_parallel_jobs=3,
        max_patch_resamples=3,
        max_patch_attempts=3,
        job_type="local",
        language="python",
        llm_models=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "o4-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        llm_kwargs=dict(
            temperatures=[0.0, 0.5, 1.0],
            reasoning_efforts=["auto", "low", "medium", "high"],
            max_tokens=32768,
        ),
        meta_rec_interval=10,
        meta_llm_models=["gpt-5-nano"],
        meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
        embedding_model="text-embedding-3-small",
        code_embed_sim_threshold=0.995,
        novelty_llm_models=["gpt-5-nano"],
        novelty_llm_kwargs=dict(temperatures=[0.0], max_tokens=16384),
        llm_dynamic_selection="ucb1",
        llm_dynamic_selection_kwargs=dict(exploration_coef=1.0),
        init_program_path="eplb/initial_program.py",
        results_dir="results/eplb",
    )
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="weighted",
                        choices=["uniform", "hill_climbing", "weighted", "power_law", "power_law_high", "beam_search"],
                        help="Parent selection strategy for evolution")
    args = parser.parse_args()
    results_data = main(strategy=args.strategy)
