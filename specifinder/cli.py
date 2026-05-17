"""Command‑line entry point for the SDP Detection Pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .io import load_msa, map_positions, build_profiles
from .preprocessing import CleanseTransformer
from .modeling import MCAClusterFeatureSelector
from .visualization import (
    plot_cleanse_heatmaps,
    plot_pareto,
    plot_perceptual_map,
    generate_wordclouds,
    generate_logos,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Specificity‑Determining Positions in a protein MSA."
    )
    parser.add_argument("msa_path", type=Path, help="Path to the MSA file (FASTA format)")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to TSV metadata file for word clouds",
    )
    parser.add_argument("--plot", action="store_true", help="Generate all plots")
    parser.add_argument("--save", action="store_true", help="Save plots to ./output/")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> bool:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading MSA from %s", args.msa_path)
    raw_msa = load_msa(args.msa_path)
    positions_map = map_positions(raw_msa)

    pipeline = Pipeline(
        [
            ("cleanse", CleanseTransformer()),
            (
                "analysis",
                MCAClusterFeatureSelector(
                    cluster_method="single-linkage",
                    min_clusters=3,
                    rf_n_estimators=1000,
                    random_state=42,
                    top_n=3,
                ),
            ),
        ]
    )

    logger.info("Fitting pipeline…")
    pipeline.fit(raw_msa)

    cleanse_step = pipeline.named_steps["cleanse"]
    analysis_step = pipeline.named_steps["analysis"]

    # Plots
    if args.plot:
        plot_cleanse_heatmaps(
            cleanse_step.dirty_, cleanse_step.clean_, save=args.save, show=args.show
        )
        plot_pareto(analysis_step.column_importances_, save=args.save, show=args.show)
        plot_perceptual_map(
            analysis_step.mca_,
            analysis_step.unique_sequences_,
            analysis_step.coordinates_,
            analysis_step.labels_,
            analysis_step.selected_columns_,
            analysis_step.selected_feature_names_,
            save=args.save,
            show=args.show,
        )

    if args.metadata:
        generate_wordclouds(
            analysis_step.input_index_,
            analysis_step.unique_sequences_,
            analysis_step.labels_,
            metadata=args.metadata,
            save=args.save,
            show=args.show,
        )

    generate_logos(
        analysis_step.unique_sequences_,
        analysis_step.labels_,
        analysis_step.selected_columns_,
        save=args.save,
        show=args.show,
    )

    # Output profiles
    profiles = build_profiles(raw_msa, positions_map, analysis_step.selected_columns_)
    logger.info("Resulting profiles:\n%s", profiles.to_string())

    return True


if __name__ == "__main__":
    try:
        sys.exit(0 if main() else 1)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(2)