"""
============================================================
Filename: experiment.py
Author: Moritz Baumgart
Affiliation: University of Siegen, Intelligent Systems Group (ISG)
Date: December, 2024
============================================================

Description:
This file launches the reviewing part of AI-S for papers from openreview.net.
Papers have to specified in ./ai_reviewer/papers.csv, refer to README for more information.
============================================================
"""

import json
import logging
from pathlib import Path

from ai_scientist.perform_review import load_paper, perform_review

import pandas as pd
import openai
import requests


def main():

    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    papers_dir = Path(__file__).parent / "ai_reviewer"

    paper_df = pd.read_csv(papers_dir / "papers.csv")

    for _, current_paper in paper_df.iterrows():

        current_paper_id = str(current_paper["openreview_id"])

        current_dir = papers_dir / current_paper_id
        current_dir.mkdir(exist_ok=True, parents=True)
        try:
            paper_title = get_paper_title(current_paper_id)
        except Exception as _:
            # Use id if paper title could not be retrieved
            paper_title = current_paper_id

        logger.info(f"Starting review for {current_paper_id} ({paper_title})")

        pdf_pth = current_dir / f"{paper_title}.pdf"
        if not pdf_pth.exists():
            pdf_res = requests.get(f"https://openreview.net/pdf?id={current_paper_id}")
            if not pdf_res.ok:
                logger.error(f"Could not download PDF for {paper_title}, skipping!")
                continue
            with open(pdf_pth, "wb") as f:
                f.write(pdf_res.content)
            logger.info(f"Downloaded {current_paper_id} ({paper_title})")
        else:
            logger.info(f"Using existing file at {pdf_pth}")

        review_pth = current_dir / "review.txt"
        if review_pth.exists():
            logger.warning("Review already exists! Skipping...")
            continue

        try:
            paper_text = load_paper(pdf_pth)

            # Review parameters taken from official launch_scientist.py
            review = perform_review(
                paper_text,
                model="gpt-4o-2024-05-13",
                client=openai.OpenAI(),
                num_reflections=5,
                num_fs_examples=1,
                num_reviews_ensemble=5,
                temperature=0.1,
            )

            with open(review_pth, "w") as f:
                json.dump(review, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to perform review: {e}")
            logger.error(f"Error while performing review for {paper_title}, skipping!")
            continue


def get_paper_title(id: str) -> str:
    res = requests.get(f"https://api2.openreview.net/notes?id={id}")
    content = res.json()
    return content["notes"][0]["content"]["title"]["value"]


if __name__ == "__main__":
    main()
