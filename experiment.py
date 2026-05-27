#!/usr/bin/env python3
"""
Lost in the Middle: Replication Study

This experiment replicates the finding that language models exhibit U-shaped
performance when relevant information is placed at different positions in
long contexts - performing best when evidence is at the beginning or end,
worst when it's in the middle.
"""

import json
import os
import random
import string
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from openai import OpenAI

# Get API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def generate_filler_paragraph(seed: int) -> str:
    """
    Generate a unique ~300-character filler paragraph using the seed.

    Args:
        seed: Random seed to ensure unique paragraph generation

    Returns:
        A string of approximately 300 characters with neutral content
    """
    random.seed(seed)

    # Templates for varied mundane content
    templates = [
        "The weather in {city} is typically {weather} during {season}. Temperatures range from {temp1} to {temp2} degrees. Precipitation levels are {precip} throughout the year. The climate is influenced by {factor}, which affects local conditions significantly.",
        "Office supplies such as {item1}, {item2}, and {item3} are essential for daily operations. Most businesses order these items {frequency}. The average cost is approximately {cost} per month. Quality varies by {factor}, so careful selection is important.",
        "The geography of {region} features {feature1} and {feature2}. The total area spans roughly {size} square kilometers. The population density is {density} people per square kilometer. Major landmarks include {landmark}, which attracts {visitors} annually.",
        "In the field of {field}, researchers have studied {topic} extensively. Recent findings suggest {finding}. This has implications for {implication}. Further research is needed to understand {aspect} more thoroughly.",
        "The manufacturing process involves {step1}, followed by {step2}, and finally {step3}. Quality control checks occur at {frequency} intervals. The entire process takes approximately {duration}. Efficiency has improved by {percent} over recent years.",
    ]

    # Data pools for filling templates
    cities = ["Portland", "Seattle", "Denver", "Austin", "Boston", "Phoenix", "Chicago", "Atlanta"]
    weathers = ["mild", "variable", "stable", "seasonal", "moderate", "temperate"]
    seasons = ["spring", "summer", "fall", "winter", "the transition months"]
    temps = ["15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70"]
    precips = ["moderate", "low", "high", "variable", "consistent", "seasonal"]
    factors = ["ocean currents", "mountain ranges", "latitude", "elevation", "wind patterns"]

    items = ["pens", "paper clips", "staplers", "notebooks", "folders", "binders", "tape", "markers"]
    frequencies = ["weekly", "biweekly", "monthly", "quarterly", "as needed"]
    costs = ["$150", "$200", "$250", "$300", "$350", "$400"]

    regions = ["Northern Europe", "Southeast Asia", "Central America", "Eastern Africa", "Western Australia"]
    features = ["mountain ranges", "river systems", "coastal plains", "plateaus", "valleys", "forests"]
    sizes = ["50,000", "75,000", "100,000", "150,000", "200,000", "250,000"]
    densities = ["low", "moderate", "high", "very high", "sparse"]
    landmarks = ["historic sites", "natural parks", "monuments", "museums", "bridges"]
    visitors = ["thousands", "tens of thousands", "hundreds of thousands", "millions"]

    fields = ["biology", "chemistry", "physics", "sociology", "economics", "psychology"]
    topics = ["cellular processes", "molecular interactions", "behavioral patterns", "market dynamics"]
    findings = ["correlation between variables", "unexpected patterns", "consistent trends"]
    implications = ["future research", "practical applications", "theoretical frameworks"]
    aspects = ["underlying mechanisms", "causal relationships", "contextual factors"]

    steps = ["preparation", "processing", "assembly", "inspection", "packaging", "treatment"]
    durations = ["2 hours", "4 hours", "6 hours", "8 hours", "one day", "two days"]
    percents = ["15%", "20%", "25%", "30%", "35%", "40%"]

    # Select and fill a template
    template = random.choice(templates)

    filled = template.format(
        city=random.choice(cities),
        weather=random.choice(weathers),
        season=random.choice(seasons),
        temp1=random.choice(temps),
        temp2=random.choice(temps),
        precip=random.choice(precips),
        factor=random.choice(factors),
        item1=random.choice(items),
        item2=random.choice(items),
        item3=random.choice(items),
        frequency=random.choice(frequencies),
        cost=random.choice(costs),
        region=random.choice(regions),
        feature1=random.choice(features),
        feature2=random.choice(features),
        size=random.choice(sizes),
        density=random.choice(densities),
        landmark=random.choice(landmarks),
        visitors=random.choice(visitors),
        field=random.choice(fields),
        topic=random.choice(topics),
        finding=random.choice(findings),
        implication=random.choice(implications),
        aspect=random.choice(aspects),
        step1=random.choice(steps),
        step2=random.choice(steps),
        step3=random.choice(steps),
        duration=random.choice(durations),
        percent=random.choice(percents),
    )

    return filled


def build_context(needle_sentence: str, position_fraction: float, target_chars: int = 16000) -> str:
    """
    Build a context of approximately target_chars characters with the needle inserted
    at the specified position.

    Args:
        needle_sentence: The critical information to embed (e.g., "The secret passcode is ABC-123")
        position_fraction: Where to place the needle (0.0 = start, 0.5 = middle, 1.0 = end)
        target_chars: Target context length in characters (default: 16000)

    Returns:
        A string containing filler paragraphs with the needle embedded at the specified position
    """
    # Calculate target position in characters
    target_position = int(target_chars * position_fraction)

    paragraphs_before = []
    paragraphs_after = []
    current_length = 0
    seed = 0

    # Build paragraphs before the needle
    while current_length < target_position:
        para = generate_filler_paragraph(seed)
        paragraphs_before.append(para)
        current_length += len(para) + 2  # +2 for newlines
        seed += 1

    # Build paragraphs after the needle
    current_length = len("\n\n".join(paragraphs_before)) + len(needle_sentence) + 4  # +4 for newlines around needle

    while current_length < target_chars:
        para = generate_filler_paragraph(seed)
        paragraphs_after.append(para)
        current_length += len(para) + 2
        seed += 1

    # Assemble the full context
    context_parts = []
    if paragraphs_before:
        context_parts.append("\n\n".join(paragraphs_before))
    context_parts.append(needle_sentence)
    if paragraphs_after:
        context_parts.append("\n\n".join(paragraphs_after))

    return "\n\n".join(context_parts)


def query_model(context: str, question: str, api_key: str, max_retries: int = 3) -> str:
    """
    Query the OpenAI model using the Responses API.

    Args:
        context: The context containing the needle
        question: The question to ask about the context
        api_key: OpenAI API key
        max_retries: Maximum number of retry attempts on failure

    Returns:
        The model's text response
    """
    client = OpenAI(api_key=api_key)

    # Construct the full prompt
    full_prompt = f"{question}\n\n{context}\n\nAnswer with only the passcode value:"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer the question with only the exact value asked for. Be concise."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                max_tokens=20
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error on attempt {attempt + 1}: {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return ""

    return ""


def exact_match(prediction: str, answer: str) -> bool:
    """
    Check if the answer appears in the prediction (case-insensitive).

    Args:
        prediction: The model's predicted answer
        answer: The correct answer to look for

    Returns:
        True if answer is found in prediction (case-insensitive), False otherwise
    """
    return answer.lower() in prediction.lower()


def generate_random_passcode() -> str:
    """Generate a random passcode in format XYZ-NNN."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
    digits = ''.join(random.choices(string.digits, k=3))
    return f"{letters}-{digits}"


def run_experiment() -> Dict:
    """
    Run the full Lost in the Middle experiment.

    Returns:
        Dictionary containing results data
    """
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    position_labels = ["Beginning", "Early-Middle", "Middle", "Late-Middle", "End"]
    trials_per_position = 30

    results = {
        "positions": positions,
        "position_labels": position_labels,
        "trials_per_position": trials_per_position,
        "detailed_results": [],
        "accuracy_by_position": {}
    }

    print("=" * 70)
    print("Lost in the Middle: Replication Experiment")
    print("=" * 70)
    print(f"Model: gpt-4o-mini")
    print(f"Positions: {position_labels}")
    print(f"Trials per position: {trials_per_position}")
    print(f"Total trials: {len(positions) * trials_per_position}")
    print("=" * 70)
    print()

    trial_counter = 0

    for pos_idx, (position, label) in enumerate(zip(positions, position_labels)):
        print(f"\n[Position {pos_idx + 1}/{len(positions)}] {label} (fraction: {position})")
        print("-" * 70)

        correct_count = 0

        for trial in range(trials_per_position):
            trial_counter += 1

            # Generate unique passcode for this trial
            passcode = generate_random_passcode()
            needle = f"The secret passcode is {passcode}"

            # Build context
            context = build_context(needle, position)

            # Query model
            question = "Based on the following text, what is the secret passcode?"
            print(f"  Trial {trial + 1}/{trials_per_position} (Overall: {trial_counter}/{len(positions) * trials_per_position}) - Passcode: {passcode}...", end=" ")

            prediction = query_model(context, question, OPENAI_API_KEY)

            # Check accuracy
            is_correct = exact_match(prediction, passcode)
            correct_count += is_correct

            print(f"{'✓' if is_correct else '✗'} (Predicted: {prediction})")

            # Store detailed result
            results["detailed_results"].append({
                "position": position,
                "position_label": label,
                "trial": trial,
                "passcode": passcode,
                "prediction": prediction,
                "correct": is_correct,
                "context_length": len(context)
            })

        # Calculate accuracy for this position
        accuracy = correct_count / trials_per_position
        results["accuracy_by_position"][label] = {
            "position": position,
            "accuracy": accuracy,
            "correct": correct_count,
            "total": trials_per_position
        }

        print(f"\n  Position accuracy: {accuracy:.2%} ({correct_count}/{trials_per_position})")

    return results


def print_summary(results: Dict):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Position':<20} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 70)

    for label in results["position_labels"]:
        data = results["accuracy_by_position"][label]
        accuracy_pct = data["accuracy"] * 100
        print(f"{label:<20} {accuracy_pct:>6.2f}%        {data['correct']:>3}/{data['total']:<3}")

    print("=" * 70)


def plot_results(results: Dict, output_file: str = "accuracy_vs_position.png"):
    """Generate and save a plot of accuracy vs position."""
    labels = results["position_labels"]
    accuracies = [results["accuracy_by_position"][label]["accuracy"] * 100 for label in labels]

    plt.figure(figsize=(10, 6))
    plt.plot(labels, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel("Evidence Position", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Lost in the Middle: U-shaped Accuracy vs Evidence Position", fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")


def main():
    """Main execution function."""
    # Check if API key is set
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY='sk-your-api-key-here'")
        return

    # Run experiment
    results = run_experiment()

    # Print summary
    print_summary(results)

    # Save results to JSON
    output_file = "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # Generate plot
    plot_results(results)

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
