#!/usr/bin/env python3
"""
Lost in the Middle v2 — Hard version with tiktoken, distractors, multi-length sweep.

This version implements a more rigorous experiment with:
- Token-accurate context sizing via tiktoken
- Confusable distractor passcodes scattered throughout the filler
- Multi-dimensional sweep: context lengths (8k, 32k, 64k tokens) × positions
- Exponential backoff for rate limiting
- Cost tracking
- Comprehensive visualization
"""

import json
import os
import random
import string
import time
import math
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from openai import OpenAI, RateLimitError
import tiktoken

# Constants
MODEL = "gpt-4o-mini"
ENCODING = tiktoken.get_encoding("cl100k_base")
COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(ENCODING.encode(text))


def generate_filler_paragraph(seed: int) -> str:
    """
    Generate a unique filler paragraph using a local Random instance.

    Args:
        seed: Seed for the random number generator

    Returns:
        A paragraph of ~150-200 tokens
    """
    rng = random.Random(seed)

    # Expanded templates (10 different topics)
    templates = [
        # Weather
        "The weather patterns in {city} exhibit {pattern} characteristics throughout {season}. "
        "Temperature fluctuations typically range from {temp1} to {temp2} degrees Celsius, with "
        "precipitation averaging {precip} millimeters annually. The regional climate is heavily "
        "influenced by {factor1} and {factor2}, creating unique microclimates in surrounding areas. "
        "Meteorologists have documented these trends over {years} years of continuous observation. "
        "Local residents report that conditions have become increasingly {trend} in recent decades.",

        # Office supplies
        "Corporate procurement of office supplies including {item1}, {item2}, and {item3} represents "
        "a significant operational expense for most organizations. Industry surveys indicate that companies "
        "typically allocate approximately {cost} per employee annually for these essentials. Purchasing "
        "decisions are influenced by {factor1} considerations, with many firms opting for {frequency} "
        "bulk orders to optimize costs. Quality standards vary considerably, and {trend} solutions are "
        "becoming increasingly popular among forward-thinking businesses seeking to reduce environmental impact.",

        # Geography
        "The geographical region of {region} encompasses diverse terrain features including {feature1}, "
        "{feature2}, and extensive {feature3}. Spanning approximately {size} square kilometers, the area "
        "supports a population density of roughly {density} people per square kilometer. Notable landmarks "
        "such as {landmark} attract an estimated {visitors} visitors each year, contributing significantly "
        "to the local economy. The region's biodiversity is remarkable, with scientists cataloging {trend} "
        "species diversity patterns that make it a focal point for ecological research.",

        # Research
        "Contemporary research in {field} has focused extensively on investigating {topic}, yielding "
        "insights that challenge conventional understanding. Recent peer-reviewed studies published in "
        "leading journals have identified {finding}, with implications extending to {implication}. "
        "The research methodology employed {method} techniques to ensure rigor and reproducibility. "
        "Scholars emphasize that {aspect} requires further investigation, particularly regarding {trend} "
        "phenomena that emerged unexpectedly during experimental trials. Funding agencies have shown {interest} "
        "interest in supporting continued work in this domain.",

        # Manufacturing
        "Modern manufacturing protocols involve {step1}, followed by {step2}, culminating in {step3} "
        "before final quality assurance checks. The integrated production line operates {frequency}, "
        "maintaining throughput rates that have improved by {percent} compared to previous generation systems. "
        "Quality control specialists conduct {checks} inspections at critical junctures, utilizing {method} "
        "to detect potential defects. Industry analysts project that emerging {trend} technologies will "
        "further optimize efficiency metrics over the next fiscal period, reducing per-unit costs significantly.",

        # Historical events
        "Historical records from {period} document significant events that shaped {region} development "
        "trajectories. Archival sources indicate that {event} occurred during {season}, profoundly affecting "
        "subsequent political and economic structures. Contemporary scholars have analyzed these events using "
        "{method} approaches, revealing {finding} that previous historiography overlooked. The period witnessed "
        "{trend} changes in social organization, with consequences that resonated for {years} years. Primary "
        "documents preserved in {location} archives provide invaluable insights into the perspectives of {group}.",

        # Sports statistics
        "Athletic performance metrics from the {season} season reveal {trend} patterns across multiple "
        "competitive categories. Statistical analysis indicates that {team} achieved a {metric} rating of "
        "{value}, representing a {percent} improvement over their previous benchmark. Player performance "
        "data shows that {position} positions demonstrated the most significant {aspect} gains, with individual "
        "athletes recording {stat1} and {stat2} in key performance indicators. Coaching strategies emphasized "
        "{method} approaches, contributing to overall team dynamics and competitive outcomes in tournament play.",

        # Cooking recipes
        "Culinary preparation of {dish} requires {step1}, followed by {step2}, and finishing with {step3} "
        "to achieve optimal flavor profiles. Traditional recipes call for {ingredient1}, {ingredient2}, and "
        "{ingredient3} in precise proportions, though modern variations incorporate {ingredient4} for enhanced "
        "complexity. Cooking times typically range from {duration1} to {duration2} depending on {factor}, "
        "with temperature controls set to {temp} degrees. Professional chefs emphasize that {aspect} is critical "
        "for consistent results, and recommend {method} techniques to home cooks seeking restaurant-quality outcomes.",

        # Astronomy facts
        "Astronomical observations of {celestial} have yielded fascinating insights into {phenomenon} processes. "
        "Data collected from {instrument} telescopes over {years} years of monitoring reveal {finding} patterns "
        "that challenge existing theoretical models. The object exhibits {property1} characteristics with measured "
        "values of {value}, and displays {property2} variations on {frequency} cycles. Astrophysicists propose "
        "that {explanation} mechanisms account for these observations, though {aspect} remains poorly understood. "
        "Future observations using {method} spectroscopy may resolve outstanding questions about {topic} dynamics.",

        # Financial reports
        "Quarterly financial performance analysis for the {period} period demonstrates {trend} growth across "
        "key revenue segments. Total earnings reached {value} million, representing a {percent} change compared "
        "to the analogous period last fiscal year. Operational expenses were maintained at {cost} million, with "
        "{category} costs accounting for the largest proportion of the budget allocation. Market analysts note "
        "that {factor1} and {factor2} were primary drivers of performance outcomes. Management guidance projects "
        "{trend} trajectory for upcoming quarters, contingent on {condition} market conditions stabilizing."
    ]

    # Data pools
    cities = ["Portland", "Seattle", "Denver", "Austin", "Boston", "Phoenix", "Chicago", "Atlanta", "Toronto", "Vancouver"]
    patterns = ["seasonal", "variable", "stable", "cyclical", "dynamic", "predictable", "erratic"]
    seasons = ["spring", "summer", "autumn", "winter", "the monsoon season", "the dry season"]
    temps = ["5", "10", "15", "20", "25", "30", "35", "40"]
    precips = ["250", "500", "750", "1000", "1250", "1500", "2000"]
    factors = ["ocean currents", "mountain ranges", "latitude", "prevailing winds", "elevation", "proximity to water bodies"]
    years = ["25", "30", "40", "50", "75", "100"]
    trends = ["stable", "volatile", "concerning", "encouraging", "unpredictable"]

    items = ["pens", "paper clips", "staplers", "notebooks", "folders", "binders", "tape dispensers", "markers", "scissors", "desk organizers"]
    costs = ["$175", "$225", "$275", "$325", "$375", "€200", "€250", "€300"]
    frequencies = ["weekly", "biweekly", "monthly", "quarterly", "biannual"]

    regions = ["Northern Europe", "Southeast Asia", "Central America", "Eastern Africa", "Western Australia", "Patagonia", "Scandinavia"]
    features = ["mountain ranges", "river systems", "coastal plains", "plateaus", "valleys", "forests", "wetlands", "deserts"]
    sizes = ["45000", "75000", "125000", "200000", "350000", "500000"]
    densities = ["12", "25", "50", "85", "120", "200", "350"]
    landmarks = ["heritage sites", "national parks", "ancient monuments", "geological formations", "cultural centers"]
    visitors = ["fifty thousand", "two hundred thousand", "half a million", "one million", "three million"]

    fields = ["molecular biology", "computational chemistry", "theoretical physics", "cognitive psychology", "behavioral economics", "materials science"]
    topics = ["protein folding dynamics", "catalytic mechanisms", "quantum entanglement", "memory consolidation", "decision-making heuristics", "nanoscale properties"]
    findings = ["unexpected correlations", "nonlinear relationships", "emergent properties", "threshold effects", "synergistic interactions"]
    implications = ["therapeutic applications", "industrial processes", "technological innovation", "policy formulation", "theoretical frameworks"]
    aspects = ["mechanistic pathways", "temporal dynamics", "contextual dependencies", "individual variations", "environmental interactions"]
    methods = ["spectroscopic", "computational modeling", "longitudinal survey", "randomized controlled trial", "meta-analytical"]
    interests = ["substantial", "growing", "moderate", "increasing", "significant"]

    steps = ["material preparation", "precision cutting", "thermal treatment", "surface finishing", "component assembly", "quality verification"]
    percents = ["15%", "23%", "31%", "42%", "58%"]
    checks = ["rigorous", "systematic", "comprehensive", "statistical", "randomized"]

    periods = ["the early medieval period", "the Renaissance", "the Industrial Revolution", "the colonial era", "the 19th century", "the interwar period"]
    events = ["a major treaty signing", "a significant battle", "an economic crisis", "a political reform", "a cultural movement"]
    locations = ["national", "provincial", "university", "municipal", "royal"]
    groups = ["merchants", "artisans", "nobility", "clergy", "commoners", "scholars"]

    teams = ["the home team", "the visiting squad", "the defending champions", "the underdog competitors"]
    metrics = ["efficiency", "performance", "consistency", "impact", "versatility"]
    values = ["87.3", "92.1", "78.5", "95.8", "83.2"]
    positions = ["forward", "midfielder", "defensive", "goalkeeper", "striker"]
    stats = ["12.5 points per game", "8.3 assists per match", "72% completion rate", "4.2 defensive stops"]

    dishes = ["braised lamb", "pan-seared salmon", "roasted vegetables", "mushroom risotto", "beef tenderloin", "chicken marsala"]
    ingredients = ["olive oil", "garlic cloves", "fresh thyme", "white wine", "chicken stock", "butter", "shallots", "parsley", "lemon zest"]
    durations = ["25 minutes", "45 minutes", "1.5 hours", "2 hours", "35 minutes"]

    celestial = ["the exoplanet system", "the binary star", "the supernova remnant", "the galactic cluster", "the nebula"]
    phenomena = ["accretion", "stellar evolution", "gravitational", "electromagnetic", "thermodynamic"]
    instruments = ["radio", "infrared", "optical", "X-ray", "gamma-ray"]
    properties = ["luminosity", "spectral", "rotational", "magnetic field", "mass distribution"]
    explanations = ["tidal", "convective", "radiative", "magnetic reconnection", "shock wave"]

    categories = ["personnel", "infrastructure", "marketing", "research and development", "administrative"]
    conditions = ["macroeconomic", "regulatory", "competitive", "geopolitical", "technological"]

    # Select and fill template
    template = rng.choice(templates)

    # Fill with random selections
    filled = template.format(
        city=rng.choice(cities),
        pattern=rng.choice(patterns),
        season=rng.choice(seasons),
        temp1=rng.choice(temps),
        temp2=rng.choice(temps),
        precip=rng.choice(precips),
        factor=rng.choice(factors),
        factor1=rng.choice(factors),
        factor2=rng.choice(factors),
        years=rng.choice(years),
        trend=rng.choice(trends),
        item1=rng.choice(items),
        item2=rng.choice(items),
        item3=rng.choice(items),
        cost=rng.choice(costs),
        frequency=rng.choice(frequencies),
        region=rng.choice(regions),
        feature1=rng.choice(features),
        feature2=rng.choice(features),
        feature3=rng.choice(features),
        size=rng.choice(sizes),
        density=rng.choice(densities),
        landmark=rng.choice(landmarks),
        visitors=rng.choice(visitors),
        field=rng.choice(fields),
        topic=rng.choice(topics),
        finding=rng.choice(findings),
        implication=rng.choice(implications),
        aspect=rng.choice(aspects),
        method=rng.choice(methods),
        interest=rng.choice(interests),
        step1=rng.choice(steps),
        step2=rng.choice(steps),
        step3=rng.choice(steps),
        percent=rng.choice(percents),
        checks=rng.choice(checks),
        period=rng.choice(periods),
        event=rng.choice(events),
        location=rng.choice(locations),
        group=rng.choice(groups),
        team=rng.choice(teams),
        metric=rng.choice(metrics),
        value=rng.choice(values),
        position=rng.choice(positions),
        stat1=rng.choice(stats),
        stat2=rng.choice(stats),
        dish=rng.choice(dishes),
        ingredient1=rng.choice(ingredients),
        ingredient2=rng.choice(ingredients),
        ingredient3=rng.choice(ingredients),
        ingredient4=rng.choice(ingredients),
        duration1=rng.choice(durations),
        duration2=rng.choice(durations),
        temp=rng.choice(temps),
        celestial=rng.choice(celestial),
        phenomenon=rng.choice(phenomena),
        instrument=rng.choice(instruments),
        property1=rng.choice(properties),
        property2=rng.choice(properties),
        explanation=rng.choice(explanations),
        category=rng.choice(categories),
        condition=rng.choice(conditions),
    )

    return filled


def generate_passcode(rng: random.Random) -> str:
    """
    Generate a random passcode using the provided RNG instance.

    Args:
        rng: Random number generator instance

    Returns:
        Passcode in format LLLL-DDDD (4 letters, hyphen, 4 digits)
    """
    letters = ''.join(rng.choices(string.ascii_uppercase, k=4))
    digits = ''.join(rng.choices(string.digits, k=4))
    return f"{letters}-{digits}"


def generate_distractor_sentence(rng: random.Random, real_passcode: str) -> str:
    """
    Generate a confusable fake passcode sentence.

    Args:
        rng: Random number generator instance
        real_passcode: The real passcode to make confusable variants of

    Returns:
        A sentence containing a fake passcode
    """
    # Create a fake passcode by changing 1-2 characters
    parts = real_passcode.split('-')
    letters = list(parts[0])
    digits = list(parts[1])

    # Change one letter
    letter_idx = rng.randint(0, 3)
    letters[letter_idx] = rng.choice([c for c in string.ascii_uppercase if c != letters[letter_idx]])

    # Change one digit
    digit_idx = rng.randint(0, 3)
    digits[digit_idx] = rng.choice([c for c in string.digits if c != digits[digit_idx]])

    fake_passcode = f"{''.join(letters)}-{''.join(digits)}"

    # Templates for distractor sentences
    templates = [
        f"Note: the backup access code is {fake_passcode}.",
        f"The temporary override code was listed as {fake_passcode}.",
        f"An earlier version used the code {fake_passcode} before it was updated.",
        f"For reference, the legacy system code was {fake_passcode}.",
        f"The secondary authorization code is {fake_passcode}.",
    ]

    return rng.choice(templates)


def build_context_by_tokens(
    needle: str,
    position_fraction: float,
    target_tokens: int,
    real_passcode: str,
    n_distractors: int = 5,
    trial_seed: int = 0
) -> Tuple[str, int]:
    """
    Build a context with precise token count, including distractors.

    Args:
        needle: The needle sentence containing the real passcode
        position_fraction: Where to place needle (0.0-1.0)
        target_tokens: Target token count
        real_passcode: The real passcode (for generating distractors)
        n_distractors: Number of distractor sentences to add
        trial_seed: Seed for reproducibility

    Returns:
        Tuple of (context_string, actual_token_count)
    """
    rng = random.Random(trial_seed)

    # Calculate target position in tokens
    needle_tokens = count_tokens(needle)
    target_position_tokens = int(target_tokens * position_fraction)

    # Generate distractor sentences
    distractors = [generate_distractor_sentence(rng, real_passcode) for _ in range(n_distractors)]

    # Build paragraphs
    paragraphs = []
    current_tokens = 0
    para_idx = 0

    # Reserve tokens for needle and distractors
    distractor_tokens = sum(count_tokens(d) for d in distractors)
    available_tokens = target_tokens - needle_tokens - distractor_tokens - 50  # 50 token buffer

    # Generate filler paragraphs
    while current_tokens < available_tokens:
        para_seed = trial_seed * 10000 + para_idx
        para = generate_filler_paragraph(para_seed)
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens <= available_tokens:
            paragraphs.append(para)
            current_tokens += para_tokens
            para_idx += 1
        else:
            break

    # Insert distractors at random positions (not at needle position)
    distractor_positions = []
    for distractor in distractors:
        # Pick a random paragraph index (avoiding the needle position)
        while True:
            pos = rng.randint(0, len(paragraphs))
            # Calculate approximate position fraction
            tokens_before = sum(count_tokens(p) for p in paragraphs[:pos])
            pos_fraction = tokens_before / max(current_tokens, 1)
            # Ensure distractor is not too close to needle position (±5%)
            if abs(pos_fraction - position_fraction) > 0.05:
                distractor_positions.append(pos)
                break

    # Sort distractor positions to insert from end to beginning
    distractor_positions.sort(reverse=True)
    for i, pos in enumerate(distractor_positions):
        paragraphs.insert(pos, distractors[i])

    # Now insert the needle at the appropriate position
    # Recalculate positions with distractors included
    current_tokens = 0
    needle_insert_idx = 0

    for idx, para in enumerate(paragraphs):
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens >= target_position_tokens:
            needle_insert_idx = idx
            break
        current_tokens += para_tokens

    paragraphs.insert(needle_insert_idx, needle)

    # Join all paragraphs
    context = "\n\n".join(paragraphs)
    actual_tokens = count_tokens(context)

    return context, actual_tokens


def query_model(
    context: str,
    question: str,
    client: OpenAI,
    max_retries: int = 5
) -> Tuple[str, int, int]:
    """
    Query the model with exponential backoff on rate limits.

    Args:
        context: The context containing the needle
        question: The question to ask
        client: OpenAI client instance
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens)
    """
    system_prompt = "You are a precise assistant. Answer with ONLY the exact passcode value, nothing else. Format: LLLL-DDDD"
    user_prompt = f"Read the following document carefully and find the SECRET PASSCODE (not backup codes or temporary codes).\n\nDOCUMENT:\n{context}\n\nWhat is the SECRET PASSCODE? Answer with only the code:"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=20
            )

            response_text = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            return response_text, prompt_tokens, completion_tokens

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt * 5, 120)
                print(f"  Rate limit hit. Waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts due to rate limits: {e}")
                return "", 0, 0

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Error on attempt {attempt + 1}: {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return "", 0, 0

    return "", 0, 0


def exact_match(prediction: str, answer: str) -> bool:
    """
    Check if answer appears in prediction (case-insensitive).

    Args:
        prediction: Model's prediction
        answer: Correct answer

    Returns:
        True if answer is in prediction
    """
    return answer.upper() in prediction.upper()


def run_experiment(
    context_lengths_k: List[int],
    positions: List[float],
    trials_per_cell: int
) -> Dict:
    """
    Run the full experiment across context lengths and positions.

    Args:
        context_lengths_k: List of context lengths in thousands of tokens
        positions: List of position fractions (0.0-1.0)
        trials_per_cell: Number of trials per (length, position) combination

    Returns:
        Dictionary containing all results and metadata
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = {
        "model": MODEL,
        "context_lengths_k": context_lengths_k,
        "positions": positions,
        "trials_per_cell": trials_per_cell,
        "detailed_results": [],
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0
    }

    total_trials = len(context_lengths_k) * len(positions) * trials_per_cell
    trial_counter = 0

    print("=" * 80)
    print("Lost in the Middle v2: Enhanced Replication Experiment")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Context lengths: {context_lengths_k}k tokens")
    print(f"Positions: {positions}")
    print(f"Trials per cell: {trials_per_cell}")
    print(f"Total trials: {total_trials}")
    print("=" * 80)
    print()

    for ctx_k in context_lengths_k:
        print(f"\n{'=' * 80}")
        print(f"CONTEXT LENGTH: {ctx_k}k tokens")
        print(f"{'=' * 80}")

        for pos in positions:
            print(f"\n[Position {pos:.2f}]")
            print("-" * 80)

            for trial in range(trials_per_cell):
                trial_counter += 1

                # Generate unique trial seed
                trial_seed = ctx_k * 100000 + int(pos * 1000) * 100 + trial

                # Generate passcode with separate RNG
                passcode_rng = random.Random(trial_seed + 99999)
                passcode = generate_passcode(passcode_rng)

                needle = f"SECRET PASSCODE: {passcode}"

                # Build context
                context, actual_tokens = build_context_by_tokens(
                    needle=needle,
                    position_fraction=pos,
                    target_tokens=ctx_k * 1000,
                    real_passcode=passcode,
                    n_distractors=5,
                    trial_seed=trial_seed
                )

                # Query model
                question = "What is the SECRET PASSCODE in this document?"
                print(f"  Trial {trial + 1}/{trials_per_cell} (Overall: {trial_counter}/{total_trials}) "
                      f"- Passcode: {passcode}, Tokens: {actual_tokens}...", end=" ", flush=True)

                prediction, prompt_tokens, completion_tokens = query_model(context, question, client)

                # Check correctness
                is_correct = exact_match(prediction, passcode)

                print(f"{'✓' if is_correct else '✗'} (Predicted: {prediction})")

                # Accumulate tokens
                results["total_prompt_tokens"] += prompt_tokens
                results["total_completion_tokens"] += completion_tokens

                # Store result
                results["detailed_results"].append({
                    "context_length_k": ctx_k,
                    "position": pos,
                    "trial": trial,
                    "passcode": passcode,
                    "prediction": prediction,
                    "correct": is_correct,
                    "actual_tokens": actual_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                })

    return results


def compute_cost(total_prompt_tokens: int, total_completion_tokens: int) -> float:
    """
    Calculate total cost based on token usage.

    Args:
        total_prompt_tokens: Total input tokens
        total_completion_tokens: Total output tokens

    Returns:
        Total cost in USD
    """
    input_cost = (total_prompt_tokens / 1e6) * COST_PER_1M_INPUT
    output_cost = (total_completion_tokens / 1e6) * COST_PER_1M_OUTPUT
    return input_cost + output_cost


def print_summary(results: Dict):
    """Print comprehensive summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Build accuracy matrix
    ctx_lengths = results["context_lengths_k"]
    positions = results["positions"]

    # Calculate accuracies
    accuracy_matrix = defaultdict(lambda: defaultdict(list))

    for result in results["detailed_results"]:
        ctx_k = result["context_length_k"]
        pos = result["position"]
        correct = result["correct"]
        accuracy_matrix[ctx_k][pos].append(correct)

    # Print table
    print(f"\n{'Context':<10}", end="")
    for pos in positions:
        print(f"{pos:>8.2f}", end="")
    print()
    print("-" * (10 + 8 * len(positions)))

    for ctx_k in ctx_lengths:
        print(f"{ctx_k}k tokens", end="")
        for pos in positions:
            trials = accuracy_matrix[ctx_k][pos]
            if trials:
                accuracy = sum(trials) / len(trials) * 100
                print(f"{accuracy:>7.1f}%", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()

    # Print cost information
    print("\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)
    total_cost = compute_cost(results["total_prompt_tokens"], results["total_completion_tokens"])
    print(f"Total prompt tokens: {results['total_prompt_tokens']:,}")
    print(f"Total completion tokens: {results['total_completion_tokens']:,}")
    print(f"Total cost: ${total_cost:.2f}")
    print("=" * 80)


def plot_results(results: Dict):
    """Generate and save visualization plots."""
    ctx_lengths = results["context_lengths_k"]
    positions = results["positions"]

    # Aggregate accuracies
    accuracy_data = defaultdict(lambda: defaultdict(list))

    for result in results["detailed_results"]:
        ctx_k = result["context_length_k"]
        pos = result["position"]
        correct = result["correct"]
        accuracy_data[ctx_k][pos].append(correct)

    # Compute mean accuracies
    accuracies_by_ctx_pos = {}
    for ctx_k in ctx_lengths:
        accuracies_by_ctx_pos[ctx_k] = {}
        for pos in positions:
            trials = accuracy_data[ctx_k][pos]
            if trials:
                accuracies_by_ctx_pos[ctx_k][pos] = sum(trials) / len(trials) * 100
            else:
                accuracies_by_ctx_pos[ctx_k][pos] = 0

    # Plot 1: Accuracy vs Position (one line per context length)
    plt.figure(figsize=(12, 6))
    colors = cm.viridis(np.linspace(0, 1, len(ctx_lengths)))

    for idx, ctx_k in enumerate(ctx_lengths):
        accs = [accuracies_by_ctx_pos[ctx_k][pos] for pos in positions]
        plt.plot(positions, accs, marker='o', linewidth=2, markersize=8,
                 label=f"{ctx_k}k tokens", color=colors[idx])

    plt.xlabel("Needle Position (fraction of context)", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Lost in the Middle: Accuracy vs Needle Position (gpt-4o-mini)",
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("plot_by_position.png", dpi=300, bbox_inches='tight')
    print("\nPlot saved: plot_by_position.png")
    plt.close()

    # Plot 2: Accuracy vs Context Length (one line per position bucket)
    plt.figure(figsize=(12, 6))

    # Define position buckets
    position_buckets = {
        "Beginning (0.05-0.15)": [0.05, 0.15],
        "Middle (0.3-0.7)": [0.3, 0.5, 0.7],
        "End (0.85-0.95)": [0.85, 0.95]
    }

    bucket_colors = {'Beginning (0.05-0.15)': '#2E86AB',
                     'Middle (0.3-0.7)': '#A23B72',
                     'End (0.85-0.95)': '#F18F01'}

    for bucket_name, bucket_positions in position_buckets.items():
        bucket_accs = []
        for ctx_k in ctx_lengths:
            # Average over positions in this bucket
            accs = [accuracies_by_ctx_pos[ctx_k][pos] for pos in bucket_positions
                    if pos in accuracies_by_ctx_pos[ctx_k]]
            if accs:
                bucket_accs.append(sum(accs) / len(accs))
            else:
                bucket_accs.append(0)

        plt.plot(ctx_lengths, bucket_accs, marker='s', linewidth=2, markersize=10,
                 label=bucket_name, color=bucket_colors[bucket_name])

    plt.xlabel("Context Length (k tokens)", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("Accuracy Degrades with Context Length (gpt-4o-mini)",
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-5, 105)
    plt.tight_layout()
    plt.savefig("plot_by_length.png", dpi=300, bbox_inches='tight')
    print("Plot saved: plot_by_length.png")
    plt.close()


def main():
    """Main execution function."""
    # Experiment configuration
    CONTEXT_LENGTHS_K = [8, 32, 64]
    POSITIONS = [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
    TRIALS_PER_CELL = 15

    # Check API key
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY='sk-your-api-key-here'")
        sys.exit(1)

    # Estimate cost upfront
    total_trials = len(CONTEXT_LENGTHS_K) * len(POSITIONS) * TRIALS_PER_CELL
    avg_ctx_tokens = sum(CONTEXT_LENGTHS_K) / len(CONTEXT_LENGTHS_K) * 1000
    avg_tokens_per_trial = avg_ctx_tokens * 1.1  # 10% overhead

    estimated_input_cost = (total_trials * avg_tokens_per_trial / 1e6) * COST_PER_1M_INPUT
    estimated_output_cost = (total_trials * 20 / 1e6) * COST_PER_1M_OUTPUT
    estimated_total_cost = estimated_input_cost + estimated_output_cost

    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Context lengths: {CONTEXT_LENGTHS_K}k tokens")
    print(f"Positions: {POSITIONS}")
    print(f"Trials per cell: {TRIALS_PER_CELL}")
    print(f"Total trials: {total_trials}")
    print(f"\nEstimated cost: ${estimated_total_cost:.2f}")
    print("=" * 80)
    print("\nProceeding with experiment automatically...")
    print("=" * 80)

    # Run experiment
    results = run_experiment(CONTEXT_LENGTHS_K, POSITIONS, TRIALS_PER_CELL)

    # Save results
    output_file = "results_v2.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # Print summary
    print_summary(results)

    # Generate plots
    plot_results(results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
