#!/usr/bin/env python3
"""
Smoke tests for experiment_v2.py - verifies core functions work without API calls.
"""

import random
import re
from experiment_v2 import (
    count_tokens,
    generate_filler_paragraph,
    generate_passcode,
    generate_distractor_sentence,
    build_context_by_tokens
)


def test_count_tokens():
    """Test token counting."""
    result = count_tokens("hello world")
    assert isinstance(result, int), "count_tokens should return an int"
    assert result > 0, "count_tokens should return positive value"
    print(f"✓ count_tokens('hello world') = {result}")


def test_generate_filler_paragraph():
    """Test filler paragraph generation."""
    result = generate_filler_paragraph(42)
    assert isinstance(result, str), "generate_filler_paragraph should return a string"
    assert len(result) > 0, "generate_filler_paragraph should return non-empty string"
    assert len(result) > 100, "generate_filler_paragraph should return substantial text"
    print(f"✓ generate_filler_paragraph(42) returned {len(result)} chars")


def test_generate_passcode():
    """Test passcode generation."""
    rng = random.Random(1)
    result = generate_passcode(rng)
    assert isinstance(result, str), "generate_passcode should return a string"

    # Check pattern LLLL-DDDD (4 uppercase letters, hyphen, 4 digits)
    pattern = r'^[A-Z]{4}-\d{4}$'
    assert re.match(pattern, result), f"Passcode '{result}' should match pattern LLLL-DDDD"
    print(f"✓ generate_passcode(Random(1)) = '{result}' (matches LLLL-DDDD)")


def test_generate_distractor_sentence():
    """Test distractor sentence generation."""
    rng = random.Random(1)
    real_passcode = "ABCD-1234"
    result = generate_distractor_sentence(rng, real_passcode)

    assert isinstance(result, str), "generate_distractor_sentence should return a string"
    assert len(result) > 0, "generate_distractor_sentence should return non-empty string"

    # Should contain a code pattern
    pattern = r'[A-Z]{4}-\d{4}'
    assert re.search(pattern, result), f"Distractor should contain a passcode pattern"

    # Should NOT contain the exact real passcode (it should be modified)
    fake_code = re.search(pattern, result).group()
    assert fake_code != real_passcode, f"Distractor code '{fake_code}' should differ from real '{real_passcode}'"

    print(f"✓ generate_distractor_sentence(Random(1), 'ABCD-1234') = '{result}'")
    print(f"  Fake code: {fake_code} (differs from ABCD-1234)")


def test_build_context_by_tokens():
    """Test context building with token accuracy."""
    needle = "SECRET PASSCODE: ABCD-1234"
    position = 0.5
    target_tokens = 2000
    real_passcode = "ABCD-1234"
    n_distractors = 2
    trial_seed = 42

    result = build_context_by_tokens(
        needle, position, target_tokens, real_passcode, n_distractors, trial_seed
    )

    assert isinstance(result, tuple), "build_context_by_tokens should return a tuple"
    assert len(result) == 2, "build_context_by_tokens should return 2-element tuple"

    context, actual_tokens = result

    assert isinstance(context, str), "First element should be a string"
    assert isinstance(actual_tokens, int), "Second element should be an int"

    assert needle in context, f"Context should contain the needle '{needle}'"

    # Check token count is roughly on target (within 20%)
    lower_bound = target_tokens * 0.8
    upper_bound = target_tokens * 1.2
    assert lower_bound <= actual_tokens <= upper_bound, \
        f"Token count {actual_tokens} should be within 20% of target {target_tokens}"

    print(f"✓ build_context_by_tokens(..., target=2000) returned:")
    print(f"  - Context length: {len(context)} chars")
    print(f"  - Actual tokens: {actual_tokens} (target: {target_tokens})")
    print(f"  - Contains needle: {needle in context}")
    print(f"  - Token accuracy: {actual_tokens/target_tokens*100:.1f}%")


def main():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING SMOKE TESTS FOR EXPERIMENT_V2.PY")
    print("=" * 70)
    print()

    tests = [
        ("Token counting", test_count_tokens),
        ("Filler paragraph generation", test_generate_filler_paragraph),
        ("Passcode generation", test_generate_passcode),
        ("Distractor sentence generation", test_generate_distractor_sentence),
        ("Context building by tokens", test_build_context_by_tokens),
    ]

    for test_name, test_func in tests:
        print(f"\n[Test: {test_name}]")
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            return False
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return False

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
