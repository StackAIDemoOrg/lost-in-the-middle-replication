# Lost in the Middle: Replication Study

This repository contains a replication of the key finding from the paper ["Lost in the Middle: How Language Models Use Long Contexts"](https://arxiv.org/abs/2307.03172) by Liu et al. (2023).

## The Finding

The paper demonstrates that language models exhibit **U-shaped performance** when relevant information is placed at different positions within long contexts:

- **Best performance**: Information at the beginning or end of the context
- **Worst performance**: Information in the middle of the context

This has important implications for RAG (Retrieval-Augmented Generation) systems and prompt engineering, suggesting that critical information should be placed at the start or end of prompts rather than buried in the middle.

## Experimental Design

This replication study tests GPT-4o-mini's ability to retrieve a "needle in a haystack":

- **Context length**: ~16,000 characters of filler text
- **Needle**: A secret passcode in format `XXX-NNN` (e.g., "ABC-123")
- **Positions tested**: 5 locations (Beginning, Early-Middle, Middle, Late-Middle, End)
- **Trials per position**: 30 trials with unique passcodes
- **Metric**: Exact match accuracy (case-insensitive)

Each trial:
1. Generates a random passcode (e.g., "The secret passcode is QRT-847")
2. Embeds it at a specific position within ~16,000 characters of filler paragraphs
3. Asks the model: "Based on the following text, what is the secret passcode?"
4. Checks if the model's answer contains the correct passcode

## How to Run

### Prerequisites

- Python 3.7+
- OpenAI API key

### Installation and Execution

```bash
# Clone the repository
git clone https://github.com/StackAIDemoOrg/lost-in-the-middle-replication.git
cd lost-in-the-middle-replication

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Run the experiment
python experiment.py
```

The experiment will:
- Run 150 total trials (5 positions × 30 trials each)
- Print progress for each trial
- Display a summary table of accuracy by position
- Save detailed results to `results.json`
- Generate a plot saved as `accuracy_vs_position.png`

### Expected Runtime

- Approximately 10-15 minutes for all 150 trials
- Progress is printed for each trial

## Expected Outcome

If the "Lost in the Middle" effect holds for GPT-4o-mini, you should observe:

- **High accuracy** (~80-100%) when the passcode is at the **beginning** (position 0.1)
- **Lower accuracy** (~40-70%) when the passcode is in the **middle** (position 0.5)
- **High accuracy** (~80-100%) when the passcode is at the **end** (position 0.9)

The resulting plot should show a **U-shaped curve** where performance dips in the middle positions.

## Practical Notes

### Token Counting

- **Context size**: ~16,000 characters ≈ 4,000 tokens (at ~4 chars/token)
- **Total tokens per trial**: ~4,000 input + 20 output = ~4,020 tokens
- **Total tokens for experiment**: 150 trials × 4,020 ≈ 603,000 tokens

### Cost Estimation

Using GPT-4o-mini pricing (as of 2026):
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens

**Estimated cost for full experiment**:
- Input: 600,000 tokens × $0.15/1M ≈ $0.09
- Output: 3,000 tokens × $0.60/1M ≈ $0.002
- **Total**: ~$0.10 USD

### Prompt Format

The experiment uses this prompt structure:

```
System: Answer the question with only the exact value asked for. Be concise.

User: Based on the following text, what is the secret passcode?

[16,000 characters of context with embedded needle]

Answer with only the passcode value:
```

### Error Handling

- The script includes automatic retry logic (up to 3 attempts with 2-second delays)
- Failed trials will print error messages but won't crash the experiment
- Empty predictions are counted as incorrect

## Results Files

After running the experiment:

- **`results.json`**: Complete trial-by-trial data including:
  - Position, passcode, prediction, correctness for each trial
  - Aggregated accuracy by position
  - Context lengths

- **`accuracy_vs_position.png`**: Visual plot showing accuracy across positions

## Extending the Experiment

You can modify the experiment by editing `experiment.py`:

- **Change positions**: Edit the `positions` list (values between 0.0 and 1.0)
- **More trials**: Adjust `trials_per_position`
- **Different context length**: Change `target_chars` in `build_context()`
- **Different model**: Change `model="gpt-4o-mini"` to another model
- **Different needle format**: Modify the passcode generation and needle text

## References

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023).
Lost in the Middle: How Language Models Use Long Contexts.
*arXiv preprint arXiv:2307.03172*.

## License

MIT License - Feel free to use and modify for your own experiments.
