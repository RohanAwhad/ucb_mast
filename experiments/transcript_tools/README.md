# Transcript Tools

Generate tau2 transcripts with stable IDs so MAST can cite exact evidence locations.

## Generate one transcript

```bash
uv run python experiments/transcript_tools/generate_tau2_transcript_with_ids.py \
  /Users/rawhad/1_Projects/tau2-bench/data/simulations/<simulation_file>.json \
  1 \
  --output /Users/rawhad/1_Projects/tau2-bench/test_transcript_with_ids.txt
```

ID conventions in output:
- `m_####` = message ID
- `tc_####` = tool call ID
- `tr_####` = tool result ID
