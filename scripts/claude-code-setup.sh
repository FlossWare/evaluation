#!/bin/bash
# Add evaluation-ai integration to your CLAUDE.md
set -e

CLAUDE_MD="${CLAUDE_MD:-./CLAUDE.md}"

if [ ! -f "$CLAUDE_MD" ]; then
    echo "Creating $CLAUDE_MD"
    touch "$CLAUDE_MD"
fi

cat >> "$CLAUDE_MD" << 'EOF'

## Adversarial Evaluation (evaluation-ai)

This project uses [evaluation-ai](https://github.com/FlossWare/evaluation-ai) for adversarial verification and feedback loop detection.

**Install:** `pip install "git+https://github.com/FlossWare/evaluation-ai.git"`

**Key imports:**
```python
from evaluation_ai import adversarial_verify, AdversarialVerifier, SimpleFeedbackLoopDetector
```

**Usage patterns:**
- Decorator: `@adversarial_verify(backend, available_models, panel_size=3)`
- Verifier: `AdversarialVerifier(backend, available_models)` for manual verification
- Feedback loops: `SimpleFeedbackLoopDetector().analyze(execution_log)`
- Panel: `select_panel(available_models, panel_size)` for diverse model selection
- Zero external dependencies (stdlib only)
EOF

echo "Added evaluation-ai integration to $CLAUDE_MD"
