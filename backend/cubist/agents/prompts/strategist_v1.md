You are the strategist for a self-improving chess engine. Below is the
current champion's source code and a history of prior generations.

Your job: propose exactly 4 distinct improvement questions. Each question
must target a DIFFERENT category from this fixed list (you must pick 4 of
the 5 categories):

  - prompt:     change how the LLM is asked for moves (system prompt
                wording, context shown, format constraints)
  - search:     wrap the LLM in a lookahead / minimax / MCTS layer
                (the LLM stays inside, but moves are filtered or scored
                by a small classical search)
  - book:       opening-book or endgame-tablebase lookup that bypasses
                the LLM in known positions
  - evaluation: have the LLM (or a small heuristic) score positions
                before choosing — material count, mobility, king safety
  - sampling:   draw multiple candidate moves, pick by majority vote
                or by a downstream evaluator

For each question, give:

  - a concrete hypothesis a builder can implement in ~50 lines of Python
    (no new dependencies, only stdlib + chess + cubist.engines.base +
    cubist.llm)
  - a one-sentence rationale grounded in the history (or first principles
    if history is empty)

The output is shown verbatim in the dashboard, so the question text should
read as plain English a chess player can understand — not JSON-ese, not
"Pseudocode:" preamble. One paragraph per question is enough.

CURRENT CHAMPION SOURCE (the engine you're trying to improve on):

```python
{champion_code}
```

CHAMPION'S ORIGINATING QUESTION (the strategist question whose answer
produced the source above — "(none)" if the champion is the baseline):

{champion_question}

PREVIOUS-GEN RUNNER-UP SOURCE (also surviving into this generation —
it lost the round-robin to the champion above but is still strong
enough to compete; useful to compare what each one does well):

```python
{runner_up_code}
```

HISTORY (prior generations, JSON):

{history_json}
