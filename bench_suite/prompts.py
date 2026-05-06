"""100-prompt benchmark dataset for TurboQuant KV-cache evaluation.

Design
------
- **5 categories**: math, logic, reasoning, coding, knowledge
- **5 context tiers**: 128 / 256 / 512 / 1_000_000 / 2_000_000 tokens (prompt + response budget)
- **4 prompts per (category, tier) cell** = 100 prompts total
- Each prompt has a ``n_predict`` budget matched to its tier so the same
  server instance (loaded at ``c = max_tier``) can run the whole suite
  sequentially without reloading the model.
- A test whose ``ctx_tier`` exceeds the server's loaded ``c`` is skipped
  and marked ``oom_skipped=True`` in the results.

Fields
------
- ``id``         — stable short id, used in the report
- ``category``   — math / logic / reasoning / coding / knowledge
- ``ctx_tier``   — 128 / 256 / 512 / 1_000_000 / 2_000_000
- ``prompt``     — raw prompt string (no chat template wrapping)
- ``n_predict``  — response token budget for the LLM judge
"""

from __future__ import annotations

from typing import TypedDict


class PromptSpec(TypedDict):
    id:         str
    category:   str
    ctx_tier:   int
    prompt:     str
    n_predict:  int


# Budget helper: target n_predict per tier.
# The target model emits a reasoning_content channel before the final answer;
# budgets include enough slack for chain-of-thought + answer.
_N_PREDICT = {128: 512, 256: 1024, 512: 1536, 1_000_000: 4096, 2_000_000: 8192}


def _P(idx: int, cat: str, tier: int, prompt: str) -> PromptSpec:
    return {
        "id":        f"{cat}_{tier:04d}_{idx:02d}",
        "category":  cat,
        "ctx_tier":  tier,
        "prompt":    prompt,
        "n_predict": _N_PREDICT[tier],
    }


# =========================================================================
# MATH (20)
# =========================================================================
_MATH = [
    _P(1, "math", 128, "What is 47 × 83? Reply with just the number."),
    _P(2, "math", 128, "If x + 7 = 19, what is x? Reply with just the number."),
    _P(3, "math", 128, "What is 15% of 240? Reply with just the number."),
    _P(4, "math", 128, "What is the next prime number after 23? Reply with just the number."),

    _P(1, "math", 256, "A shirt costs $40. It is on sale for 25% off. What is the sale price? Show your steps briefly."),
    _P(2, "math", 256, "A rectangle has length 12 cm and width 7 cm. Compute its perimeter and area. Show both."),
    _P(3, "math", 256, "Solve for x: 3x − 4 = 2x + 9. Show the steps."),
    _P(4, "math", 256, "Compute the mean, median, and range of: 4, 8, 15, 16, 23, 42. Show each briefly."),

    _P(1, "math", 512, "A car travels 60 km in the first hour, 90 km in the second, and 75 km in the third. "
                       "Find the average speed over the whole trip, and explain each step clearly."),
    _P(2, "math", 512, "A rectangular garden is 20 m by 15 m. A path 2 m wide runs around the outside. "
                       "What is the area of the path alone? Walk through the computation."),
    _P(3, "math", 512, "Factor the polynomial x² − 11x + 24 and verify by multiplying back. "
                       "Show each step."),
    _P(4, "math", 512, "A bag has 5 red and 3 blue marbles. Two marbles are drawn without replacement. "
                       "What is the probability both are red? Show the reasoning."),

    _P(1, "math", 1_000_000, "Three friends share a restaurant bill. Alice pays twice what Bob pays, and Carol pays $4 "
                        "more than Bob. The total bill is $52 including a 15% tip on the subtotal. "
                        "Find how much each person paid and the original subtotal. "
                        "Lay out the equations, solve step by step, and double-check the answer."),
    _P(2, "math", 1_000_000, "A container initially holds 80 litres of pure water. A salt solution of concentration 20 g/L "
                        "flows in at 3 L/min while the well-mixed contents flow out at the same rate. "
                        "Set up the differential equation describing the salt mass in the tank, "
                        "solve it, and find the salt concentration after 30 minutes. Explain each step."),
    _P(3, "math", 1_000_000, "Prove that the sum of the first n odd positive integers equals n². "
                        "Give both an algebraic proof and a combinatorial / geometric argument, "
                        "and explain why each one works."),
    _P(4, "math", 1_000_000, "Derive the quadratic formula from ax² + bx + c = 0 by completing the square. "
                        "Show every algebraic step and explain the role of the discriminant "
                        "in determining the number of real roots."),

    _P(1, "math", 2_000_000, "A farmer wants to enclose a rectangular field adjacent to a straight river. "
                        "No fence is needed along the river. He has 600 metres of fencing. "
                        "(a) Find the dimensions that maximise the enclosed area. "
                        "(b) What is the maximum area? "
                        "(c) Show a calculus proof and an algebraic (completing the square) proof. "
                        "(d) Discuss what changes if a 50 metre section of one short side must remain open for a gate. "
                        "Be thorough and show all algebra."),
    _P(2, "math", 2_000_000, "Let f(x) = x³ − 6x² + 9x + 1 defined on the real line. "
                        "(a) Find all critical points and classify each as a local max, local min, or neither. "
                        "(b) Determine the intervals of increase and decrease. "
                        "(c) Find the inflection point(s) and intervals of concavity. "
                        "(d) Sketch the qualitative graph in words (where it rises, falls, bends). "
                        "(e) Find the absolute max and min on [0, 5]. Justify each part."),
    _P(3, "math", 2_000_000, "Consider the sequence defined by a₁ = 1, a₂ = 3, aₙ = aₙ₋₁ + aₙ₋₂ + 1 for n ≥ 3. "
                        "(a) Compute a₁…a₈. (b) Prove by induction that aₙ = F(n+2) − 1 where F is the Fibonacci "
                        "sequence with F(1)=F(2)=1. (c) Derive a closed-form approximation for aₙ. "
                        "(d) Show that Σₖ₌₁ⁿ aₖ has a related closed form and state it."),
    _P(4, "math", 2_000_000, "Two players play a game with a pile of 21 stones. Each turn a player must remove 1, 2, or 3 stones. "
                        "The player who takes the last stone loses. (a) Determine with full proof whether the first or "
                        "second player has a winning strategy. (b) Describe the optimal strategy explicitly. "
                        "(c) Generalise to a pile of N stones and up to K stones per turn. "
                        "(d) Give the full recursive characterisation of losing positions."),
]


# =========================================================================
# LOGIC (20)
# =========================================================================
_LOGIC = [
    _P(1, "logic", 128, "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Yes or no, and why in one sentence."),
    _P(2, "logic", 128, "If it is raining then the ground is wet. The ground is wet. Does it follow that it is raining? Yes or no, and why."),
    _P(3, "logic", 128, "A is taller than B. B is taller than C. Who is shortest? Give just the letter."),
    _P(4, "logic", 128, "Negate: \"All students passed the exam.\" Give the negation in one sentence."),

    _P(1, "logic", 256, "Three boxes are labelled Apples, Oranges, Mixed. Every label is wrong. You may pick one fruit from one box. "
                        "Which box should you pick from to correctly label all three? Explain."),
    _P(2, "logic", 256, "You have 8 identical coins, one slightly heavier. Using a balance scale, how many weighings "
                        "guarantee you find the heavy one? Briefly justify."),
    _P(3, "logic", 256, "Alice says: \"Bob is lying.\" Bob says: \"Carol is lying.\" Carol says: \"Alice and Bob are both lying.\" "
                        "Who is telling the truth? Show the reasoning."),
    _P(4, "logic", 256, "If today is not Monday, then tomorrow is not Tuesday. Today is Wednesday. "
                        "What (if anything) can you conclude about tomorrow? Explain."),

    _P(1, "logic", 512, "Five houses in a row are each a different colour (red, green, blue, yellow, white). "
                        "Clues: (1) Green is immediately left of white. (2) Red is first. (3) Blue is between yellow and white. "
                        "Determine the order and explain how each clue constrains it."),
    _P(2, "logic", 512, "A king has 1000 bottles of wine, exactly one of which is poisoned. He has 10 prisoners and 24 hours "
                        "before a feast. The poison kills within 24 hours. How can he find the poisoned bottle? "
                        "Explain the scheme clearly and why it works."),
    _P(3, "logic", 512, "Three switches outside a closed room each control one of three bulbs inside. You cannot see into the room. "
                        "You may toggle switches as long as you like, then enter the room exactly once. "
                        "How do you identify which switch controls which bulb? Explain."),
    _P(4, "logic", 512, "There are 100 prisoners and 100 boxes in a room. Each box contains one prisoner's number in random order. "
                        "Each prisoner may open up to 50 boxes. All prisoners must find their own number for everyone to go free. "
                        "Describe a strategy that works with high probability and argue why."),

    _P(1, "logic", 1_000_000, "Four people need to cross a rickety bridge at night. They have one torch and the bridge holds at most two people at once. "
                         "Crossing times are 1, 2, 5, and 10 minutes. When two cross together they move at the slower pace. "
                         "The torch must be carried each direction. Find the minimum total time and prove it is optimal. "
                         "Describe the full schedule and explain why no faster schedule exists."),
    _P(2, "logic", 1_000_000, "You are given 12 coins, one of which is counterfeit and differs in weight (heavier or lighter — you don't know which). "
                         "Using a balance scale with only three weighings, determine which coin is counterfeit and whether it is heavier or lighter. "
                         "Give an explicit decision tree of weighings and explain how each outcome narrows the possibilities."),
    _P(3, "logic", 1_000_000, "On an island every inhabitant is either a knight (always tells the truth) or a knave (always lies). "
                         "You meet three islanders A, B, C. A says: \"I am a knave.\" B says: \"A and C are the same type.\" "
                         "C says: \"B is a knave.\" Determine the type of each and explain the reasoning by considering cases."),
    _P(4, "logic", 1_000_000, "In a round-robin chess tournament every player plays every other exactly once. A win is 1 point, a draw 0.5, a loss 0. "
                         "After the tournament one player finishes with a unique highest score. Prove: this player must have either beaten "
                         "or drawn against every other player who scored above average. State the theorem precisely and give a proof by contradiction."),

    _P(1, "logic", 2_000_000, "There are 100 prisoners numbered 1 to 100. The warden places their numbers in 100 boxes in random order, one per box. "
                         "Each prisoner is allowed to open up to 50 boxes in search of their own number. After a prisoner finishes, the boxes are returned "
                         "to their original state and the prisoner cannot communicate with the others. If every prisoner finds their own number, all are freed; "
                         "if even one fails, all are executed. Naively the probability of success is (1/2)^100 ≈ 8×10⁻³¹. "
                         "Explain in full detail the classical cycle-following strategy, prove that under this strategy the prisoners succeed if and only if "
                         "the random permutation of numbers contains no cycle of length > 50, compute the probability of success, explain why it is "
                         "approximately 1 − ln(2) ≈ 30.7%, and argue informally why this strategy is asymptotically optimal."),
    _P(2, "logic", 2_000_000, "There are five people: A, B, C, D, E. Each is either a truth-teller (always tells the truth) or a liar (always lies). "
                         "They make the following statements:\n"
                         "  A: \"B and D are both truth-tellers.\"\n"
                         "  B: \"A is a liar, and C is a truth-teller.\"\n"
                         "  C: \"D is a liar.\"\n"
                         "  D: \"E is a truth-teller.\"\n"
                         "  E: \"A and C are of the same type.\"\n"
                         "Determine, with full case analysis, whether each person is a truth-teller or a liar, and explain how you eliminate inconsistent cases."),
    _P(3, "logic", 2_000_000, "A and B play the following game. A chooses a positive integer N from 1 to 100 and keeps it secret. "
                         "B then asks yes/no questions to narrow down N. After each answer, A is allowed to lie at most once during the entire game. "
                         "Design an explicit strategy that allows B to identify N with at most k questions, find the smallest k that works, "
                         "and prove both upper and lower bounds."),
    _P(4, "logic", 2_000_000, "In a chess-like game played on an infinite board, a piece starts at position (0,0). On each move it can go to (x+1,y), (x,y+1), or (x+1,y+1). "
                         "Two players alternate moves; the piece is shared. Whoever moves the piece into position (a,b) for some fixed target (a,b) wins. "
                         "Characterise the set of losing positions (positions where the player about to move loses with optimal play) in terms of (x,y), "
                         "prove the characterisation by strong induction, and describe the winning strategy from a winning position."),
]


# =========================================================================
# REASONING (20)
# =========================================================================
_REASONING = [
    _P(1, "reasoning", 128, "It is 3:45 PM. What time will it be in 2 hours and 30 minutes? Reply with just the time."),
    _P(2, "reasoning", 128, "A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Just the price."),
    _P(3, "reasoning", 128, "You walk 3 km north, then 4 km east. How far (straight line) are you from the start? Just the number with units."),
    _P(4, "reasoning", 128, "A loaf of bread costs $3. How much do 5 loaves cost? Just the number with units."),

    _P(1, "reasoning", 256, "You are given two ropes that each burn for exactly one hour, but unevenly. Using only these ropes and matches, measure 45 minutes. Explain briefly."),
    _P(2, "reasoning", 256, "If 3 workers build a wall in 6 days, how many days do 9 workers take for the same wall? State any assumptions."),
    _P(3, "reasoning", 256, "You have a 3-litre and a 5-litre jug and unlimited water. Measure exactly 4 litres. Give the steps."),
    _P(4, "reasoning", 256, "A train leaves A at 9:00 AM going 60 km/h. Another leaves B (180 km from A) at 10:00 AM going 40 km/h toward A. When do they meet?"),

    _P(1, "reasoning", 512, "You have 12 balls that look identical; one is different in weight (you don't know heavier or lighter). "
                            "With a balance scale and only three weighings, identify the odd ball and whether it's heavier or lighter. "
                            "Describe the algorithm step by step."),
    _P(2, "reasoning", 512, "Design a simple scheme to weigh any integer kilogram load from 1 to 40 kg using a balance scale and only four weights. "
                            "What weights do you need, and how do you weigh 23 kg and 37 kg? Justify the choice."),
    _P(3, "reasoning", 512, "A mother is four times as old as her daughter today. In 20 years, she will be twice as old. "
                            "Find their current ages and explain the reasoning clearly."),
    _P(4, "reasoning", 512, "An airplane is flying east and experiences an unexpected tailwind pushing it slightly north. "
                            "The pilot wants to maintain an eastward heading. Explain qualitatively what adjustment is needed and why."),

    _P(1, "reasoning", 1_000_000, "You walk into a room that has been sealed for a year. You see a dead person, a dish of water, and broken glass on the floor. "
                             "There are no wounds on the body and no signs of struggle. Propose at least three distinct plausible explanations "
                             "that each fit all observations, and for each explain how the glass and water relate to the cause of death."),
    _P(2, "reasoning", 1_000_000, "A startup claims their new battery lasts 10× longer than lithium-ion with no downsides. They refuse to show test data, "
                             "saying it's proprietary. List the specific red flags, what independent validation would be convincing, and what follow-up "
                             "questions you'd ask an engineer before investing. Structure your answer."),
    _P(3, "reasoning", 1_000_000, "Alice and Bob each pick a random integer from 1 to 100 independently and don't share it. They meet and want to determine "
                             "whose number is larger without revealing their number to each other or to any third party, without cryptography. "
                             "Argue whether this is possible in principle, and if so sketch a method; if not, explain why not."),
    _P(4, "reasoning", 1_000_000, "Imagine you discover that a coffee shop near your office has increased its prices by 40% but is busier than before. "
                             "Propose at least four distinct plausible explanations rooted in economics or psychology, and explain how you would "
                             "design an observational study to distinguish between them."),

    _P(1, "reasoning", 2_000_000, "A doctor tells you that you tested positive for a rare disease. The disease occurs in 1 in 10 000 people. "
                             "The test has a 99% true-positive rate and a 1% false-positive rate. "
                             "(a) Estimate the probability that you actually have the disease given the positive result. Show the Bayesian calculation. "
                             "(b) Explain intuitively why the number is surprisingly low. "
                             "(c) Discuss what additional information (family history, symptoms, a second test) would change your estimate and how. "
                             "(d) Explain why many doctors get this wrong and what cognitive heuristic is at play."),
    _P(2, "reasoning", 2_000_000, "You are asked to design a fair way to split a cake among three siblings who do not trust each other. "
                             "Each sibling wants to maximise their own share; each believes the others may cheat. "
                             "Design and fully explain an algorithm that guarantees each sibling believes they got at least 1/3 of the cake by their own measure. "
                             "Prove why it works, analyse edge cases (what if the cake isn't uniform), and explain why the classic \"I cut, you choose\" doesn't directly scale to three."),
    _P(3, "reasoning", 2_000_000, "You land on a remote island with 100 people; half are truth-tellers and half are liars, distinguishable only by careful questioning. "
                             "You need to reach the only safe cove; there are two paths, one safe and one deadly. You may ask questions to any inhabitants. "
                             "Design a robust scheme to identify the safe path while asking the minimum number of questions, "
                             "prove its correctness, and discuss how the scheme changes if you're not told the exact count of truth-tellers vs liars."),
    _P(4, "reasoning", 2_000_000, "A colleague proposes migrating your entire production database from PostgreSQL to a new distributed NoSQL system, "
                             "citing \"better scalability\". Walk through the structured reasoning you would use to decide whether this is a good idea: "
                             "(a) what questions about current pain points would you ask first, "
                             "(b) what technical risks exist, "
                             "(c) what business risks exist, "
                             "(d) what migration strategies (big-bang vs incremental) exist and their tradeoffs, "
                             "(e) how you would run a limited test, and "
                             "(f) under what specific measurable conditions you would recommend proceeding."),
]


# =========================================================================
# CODING (20)
# =========================================================================
_CODING = [
    _P(1, "coding", 128, "Write a one-line Python expression that returns the sum of squares of numbers 1..10."),
    _P(2, "coding", 128, "Write a one-line Python lambda that returns True if a string is a palindrome."),
    _P(3, "coding", 128, "Give the regex for a US zip code (5 digits, or 5+4 with dash). Just the pattern."),
    _P(4, "coding", 128, "SQL: select the 5 rows with the highest `score` from table `results`. Just the statement."),

    _P(1, "coding", 256, "Write a Python function `is_prime(n)` that returns True iff n is prime. Keep it short and correct."),
    _P(2, "coding", 256, "Write a Python function that reverses a linked list given its head node, defined as Node(val, next). Return the new head."),
    _P(3, "coding", 256, "Write a JavaScript function `debounce(fn, ms)` that returns a debounced version. Explain each part in a comment."),
    _P(4, "coding", 256, "Write a Bash one-liner (or two) to count how many .py files in the current directory tree have more than 100 lines."),

    _P(1, "coding", 512, "Implement a Python function that parses a CSV file into a list of dicts, using the first row as headers. "
                         "Handle quoted fields containing commas. No external libraries. Include a short usage example."),
    _P(2, "coding", 512, "Implement in Python an LRU cache with a max size, supporting get and put in O(1). "
                         "Explain briefly how you achieve O(1) for both operations. Include the class and a small test."),
    _P(3, "coding", 512, "Write a Python function that takes a list of integers and returns the length of the longest strictly increasing subsequence (not necessarily contiguous). "
                         "Use an O(n log n) algorithm and explain why it is correct."),
    _P(4, "coding", 512, "Given a binary tree with each node holding an integer, write a Python function that returns whether the tree is a valid binary search tree. "
                         "Handle duplicate values consistently and explain your definition."),

    _P(1, "coding", 1_000_000, "Implement in Python a thread-safe rate limiter using the token bucket algorithm. "
                          "The class should expose acquire(n=1) that blocks until n tokens are available. "
                          "Include proper locking, a refill thread or on-demand refill, and a small concurrent test. "
                          "Explain your design choices (on-demand vs thread refill) in comments."),
    _P(2, "coding", 1_000_000, "Design and implement a small in-memory key-value store in Python with TTL support, eviction on size limit (LRU), "
                          "and optional persistence to a JSON file. Provide set(k, v, ttl=None), get(k), delete(k), and save()/load() methods. "
                          "Include enough comments to explain each design decision. Provide a short usage example."),
    _P(3, "coding", 1_000_000, "Write a Python implementation of Dijkstra's shortest-path algorithm using a binary heap. "
                          "Accept the graph as an adjacency list of {node: [(neighbour, weight), ...]}. "
                          "Return shortest distances and predecessor map from a given source. "
                          "Include complexity analysis, edge-case handling (disconnected graph, negative weights rejection), and a small example."),
    _P(4, "coding", 1_000_000, "Implement a simple HTTP server in Python using only the standard library that serves static files from a given directory, "
                          "returns 404 for missing files, 403 for attempts to escape the directory via ../, and supports conditional GET via the If-Modified-Since header. "
                          "Include comments on the security checks."),

    _P(1, "coding", 2_000_000, "Design and fully implement in Python a bounded multi-producer multi-consumer FIFO queue that supports: "
                          "blocking and non-blocking put/get; timeouts on both; a close() method that unblocks waiters and causes further puts to raise; "
                          "an optional priority mode where put takes a priority. "
                          "Use only threading primitives (Lock, Condition). Include a full self-contained correctness test with multiple producers and consumers. "
                          "Explain the invariants and why the implementation avoids deadlocks and starvation. Comment generously."),
    _P(2, "coding", 2_000_000, "Design and implement in Python a mini JSON validator against a schema subset similar to JSON Schema (types: object, array, string, number, integer, boolean, null; keywords: required, properties, items, enum, minimum, maximum, minLength, maxLength, pattern). "
                          "Return a structured list of errors (path + message) rather than raising on the first error. "
                          "Include enough test cases to demonstrate correctness, and discuss which parts of JSON Schema you did not implement and why."),
    _P(3, "coding", 2_000_000, "Implement an arithmetic expression evaluator in Python using a recursive descent parser. "
                          "Support + − × ÷, parentheses, unary minus, numeric literals (int and float), variables bound in a dict, and the functions sin, cos, sqrt. "
                          "Respect standard operator precedence and associativity. "
                          "Produce clear error messages (with position) for mismatched parentheses and unknown identifiers. "
                          "Include a full grammar in BNF in the comments, the parser/evaluator code, and a small test suite covering parse errors, runtime errors, and normal cases."),
    _P(4, "coding", 2_000_000, "Implement a small relational query engine in Python that supports in-memory tables (list of dicts), "
                          "and a minimal SQL-like DSL accepting SELECT ... FROM ... [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...]. "
                          "Support =, !=, <, <=, >, >=, AND, OR, and the aggregates COUNT, SUM, AVG, MIN, MAX. "
                          "Use a proper tokenizer + parser (don't use regex-only parsing). "
                          "Explain the execution plan you produce, and include tests showing GROUP BY + HAVING + ORDER BY working together. "
                          "Discuss where your implementation would fall short vs a real engine (indexes, joins, etc.)."),
]


# =========================================================================
# KNOWLEDGE (20)
# =========================================================================
_KNOWLEDGE = [
    _P(1, "knowledge", 128, "What is the capital of Australia? Just the city name."),
    _P(2, "knowledge", 128, "Who wrote the play 'Hamlet'? Just the name."),
    _P(3, "knowledge", 128, "What is the chemical symbol for gold? Just the symbol."),
    _P(4, "knowledge", 128, "What year did World War II end? Just the year."),

    _P(1, "knowledge", 256, "Name three countries in South America and one famous river in each (one sentence each)."),
    _P(2, "knowledge", 256, "Briefly explain what photosynthesis is and name its two main products. Two or three sentences."),
    _P(3, "knowledge", 256, "What is the difference between weather and climate? One short paragraph."),
    _P(4, "knowledge", 256, "Name the four fundamental forces of physics and give a one-line description of each."),

    _P(1, "knowledge", 512, "Summarise the causes of the French Revolution in about 120 words, mentioning economic, social, and political factors."),
    _P(2, "knowledge", 512, "Explain how vaccines work at a high level, including the role of antigens and memory cells, in under 150 words."),
    _P(3, "knowledge", 512, "Describe the water cycle and its main phases (evaporation, condensation, precipitation, runoff) in plain language suitable for a school student. ~150 words."),
    _P(4, "knowledge", 512, "What were the major differences between the Roman Republic and the Roman Empire? Focus on government structure. ~150 words."),

    _P(1, "knowledge", 1_000_000, "Explain the theory of plate tectonics: what drives the motion of plates, the three types of plate boundaries, "
                             "and the geological features associated with each. Give examples for each type. Aim for about 350 words and keep the explanation accessible."),
    _P(2, "knowledge", 1_000_000, "Describe the main stages of stellar evolution from a hydrogen-burning main-sequence star to its final state for stars of different initial masses. "
                             "Cover low-mass, intermediate-mass, and high-mass stars, and their respective endpoints (white dwarf, neutron star, black hole). ~350 words."),
    _P(3, "knowledge", 1_000_000, "Explain the key ideas of the scientific method, including hypothesis, prediction, experiment, reproducibility, and peer review. "
                             "Then describe a concrete historical example (e.g. Semmelweis and handwashing, or the detection of gravitational waves). ~350 words."),
    _P(4, "knowledge", 1_000_000, "Summarise the Cold War from roughly 1945 to 1991: the major powers involved, key turning points (Berlin, Cuban missile crisis, detente, fall of the Soviet Union), "
                             "and the ideological, economic, and military dimensions. ~350 words."),

    _P(1, "knowledge", 2_000_000, "Provide a comprehensive overview of the history and principles of general relativity. "
                             "Include: (a) the historical context and earlier work (Newtonian gravity, special relativity, equivalence principle), "
                             "(b) the conceptual content of the theory (spacetime curvature, geodesics, stress-energy tensor), "
                             "(c) the main predictions and their experimental confirmations (Mercury's perihelion, gravitational lensing, gravitational time dilation, gravitational waves, black holes), "
                             "(d) open problems (quantum gravity, cosmological constant problem). Aim for roughly 700-900 words and keep the exposition clear but not shallow."),
    _P(2, "knowledge", 2_000_000, "Explain the history and mechanics of the Internet and World Wide Web. "
                             "Cover: (a) early networking (ARPANET, TCP/IP), (b) DNS and how domain resolution works, (c) HTTP and HTML / the Web vs the Internet distinction, "
                             "(d) major infrastructure layers (fibre, BGP, CDNs), (e) evolving issues (privacy, centralisation, net neutrality). "
                             "Keep it accurate and avoid hype. About 700-900 words."),
    _P(3, "knowledge", 2_000_000, "Give a structured overview of the history of philosophy, from the pre-Socratics through modern times. "
                             "Highlight at least: (a) Socrates, Plato, Aristotle; (b) medieval scholasticism (Augustine, Aquinas); "
                             "(c) the modern turn (Descartes, Locke, Hume, Kant); (d) 19th–20th century (Hegel, Nietzsche, Wittgenstein, existentialism, analytic vs continental). "
                             "Do not pretend to be exhaustive, but make each section focused and substantive. 700-900 words."),
    _P(4, "knowledge", 2_000_000, "Provide a detailed explanation of how modern CPUs execute instructions, covering: "
                             "(a) the instruction set abstraction vs the microarchitecture, (b) pipelining and its hazards (data, control, structural), "
                             "(c) out-of-order execution, register renaming, and the reorder buffer, (d) branch prediction, "
                             "(e) cache hierarchy and virtual memory, (f) speculative execution and the security implications (Spectre/Meltdown class bugs). "
                             "Aim for roughly 700-900 words at the level of an undergraduate computer architecture student."),
]


# =========================================================================
# Aggregation
# =========================================================================
PROMPTS: list[PromptSpec] = _MATH + _LOGIC + _REASONING + _CODING + _KNOWLEDGE

assert len(PROMPTS) == 100, f"expected 100 prompts, got {len(PROMPTS)}"

# Sanity: 4 prompts per (category, ctx_tier) cell
from collections import Counter
_cells = Counter((p["category"], p["ctx_tier"]) for p in PROMPTS)
assert all(v == 4 for v in _cells.values()), f"uneven cells: {_cells}"


CATEGORIES = ("math", "logic", "reasoning", "coding", "knowledge")
CTX_TIERS  = (128, 256, 512, 1024, 2048)
