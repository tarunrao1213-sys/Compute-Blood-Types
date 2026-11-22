# Repository for ss25.2.1/team150
This is the repository for you solution. You can modify this README file any way you see fit.

# Assignment 2.1: Compute Blood Types

This repository contains the data for assignment 2.1:
1. `example-problems` contains example problems,
2. `example-solutions` contains solutions to the example problems,
3. `problems` contains the problems you are supposed to solve,
4. `solution_check.py` is a script that you can use to compare solutions. Using a recent Python version (at least 3.8), you can run `python3 solution_check.py example-solutions path/to/your/example/solutions` to check if your solutions are correct.
5. `fresh.py` current python file for solving the assignment
6. `model.png` visualisation of the model generated for example-problems/problem-e-01.json
7. `results` directory containing the results for the solved problems, files will be overwritten by repeated runs

- runnable via `python fresh.py`
- dependencies 
  - standard-libraries json, os, sys, typing
  - [pgmpy](https://pypi.org/project/pgmpy/) `pip install pgmpy`
- no dynamic configuration available, paths to problems as well as result paths adjustable in the main guard (`if __name__ == "__main__"`)

# Solution Summary

This model computes blood type distributions in family trees using a Bayesian network. We model inheritance, blood type determination, and probabilistic test results using the pgmpy library.

## Model Contstruction
- The family tree is parsed to identify all people and their parent relationships.
- Each person is represented by three variables:
  - <name>_allele1
  - <name>_allele2
  - <name>_bloodtype
- People without parents receive alleles sampled from the country-specific distributions.
- Blood types are deterministically derived from allele pairs using co-dominance rules:
  - AA, AO -> A
  - BB, BO -> B
  - AB, BA -> AB
  - OO -> O

## Test Modeling
Test results are encoded as evidence or extra variables for more complex tests.
- bloodtype-test:
  - Always correct. Used as direct evidence on <name>_bloodtype.
- pair-bloodtype-test:
  - Two variables: test_i_result1, test_i_result2
  - depends on bloodtypes of two people
  - 80% chance that result-i is correct; 20% chance it is swapped
- mixed-bloodtype-test:
  - One variable: test_i_mixed_result
  - Deterministic mapping based on the union of blood type antigens of two people

## Inference
- After network construction, evidence from test results is set.
- Inference is done via Variable Elimination using pgmpy.
- For each query, the posterior over <name>_bloodtype is computed and formatted as a probability distribution over \[A, AB, B, O\].
