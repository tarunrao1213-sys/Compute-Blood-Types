#!/usr/bin/env python3
"""
Blood Type Bayesian Network Solver
Solves blood type probability problems using Bayesian networks with pgmpy.

Usage: python blood_type_solver.py <input_json_file>
Output: Prints the solution JSON to stdout
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Set

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class BloodTypeSolver:
    def __init__(self):
        # Country-specific allele distributions
        self.allele_distributions = {
            "North Wumponia": {"A": 0.5, "B": 0.25, "O": 0.25},
            "South Wumponia": {"A": 0.15, "B": 0.55, "O": 0.3}
        }

        # Blood type mapping from allele pairs
        self.allele_to_bloodtype = {
            ("A", "A"): "A", ("A", "O"): "A", ("O", "A"): "A",
            ("B", "B"): "B", ("B", "O"): "B", ("O", "B"): "B",
            ("A", "B"): "AB", ("B", "A"): "AB",
            ("O", "O"): "O"
        }

        self.alleles = ["A", "B", "O"]
        self.blood_types = ["A", "AB", "B", "O"]

    def solve_problem(self, problem_data: Dict) -> List[Dict]:
        """Solve a blood type problem and return the solution."""
        family_tree = problem_data["family-tree"]
        country = problem_data["country"]
        test_results = problem_data["test-results"]
        queries = problem_data["queries"]

        # Extract all people from the family tree
        people = self._extract_people(family_tree)

        # Build the Bayesian network
        model = self._build_network(people, family_tree, country, test_results)
        # model.to_graphviz().draw('model.png', prog='dot')
        # Add evidence from test results
        evidence = self._process_test_results(test_results, people)

        # Perform inference for each query
        inference = VariableElimination(model)
        solutions = []

        for query in queries:
            if query["type"] == "bloodtype":
                person = query["person"]
                bloodtype_var = f"{person}_bloodtype"

                # Query the probability distribution
                try:
                    result = inference.query(variables=[bloodtype_var], evidence=evidence)
                    prob_dist = result.values.flatten()

                    # Map to blood type probabilities (order: A, AB, B, O)
                    distribution = {
                        "A": float(prob_dist[0]),
                        "AB": float(prob_dist[1]),
                        "B": float(prob_dist[2]),
                        "O": float(prob_dist[3])
                    }

                    solutions.append({
                        "type": "bloodtype",
                        "person": person,
                        "distribution": distribution
                    })
                except Exception as e:
                    print(f"Error querying {person}: {e}", file=sys.stderr)
                    raise

        return solutions

    def _extract_people(self, family_tree: List[Dict]) -> Set[str]:
        """Extract all unique people from the family tree."""
        people = set()
        for relation in family_tree:
            people.add(relation["subject"])
            people.add(relation["object"])
        return people

    def _build_network(self, people: Set[str], family_tree: List[Dict], country: str,
                       test_results: List[Dict]) -> DiscreteBayesianNetwork:
        """Build the Bayesian network for the family."""
        edges = []

        # Create nodes for each person: two alleles + blood type
        for person in people:
            allele1_var = f"{person}_allele1"
            allele2_var = f"{person}_allele2"
            bloodtype_var = f"{person}_bloodtype"

            # Blood type depends on both alleles
            edges.append((allele1_var, bloodtype_var))
            edges.append((allele2_var, bloodtype_var))

        # Build parent-child relationships
        parent_child_map = self._build_parent_child_map(family_tree)

        # Add inheritance edges
        for child, parents in parent_child_map.items():
            if "father" in parents:
                father = parents["father"]
                edges.extend([
                    (f"{father}_allele1", f"{child}_allele1"),
                    (f"{father}_allele2", f"{child}_allele1")
                ])
            if "mother" in parents:
                mother = parents["mother"]
                edges.extend([
                    (f"{mother}_allele1", f"{child}_allele2"),
                    (f"{mother}_allele2", f"{child}_allele2")
                ])

        # Add test result nodes
        test_nodes = self._add_test_nodes(test_results, edges)
        # print(f'{test_nodes=}')
        # Create the Bayesian network
        model = DiscreteBayesianNetwork(edges)

        # Add CPDs
        self._add_cpds(model, people, parent_child_map, country, test_results)

        return model

    def _build_parent_child_map(self, family_tree: List[Dict]) -> Dict[str, Dict[str, str]]:
        """Build a mapping from children to their parents."""
        parent_child_map = {}

        for relation in family_tree:
            relation_type = relation["relation"]
            parent = relation["subject"]
            child = relation["object"]

            if child not in parent_child_map:
                parent_child_map[child] = {}

            if relation_type == "father-of":
                parent_child_map[child]["father"] = parent
            elif relation_type == "mother-of":
                parent_child_map[child]["mother"] = parent

        return parent_child_map

    def _add_test_nodes(self, test_results: List[Dict], edges: List[Tuple[str, str]]) -> List[str]:
        """Add test result nodes to the network."""
        test_nodes = []

        for i, test in enumerate(test_results):
            if test["type"] == "pair-bloodtype-test":
                # Add test result nodes for pair tests
                test_node1 = f"test_{i}_result1"
                test_node2 = f"test_{i}_result2"

                person1 = test["person-1"]
                person2 = test["person-2"]

                # Test results depend on both people's blood types
                edges.extend([
                    (f"{person1}_bloodtype", test_node1),
                    (f"{person2}_bloodtype", test_node1),
                    (f"{person1}_bloodtype", test_node2),
                    (f"{person2}_bloodtype", test_node2)
                ])

                test_nodes.extend([test_node1, test_node2])

            elif test["type"] == "mixed-bloodtype-test":
                # Add test result node for mixed blood test
                test_node = f"test_{i}_mixed_result"

                person1 = test["person-1"]
                person2 = test["person-2"]

                # Mixed test result depends on both people's blood types
                edges.extend([
                    (f"{person1}_bloodtype", test_node),
                    (f"{person2}_bloodtype", test_node)
                ])

                test_nodes.append(test_node)

        return test_nodes

    def _add_cpds(self, model: DiscreteBayesianNetwork, people: Set[str], parent_child_map: Dict, country: str,
                  test_results: List[Dict]):
        """Add conditional probability distributions to the model."""
        allele_dist = self.allele_distributions[country]

        for person in people:
            allele1_var = f"{person}_allele1"
            allele2_var = f"{person}_allele2"
            bloodtype_var = f"{person}_bloodtype"

            # Check if this person is a child with known parents
            parents = parent_child_map.get(person, {})

            if "father" in parents:
                # Child's first allele comes from father
                father = parents["father"]
                cpd_values = []
                for child_allele in self.alleles:
                    row = []
                    for f_a1 in self.alleles:
                        for f_a2 in self.alleles:
                            # 50% chance of each father allele
                            if child_allele == f_a1 and child_allele == f_a2:
                                row.append(1.0)
                            elif child_allele == f_a1 or child_allele == f_a2:
                                row.append(0.5)
                            else:
                                row.append(0.0)
                    cpd_values.append(row)

                cpd_allele1 = TabularCPD(
                    variable=allele1_var,
                    variable_card=3,
                    values=cpd_values,
                    evidence=[f"{father}_allele1", f"{father}_allele2"],
                    evidence_card=[3, 3]
                )
                model.add_cpds(cpd_allele1)
            else:
                # Founder: use population distribution for first allele
                probs = [[allele_dist[allele]] for allele in self.alleles]
                cpd_allele1 = TabularCPD(
                    variable=allele1_var,
                    variable_card=3,
                    values=probs
                )
                model.add_cpds(cpd_allele1)

            if "mother" in parents:
                # Child's second allele comes from mother
                mother = parents["mother"]
                cpd_values = []
                for child_allele in self.alleles:
                    row = []
                    for m_a1 in self.alleles:
                        for m_a2 in self.alleles:
                            # 50% chance of each mother allele
                            if child_allele == m_a1 and child_allele == m_a2:
                                row.append(1.0)
                            elif child_allele == m_a1 or child_allele == m_a2:
                                row.append(0.5)
                            else:
                                row.append(0.0)
                    cpd_values.append(row)

                cpd_allele2 = TabularCPD(
                    variable=allele2_var,
                    variable_card=3,
                    values=cpd_values,
                    evidence=[f"{mother}_allele1", f"{mother}_allele2"],
                    evidence_card=[3, 3]
                )
                model.add_cpds(cpd_allele2)
            else:
                # Founder: use population distribution for second allele
                probs = [[allele_dist[allele]] for allele in self.alleles]
                cpd_allele2 = TabularCPD(
                    variable=allele2_var,
                    variable_card=3,
                    values=probs
                )
                model.add_cpds(cpd_allele2)

            # Blood type CPD (deterministic based on alleles)
            cpd_values = []
            for bt in self.blood_types:
                row = []
                for a1 in self.alleles:
                    for a2 in self.alleles:
                        if self.allele_to_bloodtype.get((a1, a2)) == bt:
                            row.append(1.0)
                        else:
                            row.append(0.0)
                cpd_values.append(row)

            cpd_bloodtype = TabularCPD(
                variable=bloodtype_var,
                variable_card=4,
                values=cpd_values,
                evidence=[allele1_var, allele2_var],
                evidence_card=[3, 3]
            )
            model.add_cpds(cpd_bloodtype)

        # Add CPDs for test results
        self._add_test_cpds(model, test_results)

    def _add_test_cpds(self, model: DiscreteBayesianNetwork, test_results: List[Dict]):
        """Add CPDs for test result nodes."""
        for i, test in enumerate(test_results):
            if test["type"] == "pair-bloodtype-test":
                test_node1 = f"test_{i}_result1"
                test_node2 = f"test_{i}_result2"

                person1 = test["person-1"]
                person2 = test["person-2"]

                # CPD for first test result
                # 80% chance correct, 20% chance swapped
                cpd_values1 = []
                for result_bt in self.blood_types:
                    row = []
                    for p1_bt in self.blood_types:
                        for p2_bt in self.blood_types:
                            if result_bt == p1_bt:
                                if result_bt == p2_bt:
                                    row.append(1)  # Same bloodtype for both
                                else:
                                    row.append(0.8)  # Correct result
                            elif result_bt == p2_bt:
                                row.append(0.2)  # Swapped result
                            else:
                                row.append(0.0)
                    cpd_values1.append(row)

                cpd_test1 = TabularCPD(
                    variable=test_node1,
                    variable_card=4,
                    values=cpd_values1,
                    evidence=[f"{person1}_bloodtype", f"{person2}_bloodtype"],
                    evidence_card=[4, 4]
                )
                model.add_cpds(cpd_test1)

                # CPD for second test result (complementary to first)
                cpd_values2 = []
                for result_bt in self.blood_types:
                    row = []
                    for p1_bt in self.blood_types:
                        for p2_bt in self.blood_types:
                            if result_bt == p2_bt:
                                if result_bt == p1_bt:
                                    row.append(1)  # Same bloodtype for both
                                else:
                                    row.append(0.8)  # Correct result for person 2
                            elif result_bt == p1_bt:
                                row.append(0.2)  # Swapped result
                            else:
                                row.append(0.0)
                    cpd_values2.append(row)

                cpd_test2 = TabularCPD(
                    variable=test_node2,
                    variable_card=4,
                    values=cpd_values2,
                    evidence=[f"{person1}_bloodtype", f"{person2}_bloodtype"],
                    evidence_card=[4, 4]
                )
                model.add_cpds(cpd_test2)

            elif test["type"] == "mixed-bloodtype-test":
                test_node = f"test_{i}_mixed_result"

                person1 = test["person-1"]
                person2 = test["person-2"]

                # CPD for mixed blood test result
                # The result is the combination of antigens from both people
                cpd_values = []
                for result_bt in self.blood_types:
                    row = []
                    for p1_bt in self.blood_types:
                        for p2_bt in self.blood_types:
                            mixed_result = self._mix_blood_types(p1_bt, p2_bt)
                            if result_bt == mixed_result:
                                row.append(1.0)  # Deterministic result
                            else:
                                row.append(0.0)
                    cpd_values.append(row)

                cpd_mixed = TabularCPD(
                    variable=test_node,
                    variable_card=4,
                    values=cpd_values,
                    evidence=[f"{person1}_bloodtype", f"{person2}_bloodtype"],
                    evidence_card=[4, 4]
                )
                model.add_cpds(cpd_mixed)

    def _mix_blood_types(self, blood_type1: str, blood_type2: str) -> str:
        """
        Determine the result of mixing two blood types.
        The mixed result shows all antigens present in either blood sample.
        """
        # Map blood types to their antigens
        antigens = {
            "A": {"A"},
            "B": {"B"},
            "AB": {"A", "B"},
            "O": set()
        }

        # Combine antigens from both blood types
        combined_antigens = antigens[blood_type1] | antigens[blood_type2]
        # print(combined_antigens)
        # Map combined antigens back to blood type
        if combined_antigens == {"A"}:
            return "A"
        elif combined_antigens == {"B"}:
            return "B"
        elif combined_antigens == {"A", "B"}:
            return "AB"
        else:  # empty set
            return "O"

    def _process_test_results(self, test_results: List[Dict], people: Set[str]) -> Dict[str, int]:
        """Process test results and return evidence dictionary."""
        evidence = {}

        for i, test in enumerate(test_results):
            if test["type"] == "bloodtype-test":
                person = test["person"]
                result = test["result"]
                bloodtype_var = f"{person}_bloodtype"
                evidence[bloodtype_var] = self.blood_types.index(result)

            elif test["type"] == "pair-bloodtype-test":
                # Add evidence for test result nodes
                result1 = test["result-1"]
                result2 = test["result-2"]

                test_node1 = f"test_{i}_result1"
                test_node2 = f"test_{i}_result2"

                evidence[test_node1] = self.blood_types.index(result1)
                evidence[test_node2] = self.blood_types.index(result2)

            elif test["type"] == "mixed-bloodtype-test":
                # Add evidence for mixed test result node
                mixed_result = test["result"]
                test_node = f"test_{i}_mixed_result"
                evidence[test_node] = self.blood_types.index(mixed_result)

        return evidence


def main(input_file='example-problems/problem-a-00.json', result_folder='example-results'):
    try:
        with open(input_file, 'r') as f:
            problem_data = json.load(f)

        solver = BloodTypeSolver()
        solution = solver.solve_problem(problem_data)

        # print(f'{os.path.split(input_file)[-1]}\n{json.dumps(solution, indent=2)}\n{json.dumps(problem_data, indent=2)}')
        with open(os.path.join(result_folder, f'solution{os.path.split(input_file)[-1][7:]}'), 'w') as f:
            f.write(json.dumps(solution, indent=2))

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        # sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}", file=sys.stderr)
        # sys.exit(1)
    except Exception as e:
        print(f"{os.path.split(input_file)[-1]}\n{json.dumps(problem_data, indent=2)}\n\nError: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        # sys.exit(1)


if __name__ == "__main__":
    # main('example-problems/problem-d-00.json', 'example-results')
    # examples
    # for f in os.listdir('example-problems'):
    #     main(f'{os.path.join("example-problems", f)}', 'example-results')
    # task
    for f in os.listdir('problems'):
        main(f'{os.path.join("problems", f)}', 'results')
