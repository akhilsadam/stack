"""
Random RPN generator for training data.

Generates valid RPN expressions using the available operators and variables
from the vocabulary, with configurable complexity and distribution.
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum


class NodeType(Enum):
    """Types of nodes in expression generation."""
    VARIABLE = "variable"
    CONSTANT = "constant"
    OPERATOR = "operator"


class Arity(Enum):
    """Operator arity."""
    UNARY = 1
    BINARY = 2
    SPECIAL = 3  # For operators like jacobian


class RPNGenerator:
    """
    Generates random RPN expressions with configurable complexity.

    The generator constructs expressions as trees and converts them to
    Reverse Polish Notation (RPN) strings.
    """

    def __init__(
        self,
        vocab: Dict[str, List[str]],
        max_depth: int = 4,
        max_nodes: int = 20,
        constant_prob: float = 0.2,
        operator_distribution: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the RPN generator.

        Parameters
        ----------
        vocab : Dict[str, List[str]]
            Vocabulary with categories:
            - "variables": list of variable tokens (q, psi, u, v, x, y)
            - "unary_ops": list of unary operators (sqrt, cos, sin, exp, neg, dx, dy, lap)
            - "binary_ops": list of binary operators (+, -, *, dot)
            - "special_ops": list of special operators (jacobian, grad, div, curl)
            - "constants": list for constant placeholders (empty for now)
        max_depth : int
            Maximum tree depth for generated expressions
        max_nodes : int
            Maximum number of nodes in generated expression
        constant_prob : float
            Probability of generating a constant instead of variable
        operator_distribution : Optional[Dict[str, float]]
            Custom distribution for operator selection (normalized automatically)
        """
        self.vocab = vocab
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.constant_prob = constant_prob

        # Default operator distribution if not provided
        if operator_distribution is None:
            operator_distribution = {
                "+": 0.3, "-": 0.2, "*": 0.3, "neg": 0.02,
                "cos": 0.08, "sin": 0.08,
                "dx": 0.03, "dy": 0.03, "lap": 0.02,
                "jacobian": 0.08
            }

        # Normalize distribution
        total = sum(operator_distribution.values())
        self.operator_distribution = {k: v/total for k, v in operator_distribution.items()}

        # Build operator info mapping
        self._build_operator_info()

    def _build_operator_info(self):
        """Build internal operator information from vocab."""
        self.operator_arity = {}
        self.operator_vocab = {}

        # Unary operators
        for op in self.vocab.get("unary_ops", []):
            self.operator_arity[op] = Arity.UNARY
            self.operator_vocab[op] = "unary_ops"

        # Binary operators
        for op in self.vocab.get("binary_ops", []):
            self.operator_arity[op] = Arity.BINARY
            self.operator_vocab[op] = "binary_ops"

        # Special operators (e.g., jacobian is binary in RPN: a b jacobian)
        for op in self.vocab.get("special_ops", []):
            if op == "jacobian":
                self.operator_arity[op] = Arity.BINARY  # Takes two arguments
            elif op in {"grad", "div", "curl"}:
                self.operator_arity[op] = Arity.UNARY
            else:
                self.operator_arity[op] = Arity.SPECIAL
            self.operator_vocab[op] = "special_ops"

    def _random_operator(self) -> str:
        """Select a random operator according to distribution."""
        ops = list(self.operator_distribution.keys())
        probs = [self.operator_distribution[op] for op in ops]
        return random.choices(ops, weights=probs, k=1)[0]

    def _random_variable(self) -> str:
        """Select a random variable."""
        return random.choice(self.vocab["variables"])

    def _random_constant(self) -> str:
        """Generate a random constant value."""
        # For now, generate small integer or simple float
        if random.random() < 0.7:
            return str(random.randint(0, 10))
        else:
            return f"{random.uniform(0.1, 5.0):.2f}"

    def generate_expression_tree(self, depth: int = 0, node_count: int = 0) -> Tuple[Dict, int]:
        """
        Recursively generate an expression tree.

        Returns
        -------
        Tuple[Dict, int]
            Expression tree as nested dict and new node count
        """
        if depth >= self.max_depth or node_count >= self.max_nodes:
            # Return a leaf (variable or constant)
            if random.random() < self.constant_prob and self.vocab.get("constants"):
                return {"type": NodeType.CONSTANT, "value": self._random_constant()}, node_count + 1
            else:
                return {"type": NodeType.VARIABLE, "value": self._random_variable()}, node_count + 1

        # Decide whether to generate operator or leaf
        # Lower probability for operators near depth limit
        operator_prob = 0.7 * (1 - depth / self.max_depth)

        if random.random() < operator_prob:
            op = self._random_operator()
            arity = self.operator_arity.get(op, Arity.BINARY)

            if arity == Arity.UNARY:
                # For unary operators like dx, dy, sqrt, cos, sin, etc.
                child, new_count = self.generate_expression_tree(depth + 1, node_count + 1)
                return {
                    "type": NodeType.OPERATOR,
                    "value": op,
                    "children": [child]
                }, new_count

            elif arity == Arity.BINARY:
                # For binary operators like +, -, *, dot
                left, count1 = self.generate_expression_tree(depth + 1, node_count + 1)
                right, count2 = self.generate_expression_tree(depth + 1, count1)
                return {
                    "type": NodeType.OPERATOR,
                    "value": op,
                    "children": [left, right]
                }, count2

            else:  # SPECIAL arity
                # For jacobian: binary but both args should be variables
                if op == "jacobian":
                    left = {"type": NodeType.VARIABLE, "value": self._random_variable()}
                    right = {"type": NodeType.VARIABLE, "value": self._random_variable()}
                    return {
                        "type": NodeType.OPERATOR,
                        "value": op,
                        "children": [left, right]
                    }, node_count + 3
                else:
                    # Default to binary
                    left, count1 = self.generate_expression_tree(depth + 1, node_count + 1)
                    right, count2 = self.generate_expression_tree(depth + 1, count1)
                    return {
                        "type": NodeType.OPERATOR,
                        "value": op,
                        "children": [left, right]
                    }, count2
        else:
            # Generate leaf
            if random.random() < self.constant_prob and self.vocab.get("constants"):
                return {"type": NodeType.CONSTANT, "value": self._random_constant()}, node_count + 1
            else:
                return {"type": NodeType.VARIABLE, "value": self._random_variable()}, node_count + 1

    def tree_to_rpn(self, tree: Dict) -> List[str]:
        """
        Convert expression tree to RPN token list.

        Parameters
        ----------
        tree : Dict
            Expression tree

        Returns
        -------
        List[str]
            RPN tokens in order
        """
        if tree["type"] == NodeType.VARIABLE or tree["type"] == NodeType.CONSTANT:
            return [tree["value"]]

        # Operator node
        op = tree["value"]
        children = tree.get("children", [])

        if len(children) == 1:
            # Unary operator
            child_rpn = self.tree_to_rpn(children[0])
            return child_rpn + [op]

        elif len(children) == 2:
            # Binary operator
            left_rpn = self.tree_to_rpn(children[0])
            right_rpn = self.tree_to_rpn(children[1])
            return left_rpn + right_rpn + [op]

        else:
            # Shouldn't happen
            return []

    def generate_rpn(self, n: int = 1) -> Union[str, List[str]]:
        """
        Generate one or more RPN expressions.

        Parameters
        ----------
        n : int
            Number of expressions to generate

        Returns
        -------
        Union[str, List[str]]
            Single RPN string if n=1, list of strings otherwise
        """
        if n == 1:
            tree, _ = self.generate_expression_tree()
            tokens = self.tree_to_rpn(tree)
            return " ".join(tokens)
        else:
            return [self.generate_rpn(1) for _ in range(n)]

    def generate_batch(
        self,
        batch_size: int,
    ) -> List[str]:
        """
        Generate a batch of RPN expressions.

        Parameters
        ----------
        batch_size : int
            Number of expressions to generate
            
        Returns
        -------
            List of RPN strings
        """
        rpns = []

        for i in range(batch_size):
            rpn = self.generate_rpn()
            rpns.append(rpn)

        return rpns


def create_vocab_from_embeddings() -> Dict[str, List[str]]:
    """
    Create generator vocabulary from the embeddings vocabulary.

    Returns
    -------
    Dict[str, List[str]]
        Categorized vocabulary for the generator
    """
    from qg.solver.opt.operator.rpn.embeddings import TOKEN_TO_ID, TOKEN_TO_CAT

    vocab = {
        "variables": [],
        "unary_ops": [],
        "binary_ops": [],
        "special_ops": [],
        "constants": ["__scalar__"]  # Placeholder for scalar constants
    }

    # Categorize tokens based on their category
    from qg.solver.opt.operator.rpn.embeddings import TokenCategory

    for token, token_id in TOKEN_TO_ID.items():
        if token == "__scalar__":
            continue  # Already added to constants

        category = TOKEN_TO_CAT.get(token)

        if category == TokenCategory.VARIABLE:
            vocab["variables"].append(token)
        elif category == TokenCategory.NONLINEAR_UNARY:
            vocab["unary_ops"].append(token)
        elif category == TokenCategory.BINARY_OP:
            vocab["binary_ops"].append(token)
        elif category == TokenCategory.JACOBIAN:
            vocab["special_ops"].append(token)
        elif category == TokenCategory.VECTOR_OP:
            if token in {"grad", "div", "curl"}:
                vocab["unary_ops"].append(token)
            elif token == "dot":
                vocab["binary_ops"].append(token)
        elif category == TokenCategory.LINEAR_DIFF:
            vocab["unary_ops"].append(token)
        elif category == TokenCategory.MISC_OP:
            if token == "neg":
                vocab["unary_ops"].append(token)
            # dealias is not really an operator for generation

    return vocab


if __name__ == "__main__":
    # Test the generator
    vocab = create_vocab_from_embeddings()
    print("Vocabulary categories:")
    for key, values in vocab.items():
        print(f"  {key}: {values[:5]}{'...' if len(values) > 5 else ''}")

    generator = RPNGenerator(vocab, max_depth=3, max_nodes=10)

    print("\nGenerated RPN expressions:")
    for i in range(5):
        rpn = generator.generate_rpn()
        print(f"  {i+1}. {rpn}")

    print("\nGenerated batch (no params needed):")
    batch, _ = generator.generate_batch(3, include_scalars=False)
    for i, rpn in enumerate(batch):
        print(f"  {i+1}. {rpn}")