import random
import collections
from typing import Any, Dict, List, Callable, Tuple, Optional

class GrammarBreak(Exception):
    """Exception raised to break out of grammar rules up to the nearest loop."""
    pass

BREAK = "!!BREAK!!"

class GeneratorVM:
    """
    Virtual Machine for generating Question/Trace pairs.
    Internal state is a single unified dictionary (ctx).
    """
    def __init__(self):
        self.ctx: Dict[str, Any] = collections.defaultdict(lambda: None)

    def get_ctx(self, key: str) -> Any:
        return self.ctx[key]

    def set_ctx(self, key: str, value: Any):
        self.ctx[key] = value

    def result(self) -> Dict[str, Any]:
        return dict(self.ctx)


class Node:
    """Abstract Base Class for Grammar Nodes."""
    def expand(self, vm: GeneratorVM) -> None:
        raise NotImplementedError

class AND(Node):
    def __init__(self, steps: List[Node]):
        self.steps = steps
    def expand(self, vm: GeneratorVM) -> None:
        for step in self.steps:
            step.expand(vm)

class OR(Node):
    def __init__(self, options: List[Tuple[Node, float]]):
        self.options = options
    def expand(self, vm: GeneratorVM) -> None:
        choices, weights = zip(*self.options)
        selected_node = random.choices(choices, weights=weights, k=1)[0]
        selected_node.expand(vm)

class MANY(Node):
    def __init__(self, node: Node, probability: float = 0.5):
        self.node = node
        self.probability = probability
    def expand(self, vm: GeneratorVM) -> None:
        try:
            self.node.expand(vm)
            while True:
                if random.random() > self.probability:
                    break
                self.node.expand(vm)
        except GrammarBreak:
            pass

class APPLY(Node):
    """
    Unifying primitive for all context manipulation and tool logic.
    Accepts, triggers, and syncs requested context keys.
    """
    def __init__(self, func: Callable, access: Any = None):
        self.func = func
        self.access_keys = [access] if isinstance(access, str) else list(access) if access else []

    def expand(self, vm: GeneratorVM) -> None:
        if not self.access_keys:
            result = self.func()
        else:
            data = {k: vm.get_ctx(k) for k in self.access_keys}
            result = self.func(data)
            for k, v in data.items():
                vm.set_ctx(k, v)

        if result is BREAK:
            raise GrammarBreak()

# --- Ergonomic Helpers (Constructing optimized APPLY nodes) ---

def Rule(*steps: Node) -> Node:
    return AND(list(steps))

def Choice(*options: Tuple[Node, float]) -> Node:
    return OR(list(options))

def Push(target: str, read: str) -> Node:
    """Helper: appends/adds context[read] into context[target]."""
    def _push_logic(data: dict):
        val = data[read]
        # In-place initialization and mutation
        container = data.get(target)
        if container is None:
            data[target] = [val]
        elif isinstance(container, list):
            container.append(val)
        elif isinstance(container, set):
            container.add(val)
    return APPLY(_push_logic, access=[target, read])

def Pop(source: str, write: str) -> Node:
    """Helper: pops from context[source] into context[write]."""
    def _pop_logic(data: dict):
        container = data.get(source)
        data[write] = container.pop() if container else None
    return APPLY(_pop_logic, access=[source, write])

def Literal(text: str, target: str = "nl") -> Node:
    """Helper: injects a literal string into a list key."""
    def _lit_logic(data: dict):
        if data[target] is None: data[target] = []
        data[target].append(text)
    return APPLY(_lit_logic, access=target)
