import random
import collections
import copy
from typing import Any, Dict, List, Callable, Tuple, Optional

class RetrySignal(Exception):
    """Exception raised to trigger a retry or break out of loops."""
    def __init__(self, reason: Optional[str] = None):
        self.reason = reason
        super().__init__(reason)

class GeneratorVM:
    """
    Virtual Machine for generating Question/Trace pairs.
    Internal state is a single unified dictionary (ctx).
    """
    def __init__(self, initial_ctx: Optional[Dict[str, Any]] = None):
        self.initial_ctx = initial_ctx or {}
        self.ctx: Dict[str, Any] = collections.defaultdict(lambda: None)
        self.reset()

    def reset(self):
        """Resets the VM state to a fresh copy of the initial context."""
        self.ctx.clear()
        self.ctx.update(copy.deepcopy(self.initial_ctx))

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
    def __init__(self, node: Node, probability: Optional[float] = None, min_count: Optional[int] = None, max_count: Optional[int] = None):
        self.node = node
        self.probability = probability
        self.min_count = min_count
        self.max_count = max_count
        
        # Guard: At least one repetition strategy must be defined
        if self.probability is None and (self.min_count is None or self.max_count is None):
            raise Exception("MANY node requires either an explicit 'probability' OR both 'min_count' and 'max_count'")

    def expand(self, vm: GeneratorVM) -> None:
        if self.min_count is not None and self.max_count is not None:
             # Count-based mode
             num = random.randint(self.min_count, self.max_count)
             for _ in range(num):
                  self.node.expand(vm)
             return

        if self.probability is not None:
            # Probabilistic mode (always runs at least once)
            try:
                self.node.expand(vm)
                while True:
                    if random.random() > self.probability:
                        break
                    self.node.expand(vm)
            except RetrySignal:
                pass

class RETRY(Node):
    """Retries a node N times, catching RetrySignal or Exception, with state rollback."""
    def __init__(self, node: Node, n: int = 3):
        self.node = node
        self.n = n

    def expand(self, vm: GeneratorVM) -> None:
        last_err = None
        for i in range(self.n):
            # Snapshot state for rollback
            snapshot = copy.deepcopy(vm.ctx)
            try:
                self.node.expand(vm)
                return
            except (RetrySignal, Exception) as e:
                last_err = e
                # Rollback state
                vm.ctx = snapshot
                
                reason = getattr(e, 'reason', str(e))
                print(f"  [RETRY {i+1}/{self.n}] Backtracking due to: {reason}")
                continue
        
        # If we exhausted retries, bubble up the last error
        raise last_err or Exception(f"RETRY exhausted after {self.n} attempts")

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
            self.func()
        else:
            data = {k: vm.get_ctx(k) for k in self.access_keys}
            self.func(data)
            for k, v in data.items():
                vm.set_ctx(k, v)

# --- Ergonomic Helpers (Constructing optimized APPLY nodes) ---

def Rule(*steps: Node) -> Node:
    return AND(list(steps))

def Choice(*options: Tuple[Node, float]) -> Node:
    return OR(list(options))

def Retry(node: Node, n: int = 5) -> Node:
    return RETRY(node, n=n)

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