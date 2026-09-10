"""Complete identities for upstream fully-connected implementations, in graph order."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitPlan:
    implementations: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.implementations, tuple) or not self.implementations:
            raise ValueError(
                "A plan requires a nonempty immutable implementation tuple"
            )
        if any(type(i) is not int or i not in range(6) for i in self.implementations):
            raise ValueError(
                "Each fully-connected implementation must be an integer in 0..5"
            )

    @classmethod
    def from_model(cls, model):
        return cls(
            tuple(
                layer["implementation"]
                for layer in model["layers"]
                if layer["layer_type"] == "FullyConnected"
            )
        )

    @property
    def key(self) -> str:
        """Versioned, lossless path component; never identifies only the first layer."""
        return "fc-v1-" + "-".join(map(str, self.implementations))

    def assign(self, model) -> None:
        """Assign to a caller-owned model only after validating the full layer count."""
        layers = [
            layer
            for layer in model["layers"]
            if layer["layer_type"] == "FullyConnected"
        ]
        if len(layers) != len(self.implementations):
            raise ValueError(
                "Implementation tuple does not match the fully-connected layer count"
            )
        for layer, implementation in zip(layers, self.implementations):
            layer["implementation"] = implementation
