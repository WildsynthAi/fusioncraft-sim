from abc import ABC, abstractmethod
import numpy as np

class Module(ABC):
    """
    Base class for physics modules.
    Each module exposes:
      - state vector (numpy)
      - derivative(t, state, external_inputs) -> ndarray
      - state_labels list for ordering
    """
    def __init__(self):
        self.state = np.zeros(0)
        self.state_labels = []

    @abstractmethod
    def derivative(self, t: float, state: np.ndarray, inputs: dict) -> np.ndarray:
        pass

    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def set_state(self, vec: np.ndarray):
        self.state = vec.copy()
