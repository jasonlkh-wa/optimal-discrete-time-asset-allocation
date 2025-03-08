import numpy as np
from typing import Tuple

ActionValueDict = dict[float, float]
Allocation = float
TurnNumber = int
Wealth = float
# CR kleung: change the state to Tuple[TurnNumber, Wealth]
StateActionValueDict = dict[TurnNumber, ActionValueDict]
OneDimensionArray = np.ndarray[tuple[int], np.dtype[np.float64]]
