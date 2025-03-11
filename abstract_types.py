import numpy as np
from typing import Tuple, Literal

Mode = Literal["Prod", "Test"]
ActionValueDict = dict[float, float]
Allocation = float
TurnNumber = int
Wealth = float
StateActionValueDict = dict[Tuple[TurnNumber, Wealth], ActionValueDict]
OneDimensionArray = np.ndarray[tuple[int], np.dtype[np.float64]]
