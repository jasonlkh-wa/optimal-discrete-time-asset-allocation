"""
Abstract types used in the project.

This file contains the type hints for the abstract types used in
the asset allocation environment.
"""

import numpy as np
from typing import Tuple, Literal

Mode = Literal["Prod", "Test"]
ActionValueDict = dict[float, float]
Allocation = float
TurnNumber = int
ExpectedReturn = float
Wealth = float
StateActionValueDict = dict[Tuple[TurnNumber, Wealth], ActionValueDict]
OneDimensionArray = np.ndarray[tuple[int], np.dtype[np.float64]]
