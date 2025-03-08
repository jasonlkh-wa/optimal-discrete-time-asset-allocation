import numpy as np

ActionValueDict = dict[float, float]
Allocation = float
TurnNumber = int
StateActionValueDict = dict[TurnNumber, ActionValueDict]
OneDimensionArray = np.ndarray[tuple[int], np.dtype[np.float64]]
