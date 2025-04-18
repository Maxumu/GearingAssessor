from dataclasses import dataclass

@dataclass
class GearConfig:
    max_rear: int = 6
    max_front: int = 2
    smallest_rear: int = 11
    largest_rear: int = 20
    smallest_front: int = 34
    largest_front: int = 50
    chainring_teeth: list = (50, 34)
    use_real: bool = False
    use_generated: bool = True

@dataclass
class VarStore:
    fit_A: float = None
    fit_B: float = None
    fit_C: float = None
    peak_cadence: float = None