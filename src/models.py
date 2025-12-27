"""
Models

This module contains all data models, validation schemas, and workout patterns
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional, Sequence, TypeAlias, Union

from pydantic import BaseModel, Field

from src.utils import round_weight

# =============================================================================
# Type Aliases
# =============================================================================

LoadCoefficient: TypeAlias = Union[float, str, None]
SchemeValue: TypeAlias = Union[int, str]
LoadPattern: TypeAlias = list[tuple[LoadCoefficient, ...]]
SchemePattern: TypeAlias = list[tuple[SchemeValue, ...]]
ModulationPattern: TypeAlias = list[tuple[int, Optional[int]]]


# =============================================================================
# Enums
# =============================================================================
class LiftType(str, Enum):
    """Standard lift types."""

    SQUAT = "Squat"
    PRESS = "Press"
    BENCH_PRESS = "Bench Press"
    DEADLIFT = "Deadlift"
    PULL_UP = "Pull-ups"
    DB_ROW = "Dumbbell Rows"
    BARBELL_CURL = "Barbell Curls"
    TRICEP_EXT = "Tricep Extensions"
    CORE = "Core Work"

    def __str__(self) -> str:
        """Return the human-readable name."""
        return self.value


class Units(str, Enum):
    """Weight units."""

    POUNDS = "pounds"
    KILOGRAMS = "kilograms"


class BarType(str, Enum):
    """Barbell types."""

    STANDARD = "standard"
    WOMENS = "womens"


# Workout data structure types (defined after LiftType)
@dataclass
class SetData:
    """A single set with weight and reps."""

    weight: Union[float, str, None]
    reps: Union[int, str]


@dataclass
class ExerciseData:
    """An exercise with its lift type and sets."""

    lift_type: LiftType
    sets: list[SetData]


@dataclass
class SessionData:
    """A training session with its name and exercises."""

    name: str
    exercises: list[ExerciseData]


@dataclass
class ProgramCycle:
    """Return type for Microcycle.generate_cycle()."""

    name: str
    cycle: list[list[SessionData]]
    training_maxes: dict[str, float]
    notes: str


# =============================================================================
# Input Validation Schemas (Pydantic)
# =============================================================================
class LiftInput(BaseModel):
    """Input schema for lift data."""

    name: str = Field(..., min_length=1, max_length=100)
    squat: int = Field(..., gt=0)
    press: int = Field(..., gt=0)
    bench_press: int = Field(..., gt=0)
    deadlift: int = Field(..., gt=0)
    units: Units = Units.POUNDS
    bar_type: BarType = BarType.STANDARD
    calculate_tms: str = Field(default="maxes", pattern="^(maxes|tmaxes)$")
    light: bool = False

    class Config:
        use_enum_values = True


# =============================================================================
# Lift Models
# =============================================================================
@dataclass
class Lift:
    """Represents a lift with its training parameters."""

    lift_type: LiftType
    personal_record: int
    training_max: Optional[Union[float, int]] = None
    increment: float = 10.0
    barbell_weight: float = 45.0

    def __post_init__(self) -> None:
        """Calculate training max if not provided."""
        if self.training_max is None:
            # Default: use 90% of 1RM as training max (standard 5/3/1 methodology)
            self.training_max = self.personal_record * 0.9
        elif isinstance(self.training_max, (int, float)) and 0 <= self.training_max <= 1:
            # If a percentage is provided, multiply by 1RM
            self.training_max = float(self.training_max * self.personal_record)

        self.training_max = float(self.training_max)

    def increase_training_max(self) -> None:
        """Increment the training max by the specified amount."""
        if self.training_max is not None and self.increment:
            self.training_max += self.increment


@dataclass
class Element:
    """A single element of a workout with logic for modulation."""

    lift: Lift
    load_coefficients: LoadPattern = field(default_factory=list)
    scheme: SchemePattern = field(default_factory=list)
    training_max_modulation: list[tuple[int, Optional[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize element attributes."""
        if not self.load_coefficients:
            raise ValueError("load_coefficients cannot be empty")

        self.session_index = 0
        self.load_coefficient_index = 0
        self.scheme_index = 0

    def calculate_loads(
        self, coefficients: Sequence[LoadCoefficient]
    ) -> list[Union[float, str, None]]:
        """Calculate actual weights from percentage coefficients."""
        result: list[Union[float, str, None]] = []
        for coeff in coefficients:
            if isinstance(coeff, float) and self.lift.training_max is not None:
                result.append(
                    round_weight(coeff * self.lift.training_max, self.lift.barbell_weight)
                )
            else:
                result.append(coeff)
        return result

    def __iter__(self) -> "Element":
        return self

    def __next__(self) -> ExerciseData:
        """Generate the next set of loads and schemes for this element."""
        if self.load_coefficient_index == len(self.load_coefficients):
            self.load_coefficient_index = 0

        loads = self.load_coefficients[self.load_coefficient_index]
        load_queue = self.calculate_loads(loads)
        self.load_coefficient_index += 1

        if self.scheme_index == len(self.scheme):
            self.scheme_index = 0
        scheme = self.scheme[self.scheme_index]
        self.scheme_index += 1

        scheme_scale = (len(load_queue) + len(scheme) - 1) // len(scheme)
        scheme_queue = scheme * scheme_scale

        # Pair each weight with its rep scheme to create sets
        sets = []
        for weight, reps in zip(load_queue, scheme_queue, strict=False):
            sets.append(SetData(weight=weight, reps=reps))

        return ExerciseData(lift_type=self.lift.lift_type, sets=sets)


# =============================================================================
# Training Patterns
# =============================================================================
WENDLER_LOADS: LoadPattern = [
    (0.65, 0.75, 0.85),  # Week 1: 5s
    (0.70, 0.80, 0.90),  # Week 2: 3s
    (0.75, 0.85, 0.95),  # Week 3: 5/3/1
]

WENDLER_SCHEME: SchemePattern = [
    (5, 5, "5+"),  # Week 1
    (3, 3, "3+"),  # Week 2
    (5, 3, "1+"),  # Week 3
]

WENDLER_TM_MOD: ModulationPattern = [(3, 10), (3, 10), (1, 0)]

# First Set Last (FSL) Pattern
FSL_LOADS: LoadPattern = [(0.65,), (0.70,), (0.75,)]
FSL_SCHEME: SchemePattern = [("3-5 sets of 5-8 reps",)]

# Accessory Pattern
ACCESSORY_LOADS: LoadPattern = [("bodyweight",)]
ACCESSORY_SCHEME: SchemePattern = [("5 sets of 10",)]

# Deload Pattern (for future use)
DELOAD_LOADS: LoadPattern = [(0.40, 0.50, 0.60)]
DELOAD_SCHEME: SchemePattern = [(5, 5, 5)]


class WendlerPattern(Element):
    """Base 5/3/1 pattern."""

    def __init__(self, lift: Lift) -> None:
        super().__init__(
            lift=lift,
            load_coefficients=WENDLER_LOADS,
            scheme=WENDLER_SCHEME,
            training_max_modulation=WENDLER_TM_MOD,
        )


class FirstSetLastPattern(Element):
    """Back-off sets using first set percentage."""

    def __init__(self, lift: Lift) -> None:
        super().__init__(lift=lift, load_coefficients=FSL_LOADS, scheme=FSL_SCHEME)


class AccessoryPattern(Element):
    """Accessory lift pattern."""

    def __init__(self, lift: Lift) -> None:
        super().__init__(lift=lift, load_coefficients=ACCESSORY_LOADS, scheme=ACCESSORY_SCHEME)


class DeloadPattern(Element):
    """Deload week pattern (for future use)."""

    def __init__(self, lift: Lift) -> None:
        super().__init__(lift=lift, load_coefficients=DELOAD_LOADS, scheme=DELOAD_SCHEME)


# =============================================================================
# Workout Session Models
# =============================================================================
@dataclass
class Session:
    """
    Represents a single training session with multiple elements.

    A session typically consists of a main lift (e.g., Squat)
    followed by supplementary work (e.g., accessories).
    """

    name: str
    elements: list[Element]

    def __post_init__(self) -> None:
        """Initialize generators for elements."""
        self._validate_elements()
        self.element_generators = list(self.elements)

    def _validate_elements(self) -> None:
        """Validate that elements are properly configured."""
        if not self.elements:
            raise ValueError(f"Session '{self.name}' must have at least one element")

        # Check for valid load coefficients in elements
        for element in self.elements:
            if not element.load_coefficients:
                raise ValueError(
                    f"Element for {element.lift.lift_type} must have load coefficients"
                )

    def __iter__(self) -> Iterator[SessionData]:
        return self

    def __next__(self) -> SessionData:
        """Generate the next set of exercises for this session."""
        exercises = []
        for element in self.element_generators:
            exercises.append(next(element))
        return SessionData(name=self.name, exercises=exercises)


@dataclass
class Microcycle:
    """
    Represents a series of training sessions as a microcycle.

    A microcycle typically spans 1-4 weeks and includes multiple
    sessions targeting different lifts.
    """

    name: str
    sessions: list[Session]
    length: int
    training_max_modulation: Optional[list[tuple[int, int]]] = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Initialize cycle state."""
        self.mod_index = 0
        self.session_counter = 0

    def generate_cycle(self) -> ProgramCycle:
        """Generate a complete training cycle."""
        cycle = []
        for _ in range(self.length):
            sessions = []
            for session in self.sessions:
                sessions.append(next(session))
            cycle.append(sessions)

        return ProgramCycle(
            name=self.name,
            cycle=cycle,
            training_maxes=self.training_maxes,
            notes=self.notes,
        )

    @property
    def training_maxes(self) -> dict[str, float]:
        """Get current training maxes for all lifts."""
        maxes: dict[str, float] = {}
        for session in self.sessions:
            for element in session.elements:
                if element.lift.training_max is not None:
                    maxes[str(element.lift.lift_type)] = element.lift.training_max
        return maxes

    def increase_training_maxes(self) -> None:
        """Increase training maxes for all lifts in the cycle."""
        for session in self.sessions:
            for element in session.elements:
                element.lift.increase_training_max()


# =============================================================================
# Specific Session Types
# =============================================================================
@dataclass
class Squat531Session(Session):
    """Squat day with 5/3/1 progression."""

    name: str = "Squat"
    elements: list[Element] = field(default_factory=list)

    def __init__(self, squat_lift: Lift, core_lift: Optional[Lift] = None) -> None:
        """Initialize with configured elements."""
        elements = [WendlerPattern(squat_lift), FirstSetLastPattern(squat_lift)]
        if core_lift:
            elements.append(AccessoryPattern(core_lift))
        super().__init__(self.name, elements)


@dataclass
class Press531Session(Session):
    """Press day with 5/3/1 progression."""

    name: str = "Press"
    elements: list[Element] = field(default_factory=list)

    def __init__(
        self, press_lift: Lift, pullup_lift: Optional[Lift] = None, curl_lift: Optional[Lift] = None
    ) -> None:
        """Initialize with configured elements."""
        elements = [WendlerPattern(press_lift), FirstSetLastPattern(press_lift)]
        if pullup_lift:
            elements.append(AccessoryPattern(pullup_lift))
        if curl_lift:
            elements.append(AccessoryPattern(curl_lift))
        super().__init__(self.name, elements)


@dataclass
class Deadlift531Session(Session):
    """Deadlift day with 5/3/1 progression."""

    name: str = "Deadlift"
    elements: list[Element] = field(default_factory=list)

    def __init__(self, deadlift_lift: Lift, core_lift: Optional[Lift] = None) -> None:
        """Initialize with configured elements."""
        elements = [WendlerPattern(deadlift_lift), FirstSetLastPattern(deadlift_lift)]
        if core_lift:
            elements.append(AccessoryPattern(core_lift))
        super().__init__(self.name, elements)


@dataclass
class BenchPress531Session(Session):
    """Bench Press day with 5/3/1 progression."""

    name: str = "Bench Press"
    elements: list[Element] = field(default_factory=list)

    def __init__(
        self, bench_lift: Lift, row_lift: Optional[Lift] = None, tricep_lift: Optional[Lift] = None
    ) -> None:
        """Initialize with configured elements."""
        elements = [WendlerPattern(bench_lift), FirstSetLastPattern(bench_lift)]
        if row_lift:
            elements.append(AccessoryPattern(row_lift))
        if tricep_lift:
            elements.append(AccessoryPattern(tricep_lift))
        super().__init__(self.name, elements)


@dataclass
class Wendler531Cycle(Microcycle):
    """Standard 5/3/1 training cycle."""

    def __init__(
        self,
        main_lifts: Sequence[Lift],
        accessory_lifts: Optional[Sequence[Lift]] = None,
        name: str = "Wendler 531 Cycle",
        length: int = 3,
        notes: str = "Three week Wendler microcycle with FSL",
    ) -> None:
        """Initialize with configured sessions."""
        # Get lifts by type
        lifts = {lift.lift_type: lift for lift in main_lifts}
        acc_lifts = {lift.lift_type: lift for lift in (accessory_lifts or [])}

        # Create sessions
        sessions = [
            Squat531Session(lifts[LiftType.SQUAT], acc_lifts.get(LiftType.CORE)),
            Press531Session(
                lifts[LiftType.PRESS],
                acc_lifts.get(LiftType.PULL_UP),
                acc_lifts.get(LiftType.BARBELL_CURL),
            ),
            Deadlift531Session(lifts[LiftType.DEADLIFT], acc_lifts.get(LiftType.CORE)),
            BenchPress531Session(
                lifts[LiftType.BENCH_PRESS],
                acc_lifts.get(LiftType.DB_ROW),
                acc_lifts.get(LiftType.TRICEP_EXT),
            ),
        ]

        super().__init__(sessions=sessions, length=length, name=name, notes=notes)
