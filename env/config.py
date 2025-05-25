from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Class for initializing obstacles and goals"""

    obstacles = [
        {"center": [0.8, -3.0], "radius": 0.1},
        {"center": [-1.8, 3.0], "radius": 0.1},
    ]

    goals = [
                {"center": [0.0, 0.0], "radius": 0.1}
            ]
    
    goal_strength = 123
    obstacle_strength = -100

