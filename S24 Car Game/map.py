# Self Driving Car

# Importing the libraries
import numpy as np
from random import random
import matplotlib.pyplot as plt
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple
import os
from datetime import datetime

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.graphics import Rectangle

# Importing the Dqn object from our AI in ai.py
from ai import Dqn


# Configure logging
def setup_logging():
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create a custom formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a filter to prevent duplicate messages
    class DuplicateFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.last_log = None

        def filter(self, record):
            current_log = (record.levelno, record.getMessage())
            if current_log == self.last_log:
                return False
            self.last_log = current_log
            return True

    # Create a game-specific logger
    logger = logging.getLogger("car_game")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger

    # File handler
    log_file = f"logs/car_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(DuplicateFilter())

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(DuplicateFilter())

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create game logger
logger = setup_logging()

# Constants
WINDOW_WIDTH = 1429
WINDOW_HEIGHT = 660
CAR_SPEED_SAND = 0.5
CAR_SPEED_NORMAL = 2.0
SENSOR_DISTANCE = 30
SENSOR_RANGE = 10
GOAL_DISTANCE_THRESHOLD = 25
BOUNDARY_PADDING = 5
SPAWN_PADDING = 50

# Hardcoded goal positions
GOAL_POSITIONS = [
    (100, 100),  # Top-left
    (1200, 100),  # Top-right
    (1200, 500),  # Bottom-right
    (100, 500),  # Bottom-left
    (650, 300),  # Center
]

# Reward constants
SAND_REWARD = -0.4
BOUNDARY_REWARD = -1
ROAD_REWARD = -0.1
CLOSER_TO_GOAL_REWARD = 0.1
GOAL_REWARD = 10.0

# Kivy Configuration
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", str(WINDOW_WIDTH))
Config.set("graphics", "height", str(WINDOW_HEIGHT))


@dataclass
class GameState:
    """Class to hold the game state"""

    sand: np.ndarray = None
    goals: List[Tuple[int, int]] = None
    current_goal_index: int = 0
    last_distance: float = 0
    last_reward: float = 0
    scores: List[float] = None
    first_update: bool = True

    def __post_init__(self):
        if self.scores is None:
            self.scores = []
        if self.goals is None:
            self.goals = []

    @property
    def current_goal(self):
        return self.goals[self.current_goal_index] if self.goals else None


# Global variables
game_state = GameState()
brain = Dqn(6, 3, 0.9)
action2rotation = [0, 5, -5]
im = CoreImage("./images/MASK1.png")


def init_game_state():
    """Initialize the game state with the map and initial positions"""
    global game_state
    img = PILImage.open("./images/mask.png").convert("L")
    game_state.sand = np.asarray(img) / 255
    game_state.first_update = False
    logger.info("Game state initialized")


# Initializing the map
first_update = True


def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert("L")
    sand = np.asarray(img) / 255
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class


class Car(Widget):
    """Car class representing the AI-controlled vehicle"""

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        """Update car position and sensor readings"""
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self._update_sensors()
        self._update_signals()

    def _update_sensors(self):
        """Update sensor positions based on car angle"""
        self.sensor1 = Vector(SENSOR_DISTANCE, 0).rotate(self.angle) + self.pos
        self.sensor2 = (
            Vector(SENSOR_DISTANCE, 0).rotate((self.angle + 30) % 360) + self.pos
        )
        self.sensor3 = (
            Vector(SENSOR_DISTANCE, 0).rotate((self.angle - 30) % 360) + self.pos
        )

    def _update_signals(self):
        """Update sensor signals based on sand detection"""
        signals = []
        for sensor, signal in [
            (self.sensor1, "signal1"),
            (self.sensor2, "signal2"),
            (self.sensor3, "signal3"),
        ]:
            if self._is_sensor_out_of_bounds(sensor):
                signal_value = 10.0
            else:
                signal_value = self._calculate_signal_value(sensor)
            setattr(self, signal, signal_value)
            signals.append(signal_value)

        logger.debug(f"Sensor signals: {signals}")

    def _is_sensor_out_of_bounds(self, sensor):
        """Check if sensor is out of map bounds"""
        return (
            sensor[0] > WINDOW_WIDTH - SENSOR_RANGE
            or sensor[0] < SENSOR_RANGE
            or sensor[1] > WINDOW_HEIGHT - SENSOR_RANGE
            or sensor[1] < SENSOR_RANGE
        )

    def _calculate_signal_value(self, sensor):
        """Calculate signal value based on sand detection"""
        x, y = int(sensor[0]), int(sensor[1])
        return (
            int(
                np.sum(
                    game_state.sand[
                        x - SENSOR_RANGE : x + SENSOR_RANGE,
                        y - SENSOR_RANGE : y + SENSOR_RANGE,
                    ]
                )
            )
            / 400.0
        )


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


# Creating the game class


class GoalOrb(Widget):
    """Widget to display goal positions as colored orbs"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size = (20, 20)  # Size of the orb
        self.color = (0, 1, 0, 1)  # Green color with full opacity
        self._update_canvas()

    def _update_canvas(self):
        """Update the orb's visual representation"""
        self.canvas.clear()
        with self.canvas:
            Color(*self.color)
            Ellipse(pos=self.pos, size=self.size)

    def set_color(self, r, g, b, a=1):
        """Set the orb's color"""
        self.color = (r, g, b, a)
        self._update_canvas()


class StateDisplay(Widget):
    """Widget to display game state information"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (200, 150)
        self.pos = (10, WINDOW_HEIGHT - 160)
        self._init_labels()

    def _init_labels(self):
        """Initialize state display labels"""
        self.labels = {}
        states = ["score", "distance", "reward", "epsilon", "goal"]
        for i, state in enumerate(states):
            label = Label(
                text=f"{state}: 0",
                pos=(self.pos[0], self.pos[1] + 120 - i * 25),
                size=(200, 25),
                color=(1, 0, 0, 1),
            )
            self.labels[state] = label
            self.add_widget(label)

    def update(self, game_state, brain):
        """Update state display with current values"""
        self.labels["score"].text = f"Score: {brain.score():.2f}"
        self.labels["distance"].text = f"Distance: {game_state.last_distance:.1f}"
        self.labels["reward"].text = f"Reward: {game_state.last_reward:.2f}"
        self.labels["epsilon"].text = f"Epsilon: {brain.epsilon:.2f}"
        self.labels["goal"].text = (
            f"Goal: {game_state.current_goal_index + 1}/{len(game_state.goals)}"
        )


class Game(Widget):
    """Main game class handling the game logic"""

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    state_display = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state_display = StateDisplay()
        self.add_widget(self.state_display)
        self._init_goals()

    def _init_goals(self):
        """Initialize goals with hardcoded positions"""
        game_state.goals = GOAL_POSITIONS

        # Create goal orbs
        self.goal_orbs = []
        for i, goal in enumerate(game_state.goals):
            orb = GoalOrb()
            orb.pos = (goal[0] - 10, goal[1] - 10)
            orb.set_color(0, 1, 0) if i == 0 else orb.set_color(0.5, 0.5, 0.5)
            self.goal_orbs.append(orb)
            self.add_widget(orb)
            logger.info(f"Initialized goal {i+1} at position {goal}")

    def serve_car(self):
        """Initialize car position"""
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        """Main update loop"""
        if game_state.first_update:
            init_game_state()
            logger.info("Game initialized")

        self._update_dimensions()
        self._process_ai_decision()
        self._update_car_position()
        self._check_boundaries()
        self._check_goal_reached()

        # Log summary every 100 frames
        if int(time.time() * 30) % 100 == 0:  # Assuming 30 FPS
            logger.info(
                f"Game State Summary - "
                f"Position: ({self.car.x:.1f}, {self.car.y:.1f}), "
                f"Distance to goal: {self._calculate_distance():.2f}, "
                f"Last reward: {game_state.last_reward:.2f}, "
                f"Score: {brain.score():.2f}"
            )

    def _update_dimensions(self):
        """Update game dimensions"""
        global longueur, largeur
        longueur = self.width
        largeur = self.height

    def _process_ai_decision(self):
        """Process AI decision and update car movement"""
        orientation = self._calculate_orientation()
        distance = self._calculate_distance()
        last_signal = self._get_sensor_signals(orientation, distance)
        action = brain.update(game_state.last_reward, last_signal)
        game_state.scores.append(brain.score())
        rotation = action2rotation[action]

        logger.debug(
            f"AI Decision - Action: {action}, Rotation: {rotation}, "
            f"Score: {brain.score():.2f}, Orientation: {orientation:.2f}, "
            f"Distance: {distance:.2f}"
        )

        self.car.move(rotation)
        self.state_display.update(game_state, brain)

    def _calculate_orientation(self):
        """Calculate orientation towards goal"""
        if not game_state.current_goal:
            return 0
        xx = game_state.current_goal[0] - self.car.x
        yy = game_state.current_goal[1] - self.car.y
        return Vector(*self.car.velocity).angle((xx, yy)) / 180.0

    def _get_sensor_signals(self, orientation, distance):
        """Get all sensor signals including distance to goal"""
        return [
            self.car.signal1,
            self.car.signal2,
            self.car.signal3,
            orientation,
            -orientation,
            distance / WINDOW_WIDTH,  # Normalized distance
        ]

    def _update_car_position(self):
        """Update car position and check for sand"""
        distance = self._calculate_distance()
        self._update_sensor_balls()

        if self._is_on_sand():
            self._handle_sand_collision()
        else:
            self._handle_normal_movement(distance)

        game_state.last_distance = distance

    def _calculate_distance(self):
        """Calculate distance to current goal"""
        if not game_state.current_goal:
            return 0
        return np.sqrt(
            (self.car.x - game_state.current_goal[0]) ** 2
            + (self.car.y - game_state.current_goal[1]) ** 2
        )

    def _update_sensor_balls(self):
        """Update sensor ball positions"""
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

    def _is_on_sand(self):
        """Check if car is on sand"""
        return game_state.sand[int(self.car.x), int(self.car.y)] > 0

    def _handle_sand_collision(self):
        """Handle car movement on sand"""
        self.car.velocity = Vector(CAR_SPEED_SAND, 0).rotate(self.car.angle)
        game_state.last_reward = SAND_REWARD
        logger.info(
            f"Sand collision at ({int(self.car.x)}, {int(self.car.y)}) - "
            f"Reward: {SAND_REWARD}, Speed: {CAR_SPEED_SAND}"
        )

    def _handle_normal_movement(self, distance):
        """Handle normal car movement"""
        self.car.velocity = Vector(CAR_SPEED_NORMAL, 0).rotate(self.car.angle)
        game_state.last_reward = ROAD_REWARD
        if distance < game_state.last_distance:
            game_state.last_reward = CLOSER_TO_GOAL_REWARD
            logger.info(
                f"Moving closer to goal - Distance: {distance:.2f}, "
                f"Last distance: {game_state.last_distance:.2f}, "
                f"Reward: {CLOSER_TO_GOAL_REWARD}"
            )
        else:
            logger.debug(
                f"Normal movement - Distance: {distance:.2f}, "
                f"Last distance: {game_state.last_distance:.2f}, "
                f"Reward: {ROAD_REWARD}"
            )

    def _check_boundaries(self):
        """Check and handle boundary collisions"""
        old_x, old_y = self.car.x, self.car.y
        hit_boundary = False

        if self.car.x < BOUNDARY_PADDING:
            self.car.x = BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
            hit_boundary = True
        elif self.car.x > self.width - BOUNDARY_PADDING:
            self.car.x = self.width - BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
            hit_boundary = True
        if self.car.y < BOUNDARY_PADDING:
            self.car.y = BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
            hit_boundary = True
        elif self.car.y > self.height - BOUNDARY_PADDING:
            self.car.y = self.height - BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
            hit_boundary = True

        if hit_boundary:
            logger.info(
                f"Boundary collision - Position: ({old_x:.1f}, {old_y:.1f}) -> "
                f"({self.car.x:.1f}, {self.car.y:.1f}), Reward: {BOUNDARY_REWARD}"
            )

    def _check_goal_reached(self):
        """Check if goal is reached and update to next goal"""
        distance = self._calculate_distance()
        if distance < GOAL_DISTANCE_THRESHOLD:
            # Update goal orbs
            self.goal_orbs[game_state.current_goal_index].set_color(
                0, 0, 1
            )  # Blue for reached
            game_state.current_goal_index = (game_state.current_goal_index + 1) % len(
                game_state.goals
            )
            self.goal_orbs[game_state.current_goal_index].set_color(
                0, 1, 0
            )  # Green for current
            for i in range(len(self.goal_orbs)):
                if i != game_state.current_goal_index:
                    self.goal_orbs[i].set_color(0.5, 0.5, 0.5)  # Gray for others

            # Randomly respawn car
            self.serve_car()

            logger.info(
                f"Goal reached! - Distance: {distance:.2f}, "
                f"New goal index: {game_state.current_goal_index}, "
                f"Car position: ({self.car.x:.1f}, {self.car.y:.1f})"
            )
            game_state.last_reward = GOAL_REWARD
        else:
            logger.debug(
                f"Distance to goal: {distance:.2f}, "
                f"Car position: ({self.car.x:.1f}, {self.car.y:.1f}), "
                f"Goal: {game_state.current_goal}"
            )


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 30.0)
        clearbtn = Button(text="clear", size=(40, 40))
        savebtn = Button(text="save", size=(40, 40), pos=(parent.width, 0))
        loadbtn = Button(text="load", size=(40, 40), pos=(2 * parent.width, 0))
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(game_state.scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()


# Running the whole thing
if __name__ == "__main__":
    CarApp().run()
