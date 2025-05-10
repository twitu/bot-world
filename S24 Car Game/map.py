# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
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
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
    logger = logging.getLogger('car_game')
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
GOAL_POSITIONS = {"start": (9, 85), "end": (1420, 622)}
BOUNDARY_REWARD = -1
SAND_REWARD = -0.5
ROAD_REWARD = -0.2
CLOSER_TO_GOAL_REWARD = 0.1
GOAL_REWARD = 1

# Kivy Configuration
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "resizable", False)
Config.set("graphics", "width", str(WINDOW_WIDTH))
Config.set("graphics", "height", str(WINDOW_HEIGHT))


@dataclass
class GameState:
    """Class to hold the game state"""

    sand: np.ndarray = None
    goal_x: int = GOAL_POSITIONS["start"][0]
    goal_y: int = GOAL_POSITIONS["start"][1]
    last_distance: float = 0
    last_reward: float = 0
    scores: List[float] = None
    swap: int = 0
    first_update: bool = True

    def __post_init__(self):
        if self.scores is None:
            self.scores = []


# Global variables
game_state = GameState()
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 5, -5]
im = CoreImage("./images/MASK1.png")


def init_game_state():
    """Initialize the game state with the map and initial positions"""
    global game_state
    img = PILImage.open("./images/mask.png").convert("L")
    game_state.sand = np.asarray(img) / 255
    game_state.goal_x, game_state.goal_y = GOAL_POSITIONS["start"]
    game_state.first_update = False
    game_state.swap = 0
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


class Game(Widget):
    """Main game class handling the game logic"""

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    start_goal = ObjectProperty(None)
    end_goal = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_goals()

    def _init_goals(self):
        """Initialize goal orbs"""
        # Create start goal orb
        self.start_goal = GoalOrb()
        self.start_goal.pos = (GOAL_POSITIONS["start"][0] - 10, GOAL_POSITIONS["start"][1] - 10)
        self.start_goal.set_color(0, 1, 0)  # Green for start
        self.add_widget(self.start_goal)

        # Create end goal orb
        self.end_goal = GoalOrb()
        self.end_goal.pos = (GOAL_POSITIONS["end"][0] - 10, GOAL_POSITIONS["end"][1] - 10)
        self.end_goal.set_color(1, 0, 0)  # Red for end
        self.add_widget(self.end_goal)

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
        last_signal = self._get_sensor_signals(orientation)
        action = brain.update(game_state.last_reward, last_signal)
        game_state.scores.append(brain.score())
        rotation = action2rotation[action]
        
        logger.debug(
            f"AI Decision - Action: {action}, Rotation: {rotation}, "
            f"Score: {brain.score():.2f}, Orientation: {orientation:.2f}"
        )
        
        self.car.move(rotation)

    def _calculate_orientation(self):
        """Calculate orientation towards goal"""
        xx = game_state.goal_x - self.car.x
        yy = game_state.goal_y - self.car.y
        return Vector(*self.car.velocity).angle((xx, yy)) / 180.0

    def _get_sensor_signals(self, orientation):
        """Get all sensor signals"""
        return [
            self.car.signal1,
            self.car.signal2,
            self.car.signal3,
            orientation,
            -orientation,
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
        """Calculate distance to goal"""
        return np.sqrt(
            (self.car.x - game_state.goal_x) ** 2
            + (self.car.y - game_state.goal_y) ** 2
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
        """Check if goal is reached and update goal position"""
        distance = self._calculate_distance()
        if distance < GOAL_DISTANCE_THRESHOLD:
            old_goal = (game_state.goal_x, game_state.goal_y)
            if game_state.swap == 1:
                game_state.goal_x, game_state.goal_y = GOAL_POSITIONS["end"]
                game_state.swap = 0
                self.start_goal.set_color(0, 1, 0)  # Green
                self.end_goal.set_color(1, 0, 0)    # Red
            else:
                game_state.goal_x, game_state.goal_y = GOAL_POSITIONS["start"]
                game_state.swap = 1
                self.start_goal.set_color(1, 0, 0)  # Red
                self.end_goal.set_color(0, 1, 0)    # Green
            
            logger.info(
                f"Goal reached! - Distance: {distance:.2f}, "
                f"Old goal: {old_goal}, New goal: ({game_state.goal_x}, {game_state.goal_y}), "
                f"Car position: ({self.car.x:.1f}, {self.car.y:.1f})"
            )
        else:
            logger.debug(
                f"Distance to goal: {distance:.2f}, "
                f"Car position: ({self.car.x:.1f}, {self.car.y:.1f}), "
                f"Goal: ({game_state.goal_x}, {game_state.goal_y})"
            )


# Adding the painting tools


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.0
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8") * 255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == "left":
            touch.ud["line"].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.0
            density = n_points / (length)
            touch.ud["line"].width = int(20 * density + 1)
            sand[
                int(touch.x) - 10 : int(touch.x) + 10,
                int(touch.y) - 10 : int(touch.y) + 10,
            ] = 1

            last_x = x
            last_y = y


# Adding the API Buttons (clear, save and load)


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 30.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text="clear", size=(40, 40))
        savebtn = Button(text="save", size=(40, 40), pos=(parent.width, 0))
        loadbtn = Button(text="load", size=(40, 40), pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
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
