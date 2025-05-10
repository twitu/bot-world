# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        for sensor, signal in [
            (self.sensor1, "signal1"),
            (self.sensor2, "signal2"),
            (self.sensor3, "signal3"),
        ]:
            if self._is_sensor_out_of_bounds(sensor):
                setattr(self, signal, 10.0)
            else:
                signal_value = self._calculate_signal_value(sensor)
                setattr(self, signal, signal_value)

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


class Game(Widget):
    """Main game class handling the game logic"""

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        """Initialize car position"""
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        """Main update loop"""
        if game_state.first_update:
            init_game_state()

        self._update_dimensions()
        self._process_ai_decision()
        self._update_car_position()
        self._check_boundaries()
        self._check_goal_reached()

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
        logger.debug(f"Sand collision at ({int(self.car.x)}, {int(self.car.y)})")

    def _handle_normal_movement(self, distance):
        """Handle normal car movement"""
        self.car.velocity = Vector(CAR_SPEED_NORMAL, 0).rotate(self.car.angle)
        game_state.last_reward = ROAD_REWARD
        if distance < game_state.last_distance:
            game_state.last_reward = GOAL_REWARD

    def _check_boundaries(self):
        """Check and handle boundary collisions"""
        if self.car.x < BOUNDARY_PADDING:
            self.car.x = BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
        elif self.car.x > self.width - BOUNDARY_PADDING:
            self.car.x = self.width - BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
        if self.car.y < BOUNDARY_PADDING:
            self.car.y = BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD
        elif self.car.y > self.height - BOUNDARY_PADDING:
            self.car.y = self.height - BOUNDARY_PADDING
            game_state.last_reward = BOUNDARY_REWARD

    def _check_goal_reached(self):
        """Check if goal is reached and update goal position"""
        if self._calculate_distance() < GOAL_DISTANCE_THRESHOLD:
            if game_state.swap == 1:
                game_state.goal_x, game_state.goal_y = GOAL_POSITIONS["end"]
                game_state.swap = 0
            else:
                game_state.goal_x, game_state.goal_y = GOAL_POSITIONS["start"]
                game_state.swap = 1
            logger.info(
                f"Goal reached! New goal: ({game_state.goal_x}, {game_state.goal_y})"
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
