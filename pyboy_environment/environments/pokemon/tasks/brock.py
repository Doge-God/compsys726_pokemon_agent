from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

import collections
import cv2
import numpy.typing as npt
from typing import TypedDict, List


class GameStats(TypedDict):
    location: any
    battle_type: str
    current_pokemon_id: int
    current_pokemon_health: int
    enemy_pokemon_health: int
    party_size: int
    ids: List[int]
    pokemon: List[dict]  # Assuming pkc.get_pokemon(id) returns a dict or object, you can adjust accordingly
    levels: List[int]
    type_id: List[int]
    type: List[str]  # Assuming pkc.get_type(id) returns a string, modify as needed
    hp: List[int]
    xp: List[int]
    status: List[str]  # Modify if the status is a different data type
    badges: int
    caught_pokemon: int
    seen_pokemon: int
    money: int
    events: List[str]  # Assuming events are strings, adjust if needed
    items: List[str]   # Assuming items are strings, adjust if needed

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START,
        ]

        self.image_to_stack = collections.deque([],maxlen=3)
        self.max_level_sum = 0
        self.image_len = 84

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

        self.REMAPPED_MAPS = {
            "OAKS_LAB,": 0.07,
            "PALLET_TOWN,": 0.14, 
            "ROUTE_1,": 0.21,
            "VIRIDIAN_CITY,": 0.28,
            "VIRIDIAN_POKECENTER,": 0.35,
            "VIRIDIAN_MART,": 0.42,
            "ROUTE_2,": 0.49,
            "VIRIDIAN_FOREST_SOUTH_GATE,": 0.56,
            "VIRIDIAN_FOREST,": 0.63,
            "VIRIDIAN_FOREST_NORTH_GATE,": 0.7,
            "PEWTER_CITY,": 0.77, 
            "PEWTER_GYM,": 0.83, 
            "PEWTER_MART,": 0.9, 
        }

    # POSITION
    # {
    #     "x": x_pos,
    #     "y": y_pos,
    #     "map_id": map_n,
    #     "map": pkc.get_map_location(map_n),
    # }

    def _get_state(self) -> np.ndarray:
        # Implement your state retrieval logic here
        game_stats = self._generate_game_stats()

        # frame = np.array(self.screen.image) #144*160
        # frame:npt.NDArray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame.resize((1,144,160))

        frame = np.array(self.screen.image)
        frame = cv2.resize(frame, (self.image_len, self.image_len))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame.resize((self.image_len, self.image_len, 1))

        cv2.imshow('image window', frame)
        frame = np.moveaxis(frame, -1, 0)
        
        #======  STACK LOGIC  ==============================
        # try:
        #     if len(self.image_to_stack) != 3:
        #         for _ in range(0,3):
        #             self.image_to_stack.append(frame)
        #     else:
        #         self.image_to_stack.append(frame)
        # except:
        #     pass
        
        # stacked_frames = np.concatenate(list(self.image_to_stack), axis=0)
        
        battle_type = self._read_battle_type()

        info_vector = [
            game_stats['location']['x'], 
            game_stats['location']['y'],
            game_stats['location']['map_id'],
            self._get_pokeball_count(self._read_items_()), 
            battle_type,
            game_stats["current_pokemon_health"] if battle_type != 0 else 0,
            game_stats["enemy_pokemon_health"] if battle_type != 0 else 0,
            self._get_enemy_catch_rate() if battle_type == 1 else 0
        ]

        # print(info_vector)
        
        return {
            "image" : frame,
            "vector": info_vector
        }
    

    
    def reset(self) -> np.ndarray:
        self.steps = 0

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.prior_game_stats = self._generate_game_stats()

        # try:
        #     self.image_to_stack.clear()
        #     self.max_level_sum = 0
        # except:
        #     print("No image_to_stack")

        return self._get_state()
    
    @cached_property
    def observation_space(self) -> int:
        # return self._get_state().shape()
        # return (144,160)
        return {
            "image": (1, self.image_len, self.image_len),
            #       coord | num poke ball | battle flag ||| my poke hp, other poke hp, catch rate
            "vector": 3 +        1+            1+                1+         1+             1
        }

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        reward = -1
        reward += self._levels_reward(new_state)
        reward += self._grass_reward(new_state) * 0.5 #0.5 for touching grass
        reward += self._start_battle_reward(new_state) * 200
        reward += self._pokeball_thrown_reward(new_state) * 100
        reward += self._caught_reward(new_state) * 500
        reward += self._bought_pokeball_reward(new_state) * 100

        if not new_state["location"]["map"] in ["OAKS_LAB,","PALLET_TOWN,", "ROUTE_1,","VIRIDIAN_CITY,","VIRIDIAN_POKECENTER,","VIRIDIAN_MART,","ROUTE_2,","VIRIDIAN_FOREST_SOUTH_GATE,","VIRIDIAN_FOREST,","VIRIDIAN_FOREST_NORTH_GATE,","PEWTER_CITY,", "PEWTER_GYM,", "PEWTER_MART,", ]:
            reward -= 500

        return reward


    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        is_truncated = False

        if not game_stats["location"]["map"] in ["OAKS_LAB,","PALLET_TOWN,", "ROUTE_1,","VIRIDIAN_CITY,","VIRIDIAN_POKECENTER,","VIRIDIAN_MART,","ROUTE_2,","VIRIDIAN_FOREST_SOUTH_GATE,","VIRIDIAN_FOREST,","VIRIDIAN_FOREST_NORTH_GATE,","PEWTER_CITY,", "PEWTER_GYM,", "PEWTER_MART,", ]:
            is_truncated = True

        if self.steps >= 1000:
            is_truncated = True

        return is_truncated
    
    ################################################################
    ###### EXTEND GATHER DATA ########################################

    def _get_pokeball_count(self, items) -> int:
        total_count = 0

        # Iterate through the dictionary of items the player (keys) has and their counts (values)
        for itemId, count in items.items():
        # Iterate through the types of Pokeballs. If the item (key) matches any of the Pokeball type ids, add the count to the total number of Pokeballs
            if itemId in range(0x0, 0x5):
                total_count += count
        
        return total_count
        
    def _read_items_(self) -> dict:
        total_items = self._read_m(0xD31D)
        if (total_items == 0):
            return {}

        addr = 0xD31E
        items = {}

        for i in range(total_items):
            item_id = self._read_m(addr + 2 * i)
            item_count = self._read_m(addr + 2 * i + 1)
            items[item_id] = item_count

        return items
    
    def _get_enemy_catch_rate(self) -> int:
        return self._read_m(0xD007)

    def _get_active_pokemon_id(self) -> int:
        return self._read_m(0xD014)

    def _get_enemy_pokemon_health(self) -> int:
        return self._read_hp(0xCFE6)

    def _get_current_pokemon_health(self) -> int:
        return self._read_hp(0xD015)
    
    def _read_battle_type(self) -> int:
        return self._read_m(0xD057)
    
    def _generate_game_stats(self) -> GameStats:
        # debug-log logging.info("Logging124")
        stats:GameStats = {
            "location": self._get_location(),
            "battle_type": self._read_battle_type(),
            "current_pokemon_id": self._get_active_pokemon_id(),
            "current_pokemon_health": self._get_current_pokemon_health(),
            "enemy_pokemon_health": self._get_enemy_pokemon_health(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            "type_id": self._read_party_type(),
            "type": [pkc.get_type(id) for id in self._read_party_type()],
            "hp": self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            "caught_pokemon": self._read_caught_pokemon_count(),
            "seen_pokemon": self._read_seen_pokemon_count(),
            "money": self._read_money(),
            "events": self._read_events(),
            "items": self._read_items_(),
        }
        return stats
    

    #######################################################################
    ################ EXTEND REWARD ##########################################

    def _pokeball_thrown_reward(self, new_state) -> int:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])


        if previous_count > new_count:
            return 1
        else:
            return 0

    def _bought_pokeball_reward(self, new_state) -> int:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])

        if new_count > previous_count:
            return 1
        else:
            return 0

    def _start_battle_reward(self, new_state, battle_type=1) -> int:
        if (new_state["battle_type"] == battle_type and self.prior_game_stats["battle_type"] == 0):
            return 1
        return 0

    def _pokeball_thrown_reward(self, new_state) -> int:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])

        if previous_count > new_count:
            return 1
        else:
            return 0

    def _bought_pokeball_reward(self, new_state) -> int:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])

        if new_count > previous_count:
            return 1
        else:
            return 0
    

