# way more discrete states, minimize effect of animation
# reduce reward for starting battle
# slight penalty for same screen
# truncate based on 1000 steps no xp

# better map normalization
# map direction reward
# penalize run away

# pp

# reward for reset
# directives




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
            act_freq=330, ######################################################################## HERE THIS MF
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
            "ROUTE_2,": 0.49,
            "VIRIDIAN_FOREST_SOUTH_GATE,": 0.56,
            "VIRIDIAN_FOREST,": 0.63,
            "VIRIDIAN_FOREST_NORTH_GATE,": 0.7,
            "PEWTER_CITY,": 0.77, 
            "PEWTER_GYM,": 0.83, 
        }
        
        # 5 # 11
        #"ROUTE_2,","VIRIDIAN_FOREST_SOUTH_GATE,","VIRIDIAN_FOREST,","VIRIDIAN_FOREST_NORTH_GATE,","PEWTER_CITY,", "PEWTER_GYM,",
        self.ALLOWED_MAP = ["OAKS_LAB,","PALLET_TOWN,", "ROUTE_1,","VIRIDIAN_CITY,","VIRIDIAN_POKECENTER,", "ROUTE_2,","VIRIDIAN_FOREST_SOUTH_GATE,","VIRIDIAN_FOREST,","VIRIDIAN_FOREST_NORTH_GATE,","PEWTER_CITY,", "PEWTER_GYM,"]
        self.IS_DISCRETE_ACTION = False

        ##################################################################

        self.past_loc = None
        self.same_loc_cnt = 0

        self.last_img = np.zeros((self.image_len, self.image_len))
        self.last_img_diff_cnt = 100
        self.same_img_cnt = 0

        self.truncate_cnt = 1000

        ###################################################################

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


        # allow loiter if at pokemon center or in battle
        if game_stats['battle_type'] == 0 and game_stats['location']['map'] != "VIRIDIAN_POKECENTER,": 
            if self.past_loc == game_stats['location']:
                self.same_loc_cnt += 1
            else:
                self.same_loc_cnt = 0
        else:
            self.same_loc_cnt = 0

        self.past_loc = game_stats['location']
        frame = np.array(self.screen.image)
        frame = cv2.resize(frame, (self.image_len, self.image_len))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame.resize((1,self.image_len, self.image_len))


        self.last_img_diff_cnt = np.sum(self.last_img != frame)
        self.last_img = frame

        if self.last_img_diff_cnt <= 15:
            self.same_img_cnt += 1
        else:
            self.same_img_cnt = 0

        # cv2.imshow('image window', frame)
        # frame = np.moveaxis(frame, -1, 0)
        
        #======  STACK LOGIC  ==============================
        try:
            if len(self.image_to_stack) != 3:
                for _ in range(0,3):
                    self.image_to_stack.append(frame)
            else:
                self.image_to_stack.append(frame)
        except:
            pass
        
        stacked_frames = np.concatenate(list(self.image_to_stack), axis=0)
        
        battle_type = self._read_battle_type()

        try:
            map_index_normalized = self.ALLOWED_MAP.index(game_stats['location']['map'])/10
        except:
            print("ENTERED OUT OF BOUND MAP")
            map_index_normalized = -1

        selected_move_pp = 0
        if game_stats['battle_type'] != 0 and game_stats['menu_type'] == 199:
            selected_move_pp = game_stats['current_pp'][game_stats['menu_item']-1]
        # print(selected_move_pp)

        x,y,idx = self.process_loc(game_stats)

        # fight_flag = 0 
        # if self.directive == 0: # level up directive, always fight
        #     fight_flag = 1
        # else: # on the way to brock
        #     if battle_type == 2:
        #         fight_flag = 1 # only fight gym battles
        #     else:
        #         fight_flag = 0 # run from wild fights

        if battle_type == 0:
            fight_flag = 0
        else:
            fight_flag = 1

        
        
            
        info_vector = [
            # 1 byte int, normalize by max val
            game_stats['directive'] if battle_type == 0 else 0,
            x,
            y,
            idx, 
            fight_flag,
            1 if selected_move_pp > 0 else 0, # indicator for whether selected move has pp, only active when on that menu
            game_stats["should_fight"] if battle_type != 0 else 0 # should fight flag
        ]

        # print("-----------------------------------------------------------")
        # print(f"({round(game_stats['location']['x'],2)}, {round(game_stats['location']['y'],2)}, {round(map_index_normalized,2)}), BATTLE: {battle_type}, DIFF: {self.last_img_diff_cnt}")
        # print(f"Selected move PP: {selected_move_pp if fight_flag != 0 else 0}")
        # print(game_stats['location']['map'])
        # print(f"selected menu item {self._read_m(0xCC26)}")
        # print(f"Bitmask current menu? {self._read_m(0xCC29)}")
        

        return {
            "image" : stacked_frames,
            "vector": info_vector
        }
    

    
    def reset(self) -> np.ndarray:
        self.steps = 0

        with open(self.init_path, "rb") as f:
            self.pyboy.load_state(f)

        self.prior_game_stats = self._generate_game_stats()

        ##################################################################
        self.past_loc = None
        self.same_loc_cnt = 0

        self.last_img = np.zeros((self.image_len, self.image_len))
        self.last_img_diff_cnt = 100
        self.same_img_cnt = 0

        self.truncate_cnt = 1000

        ###################################################################

        try:
            self.image_to_stack.clear()
            self.max_level_sum = 0
        except:
            print("No image_to_stack")

        return self._get_state()
    
    @cached_property
    def observation_space(self) -> int:
        # return self._get_state().shape()
        # return (144,160)
        return {
            "image": (3, self.image_len, self.image_len),
            #       directive | coord | in battle | should fight | pp  | 
            "vector": 1+         3 +        1+          1  +        1     
        }
    
    #       coord | num poke ball | battle flag ||| my poke hp, other poke hp, catch rate
    #        "vector": 3 +        1+            1+                1+         1+             1

    def _calculate_reward(self, new_state: dict) -> float:
        # Implement your reward calculation logic here
        raw_xp_reward = self._xp_increase_reward(new_state)
        raw_heal_reward = self._heal_reward(new_state)
        directive = self.prior_game_stats["directive"] 
        raw_map_prog_reward = self.map_progress_reward(new_state)
        

        reward = -1
        
        # NAV REWARD FOR STARTING WILD FIGHTS
        if directive == 0: # fight to level up
            reward += self._grass_reward(new_state) * 0.5 #0.5 for touching grass
            reward += self._start_battle_reward(new_state) * 10 # normal battle
        
        # DIRECTIVE UPDATED
        elif directive == 0 and new_state["directive"] == 1:
            reward += 500

        # NAV REWARD FOR TRYING TO FIGHT BROCK
        else:
            reward += raw_map_prog_reward * 5000
            reward += self._start_battle_reward(new_state,2) * 5000


        # print(f"PASS TURN: {self._pass_turn_reward(new_state)}")
        # print(f"Healed: {raw_heal_reward}")
        # print(f"Truncate cnt: {self.truncate_cnt}")

        # BATTLE REWARD
        if self.prior_game_stats['should_fight'] == 1: # should fight
            reward += raw_xp_reward * 15
            reward -= self._run_away_reward(new_state) * 10
            reward += raw_heal_reward * 500
            reward += self._levels_reward(new_state) * 500
            reward += self._enemy_health_decrease_reward(new_state) * 10
            reward += self._pass_turn_reward(new_state) * 10
        else: # shouldnt fight, only reward for running
            reward -= 1
            reward += self._run_away_reward(new_state) * 10

        
        #### TRUNCATE COUNTER MANAGEMENT
        if raw_heal_reward != 0: # extend truncation if healed
            self.truncate_cnt += 1000

        # reset truncate count if xp gained
        if directive == 0:
            if raw_xp_reward == 0:
                self.truncate_cnt -= 1
            else:
                self.truncate_cnt = 1000

        if directive == 1:
            if raw_map_prog_reward == 0:
                self.truncate_cnt -= 1
            else:
                self.truncate_cnt = 1500
            
            if self._start_battle_reward(new_state,2) != 0: #started trainer battle
                self.truncate_cnt = 1000
            else:
                self.truncate_cnt -= 1


        # GENERAL
        reward += self.map_gradual_progress_reward(new_state) * 0.2

        if not new_state["location"]["map"] in self.ALLOWED_MAP:
            reward =  -50
        
        if self.same_loc_cnt >= 10:
            reward -= 0.5
        
        if self.last_img_diff_cnt <= 3:
            reward -= 0.1

        # print(f"map_gradual {self.map_gradual_progress_reward(new_state)}")

        return reward


    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:

        if not game_stats["location"]["map"] in self.ALLOWED_MAP:
            return True

        if self.steps >= 10000:
            print("TRUNCATE: MAX STEP")
            return True

        if self.truncate_cnt <= 0:
            print("TRUNCATE: NO OBJECTIVE")
            return True
        
        if self.same_loc_cnt >= 70:
            print("TRUNCATE: SAME NOT MOVING")
            return True
        
        if self.same_img_cnt >= 70:
            print("TRUNCATE: SAME IMAGE")
            return True

        return False
    
    ################################################################
    ###### OVERRIDE ACTION ########################################
    
    def _run_action_on_emulator(self, action_array: np.ndarray) -> None:

        if self.IS_DISCRETE_ACTION:
            button = action_array[0]

            # Push the button for ONE frame and release
            self.pyboy.send_input(self.valid_actions[button])
            self.pyboy.tick(9,render=False)
            self.pyboy.send_input(self.release_button[button])
            self.pyboy.tick(self.act_freq+1, render=True) # ONLY render last frame
            
        else:
            action = action_array[0]
            action = min(action, 0.99)

            # Continuous Action is a float between 0 - 1 from Value based methods
            # We need to convert this to an action that the emulator can understand
            bins = np.linspace(0, 1, len(self.valid_actions) + 1)
            button = np.digitize(action, bins) - 1

            # Push the button for ONE frame and release
            self.pyboy.send_input(self.valid_actions[button])
            self.pyboy.tick(9,render=False)
            self.pyboy.send_input(self.release_button[button])
            self.pyboy.tick(self.act_freq+1, render=True) # ONLY render last frame

    ################################################################
    ###### EXTEND GATHER DATA ########################################

    def process_loc(self, game_stats):
        '''return x,y,idx, all normalized. 0 if in battle'''
        battle_type = game_stats['battle_type']
        x = game_stats['location']['x']/35 if battle_type == 0 else 0
        y = game_stats['location']['y']/35 if battle_type == 0 else 0

        try:
            map_index_normalized = self.ALLOWED_MAP.index(game_stats['location']['map'])/5
        except:
            map_index_normalized = -1
        
        idx = map_index_normalized if battle_type == 0 else 0
        
        return x,y,idx
    
    def calc_dist_closed(self, game_stats, goal_x, goal_y):
        x = game_stats['location']['x']
        y = game_stats['location']['y']

        old_x = self.prior_game_stats['location']['x']
        old_y = self.prior_game_stats['location']['y']

        x_prog = abs(goal_x-x) - abs(goal_x-old_x)
        y_prog = abs(goal_y-y) - abs(goal_y-old_y)

        tot_manh_prog = x_prog+y_prog

        if tot_manh_prog < 0:
            return 1
        else:
            return 0 


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
    
    def _read_current_pp(self):
        return [
            self._read_m(addr) for addr in [0xD02D, 0xD02E, 0xD02F, 0xD030]
        ]
    
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
    
    def _read_menu_type(self):
        return self._read_m(0xCC29)
    
    def _read_current_menu_item(self):
        return self._read_m(0xCC26)
    
    def _read_turn(self):
        '''From 0 to 1 is passing turn'''
        return self._read_m(0xFFF3)
    
    def calc_should_fight(self): 
        '''should fight or not, only active in battle'''
        should_fight = 0
        directive = self.calc_directive()
        if self._read_battle_type() != 0: # in battle
            if directive == 0: # training, always fight
                should_fight = 1
            else: # trying to fight brock
                if self.prior_game_stats["battle_type"] == 1:
                    should_fight = 0
                else:
                    should_fight = 1

        return should_fight
    
    def calc_directive(self):
        '''level 16'''
        if sum(self._read_party_level()) <= 16:
            return 0
        else:
            return 1
    
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
            "current_pp": self._read_current_pp(),
            "menu_type": self._read_menu_type(),
            "menu_item": self._read_current_menu_item(),
            "turn_flag": self._read_turn(),
            "should_fight": self.calc_should_fight(),
            "directive": self.calc_directive()
        }
        return stats
    

    #######################################################################
    ################ EXTEND REWARD ##########################################
    def map_progress_reward(self,new_state):
        old_state = self.prior_game_stats
        try:
            new_map_idx = self.ALLOWED_MAP.index(new_state['location']['map'])
            old_map_idx = self.ALLOWED_MAP.index(old_state['location']['map'])
        except:
            return 0
        

        if new_map_idx > old_map_idx or (new_state['location']['map']=="ROUTE_2," and old_state['location']['map']=="VIRIDIAN_FOREST_NORTH_GATE,"):
            return 1
        # backed up
        elif old_map_idx > new_map_idx and old_state["battle_type"] == 0:
            return -1


    def map_gradual_progress_reward(self, new_state):
        # ["OAKS_LAB,","PALLET_TOWN,", "ROUTE_1,","VIRIDIAN_CITY,","VIRIDIAN_POKECENTER,", 
        #"ROUTE_2,","VIRIDIAN_FOREST_SOUTH_GATE,","VIRIDIAN_FOREST,","VIRIDIAN_FOREST_NORTH_GATE,",
        #"PEWTER_CITY,", "PEWTER_GYM,"]
        old_state = self.prior_game_stats

        try:
            new_map_idx = self.ALLOWED_MAP.index(new_state['location']['map'])
            old_map_idx = self.ALLOWED_MAP.index(old_state['location']['map'])
        except:
            return 0
        
        
        # progressed in map
        if new_map_idx > old_map_idx or (new_state['location']['map']=="ROUTE_2," and old_state['location']['map']=="VIRIDIAN_FOREST_NORTH_GATE,"):
            return 0
        # backed up
        elif old_map_idx > new_map_idx and old_state["battle_type"] == 0:
            return 0

        # oak
        if new_state['location']['map'] == "OAKS_LAB,":
            if new_state['location']['y'] > old_state['location']['y']:
                return 1
            else:
                return 0
        # pallet
        if new_state['location']['map'] == "PALLET_TOWN,":
            return self.calc_dist_closed(new_state,11,0)
        
        # route 1 or route 2 just move up
        if new_state['location']['map'] == "ROUTE_1," or  new_state['location']['map'] == "ROUTE_2,":
            if new_state['location']['y'] < old_state['location']['y']:
                return 1
            else:
                return 0
        # viridian
        if new_state['location']['map'] == "VIRIDIAN_CITY,":
            return self.calc_dist_closed(new_state,17,0)
        
        # gates
        if new_state['location']['map'] == "VIRIDIAN_FOREST_SOUTH_GATE," or new_state['location']['map']=="VIRIDIAN_FOREST_NORTH_GATE,":
            return self.calc_dist_closed(new_state,5,1)

        # forest
        if new_state['location']['map'] == "VIRIDIAN_FOREST,":
            return self.calc_dist_closed(new_state,1,0)

        # pewt
        if new_state['location']['map'] == "PEWTER_CITY,":
            return self.calc_dist_closed(new_state,16,18)
        
        # gym
        if new_state['location']['map'] == "PEWTER_GYM,":
            return self.calc_dist_closed(new_state,4,1)

        return 0


    def _pokeball_thrown_reward(self, new_state) -> int:
        previous_count = self._get_pokeball_count(self.prior_game_stats["items"])
        new_count = self._get_pokeball_count(new_state["items"])


        if previous_count > new_count:
            return 1
        else:
            return 0
    
    def _heal_reward(self, new_state):
        # sum(new_state['hp']['current']) > sum(self.prior_game_stats["hp"]['current'])
        if sum(new_state['hp']['current']) > sum(self.prior_game_stats["hp"]['current']) and sum(self.prior_game_stats["hp"]['current']) == 0:
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
    
    def _enemy_health_decrease_reward(self, new_state: dict[str, any]) -> int:
        if (new_state["battle_type"] != 0 and self.prior_game_stats["battle_type"] != 0):
            previous_health = self.prior_game_stats["enemy_pokemon_health"]
            current_health = new_state["enemy_pokemon_health"]

            health_decrease = previous_health - current_health
            
            return max(0, health_decrease) # Doesn't punish when enemy health goes up
        return 0
    
    def _xp_increase_reward(self, new_state: dict[str, any]) -> int:
        return sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])
    
    
    def _run_away_reward(self,new_state):
        '''1 for the new stat as a result of running'''
        if new_state["battle_type"] == 0 and self.prior_game_stats['battle_type'] != 0 and self.prior_game_stats['menu_type'] == 33 and self.prior_game_stats["menu_item"] == 3:
            return 1
        else:
            return 0 
        
    def _pass_turn_reward(self,new_state):
        if new_state['turn_flag'] == 1 and self.prior_game_stats['turn_flag'] == 0:
            return 1
        else:
            return 0


    def _levels_increase_reward(self, new_state: dict[str, any]) -> int:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if (old_levels[i] != 0):
                reward += (new_levels[i] / old_levels[i] - 1)
        return reward

