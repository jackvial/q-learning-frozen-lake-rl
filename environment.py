import re
import curses
import numpy as np
import collections
import warnings
from typing import Optional
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

warnings.filterwarnings("ignore")


class FrozenLakeEnvCustom(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        self.curses_screen = curses.initscr()
        curses.start_color()
        curses.curs_set(0)
        self.curses_color_pairs = self.build_ncurses_color_pairs()

        # Blocking reads
        self.curses_screen.timeout(-1)

        super().__init__(
            render_mode=render_mode,
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
        )

    def build_ncurses_color_pairs(self):
        """
        Based on Deepmind Pycolab https://github.com/deepmind/pycolab/blob/master/pycolab/human_ui.py
        """

        color_fg = {
            " ": (0, 0, 0),
            "S": (368, 333, 388),
            "H": (309, 572, 999),
            "P": (999, 364, 0),
            "F": (500, 999, 948),
            "G": (999, 917, 298),
            "?": (368, 333, 388),
            "←": (309, 572, 999),
            "↓": (999, 364, 0),
            "→": (500, 999, 948),
            "↑": (999, 917, 298),
        }

        color_pair = {}

        cpair_0_fg_id, cpair_0_bg_id = curses.pair_content(0)
        ids = set(range(curses.COLORS - 1)) - {
            cpair_0_fg_id,
            cpair_0_bg_id,
        }

        # We use color IDs from large to small.
        ids = list(reversed(sorted(ids)))

        # But only those color IDs we actually need.
        ids = ids[: len(color_fg)]
        color_ids = dict(zip(color_fg.values(), ids))

        # Program these colors into curses.
        for color, cid in color_ids.items():
            curses.init_color(cid, *color)

        # Now add the default colors to the color-to-ID map.
        cpair_0_fg = curses.color_content(cpair_0_fg_id)
        cpair_0_bg = curses.color_content(cpair_0_bg_id)
        color_ids[cpair_0_fg] = cpair_0_fg_id
        color_ids[cpair_0_bg] = cpair_0_bg_id

        # The color pair IDs we'll use for all characters count up from 1; note that
        # the "default" color pair of 0 is already defined, since _color_pair is a
        # defaultdict.
        color_pair.update(
            {character: pid for pid, character in enumerate(color_fg, start=1)}
        )

        # Program these color pairs into curses, and that's all there is to do.
        for character, pid in color_pair.items():

            # Get foreground and background colors for this character. Note how in
            # the absence of a specified background color, the same color as the
            # foreground is used.
            cpair_fg = color_fg.get(character, cpair_0_fg_id)
            cpair_bg = color_fg.get(character, cpair_0_fg_id)

            # Get color IDs for those colors and initialise a color pair.
            cpair_fg_id = color_ids[cpair_fg]
            cpair_bg_id = color_ids[cpair_bg]
            curses.init_pair(pid, cpair_fg_id, cpair_bg_id)

        return color_pair

    def render_ncurses_ui(self, screen, board, color_pair, title, q_table):
        screen.erase()

        # Draw the title
        screen.addstr(0, 2, title)

        # Draw the game board
        for row_index, board_line in enumerate(board, start=1):
            screen.move(row_index, 2)
            for codepoint in "".join(list(board_line)):
                screen.addch(codepoint, curses.color_pair(color_pair[codepoint]))

        def action_to_char(action):
            if action == 0:
                return "←"
            elif action == 1:
                return "↓"
            elif action == 2:
                return "→"
            elif action == 3:
                return "↑"
            else:
                return "?"

        # Draw the action grid
        max_action_table = np.argmax(q_table, axis=1).reshape(4, 4)
        for row_index, row in enumerate(max_action_table, start=1):
            screen.move(row_index, 8)
            for action in row:
                char = action_to_char(action)
                screen.addch(char, curses.color_pair(color_pair[char]))

        # Draw the Q-table
        q_table_2d = q_table.reshape(4, 16)
        for row_index, row in enumerate(q_table_2d, start=1):
            screen.move(row_index, 14)
            for col_index, col in enumerate(row):
                action = col_index % 4
                char = action_to_char(action)
                screen.addstr(f" {col:.2f}", curses.color_pair(color_pair[char]))
                if action == 3:
                    screen.addstr(" ", curses.color_pair(color_pair[" "]))

        # Redraw the game screen (but in the curses memory buffer only).
        screen.noutrefresh()

    def ansi_frame_to_board(self, frame_string):
        parts = frame_string.split("\n")
        board = []
        p = "\x1b[41m"
        for part in parts[1:]:
            if len(part):
                row = re.findall(r"S|F|H|G", part)
                try:
                    row[part.index(p)] = "P"
                except:
                    pass
                board.append(row)

        return np.array(board)

    def render(self, title=None, q_table=None):
        if self.render_mode == "curses":
            frame = self._render_text()

            board = self.ansi_frame_to_board(frame)
            self.render_ncurses_ui(
                self.curses_screen, board, self.curses_color_pairs, title, q_table
            )

            # Show the screen to the user.
            curses.doupdate()
            return board

        return super().render()

    def get_expected_new_state_for_action(self, action):
        return self.P[self.s][action][1][1]
