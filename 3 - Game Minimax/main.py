from __future__ import annotations
import random
import time
import turtle
import math
import time
from copy import deepcopy
from enum import Enum


class OthelloUI:
    def __init__(self, board_size=6, square_size=60):
        self.board_size = board_size
        self.square_size = square_size
        self.screen = turtle.Screen()
        self.screen.setup(
            self.board_size * self.square_size + 50,
            self.board_size * self.square_size + 50,
        )
        self.screen.bgcolor("white")
        self.screen.title("Othello- low mist")
        self.pen = turtle.Turtle()
        self.pen.hideturtle()
        self.pen.speed(0)
        turtle.tracer(0, 0)

    def draw_board(self, board):
        self.pen.penup()
        x, y = (
            -self.board_size / 2 * self.square_size,
            self.board_size / 2 * self.square_size,
        )
        for i in range(self.board_size):
            self.pen.penup()
            for j in range(self.board_size):
                self.pen.goto(x + j * self.square_size, y - i * self.square_size)
                self.pen.pendown()
                self.pen.fillcolor("green")
                self.pen.begin_fill()
                self.pen.setheading(0)
                for _ in range(4):
                    self.pen.forward(self.square_size)
                    self.pen.right(90)
                self.pen.penup()
                self.pen.end_fill()
                self.pen.goto(
                    x + j * self.square_size + self.square_size / 2,
                    y - i * self.square_size - self.square_size + 5,
                )
                if board[i][j] == 1:
                    self.pen.fillcolor("white")
                    self.pen.begin_fill()
                    self.pen.circle(self.square_size / 2 - 5)
                    self.pen.end_fill()
                elif board[i][j] == -1:
                    self.pen.fillcolor("black")
                    self.pen.begin_fill()
                    self.pen.circle(self.square_size / 2 - 5)
                    self.pen.end_fill()

        turtle.update()


HUMAN, COMPUTER = 1, -1
Move = tuple[int, int]
TOTAL_TESTS = 20


class Othello:
    def __init__(self, ui=True, minimax_depth=1, prune=True):
        self.size = 6
        self.ui = OthelloUI(self.size) if ui else None
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.board[int(self.size / 2) - 1][int(self.size / 2) - 1] = self.board[int(self.size / 2)][int(self.size / 2)] = 1
        self.board[int(self.size / 2) - 1][int(self.size / 2)] = self.board[int(self.size / 2)][int(self.size / 2) - 1] = -1
        self.current_turn = random.choice([1, -1])
        self.minimax_depth = minimax_depth
        self.prune = prune
        self.CORNER_WEIGHT = 10
        self.BORDER_WEIGHT = 2
        self.TOTAL_WEIGHT = 1
        self.WIN_HEURISTIC = 1000
        self.seen_nodes = 0

    def set_minimax_depth(self, depth: int):
        self.minimax_depth = depth

    def set_pruning(self, prune: bool):
        self.prune = prune

    def get_winner(self):
        white_count = sum([row.count(HUMAN) for row in self.board])
        black_count = sum([row.count(COMPUTER) for row in self.board])
        if white_count > black_count:
            return HUMAN
        elif white_count < black_count:
            return COMPUTER
        else:
            return 0

    def get_valid_moves(self, player):
        moves = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            x, y = i, j
                            captured = []
                            while (0 <= x + di < self.size and 
                                0 <= y + dj < self.size and 
                                self.board[x + di][y + dj] == -player):
                                captured.append((x + di, y + dj))
                                x += di
                                y += dj
                            if (
                                0 <= x + di < self.size
                                and 0 <= y + dj < self.size
                                and self.board[x + di][y + dj] == player
                                and len(captured) > 0
                            ):
                                moves.add((i, j))
        return list(moves)

    def make_move(self, player, move):
        i, j = move
        self.board[i][j] = player
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                x, y = i, j
                captured = []
                while (
                    0 <= x + di < self.size
                    and 0 <= y + dj < self.size
                    and self.board[x + di][y + dj] == -player
                ):
                    captured.append((x + di, y + dj))
                    x += di
                    y += dj
                if (
                    0 <= x + di < self.size
                    and 0 <= y + dj < self.size
                    and self.board[x + di][y + dj] == player
                ):
                    for cx, cy in captured:
                        self.board[cx][cy] = player

    def get_cpu_move(self):
        moves = self.get_valid_moves(COMPUTER)
        if len(moves) == 0:
            return None
        return random.choice(moves)

    def get_human_move(self):
        value, move = self.minimax(self.minimax_depth, HUMAN)
        return move

    def minimax(
        self, depth: int, turn: int, alpha: float = -math.inf, beta: float = math.inf
    ) -> tuple[int, Move]:
        self.seen_nodes += 1
        if self.terminal_test():
            value = (
                self.WIN_HEURISTIC
                if self.get_winner() == HUMAN
                else -self.WIN_HEURISTIC
            )
            return value, None

        if depth <= 0:
            return self.heuristic(), None

        backup_board = [[x for x in row] for row in self.board]
        optimal_move = None

        if turn == HUMAN and len(self.get_valid_moves(turn)) == 0:
            turn *= -1

        if turn == HUMAN:
            node_value = -math.inf
            for move in self.get_valid_moves(HUMAN):
                self.make_move(HUMAN, move)
                value, successor_move = self.minimax(depth - 1, COMPUTER, alpha, beta)
                self.board = [[x for x in row] for row in backup_board]
                if value > node_value:
                    optimal_move = move
                    node_value = value
                    if self.prune and node_value >= beta:
                        break
                    alpha = max(alpha, value)

            return node_value, optimal_move

        elif turn == COMPUTER:
            node_value = math.inf
            for move in self.get_valid_moves(COMPUTER):
                self.make_move(COMPUTER, move)
                value, successor_move = self.minimax(depth - 1, HUMAN, alpha, beta)
                self.board = [[x for x in row] for row in backup_board]
                if value < node_value:
                    optimal_move = move
                    node_value = value
                    if self.prune and node_value <= alpha:
                        break
                    beta = min(beta, value)

            return node_value, optimal_move

    def heuristic(self) -> int:
        human_corners = self.count_corners(HUMAN)
        computer_corners = self.count_corners(COMPUTER)
        corners_coefficient = human_corners - computer_corners

        human_total = self.count_total(HUMAN)
        computer_total = self.count_total(COMPUTER)
        total_coefficient = human_total - computer_total

        return (
            self.CORNER_WEIGHT * corners_coefficient
            + self.TOTAL_WEIGHT * total_coefficient
        )
        #    self.BORDER_WEIGHT * self.count_empty() * (self.count_borders(HUMAN) - self.count_borders(COMPUTER)) + \

    def count_corners(self, player: int) -> int:
        sum = 0
        sum += self.board[0][0] == player
        sum += self.board[0][-1] == player
        sum += self.board[-1][0] == player
        sum += self.board[-1][-1] == player
        return sum

    def count_borders(self, player: int) -> int:
        sum = 0
        for i in range(self.size):
            sum += self.board[0][i] == player
            sum += self.board[-1][i] == player
            sum += self.board[i][0] == player
            sum += self.board[i][-1] == player
        return sum

    def count_total(self, player: int) -> int:
        return sum(row.count(player) for row in self.board)

    def terminal_test(self):
        return (
            len(self.get_valid_moves(HUMAN)) == 0
            and len(self.get_valid_moves(COMPUTER)) == 0
        )

    def play(self):
        winner = None
        while not self.terminal_test():
            if self.ui:
                self.ui.draw_board(self.board)
            if self.current_turn == HUMAN:
                move = self.get_human_move()
                if move:
                    self.make_move(self.current_turn, move)
            else:
                move = self.get_cpu_move()
                if move:
                    self.make_move(self.current_turn, move)
            self.current_turn = -self.current_turn
            if self.ui:
                self.ui.draw_board(self.board)
                time.sleep(1)

        winner = self.get_winner()
        print(self.seen_nodes)
        return winner

    def reset(self):
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.board[int(self.size / 2) - 1][int(self.size / 2) - 1] = self.board[
            int(self.size / 2)
        ][int(self.size / 2)] = 1
        self.board[int(self.size / 2) - 1][int(self.size / 2)] = self.board[
            int(self.size / 2)
        ][int(self.size / 2) - 1] = -1
        self.current_turn = random.choice([1, -1])
        self.seen_nodes = 0

    def test(
        self, depth: int, prune: bool = True, num_of_test: int = TOTAL_TESTS
    ) -> tuple[float, float, int]:
        ui = self.ui
        self.ui = None

        win = 0
        time_elapsed = 0
        seen_nodes = 0
        self.set_minimax_depth(depth)
        self.set_pruning(prune)

        for _ in range(num_of_test):
            start = time.time()
            win += self.play() == HUMAN
            time_elapsed += time.time() - start
            seen_nodes += self.seen_nodes
            self.reset()

        self.ui = ui

        return time_elapsed / num_of_test, win / num_of_test, seen_nodes / num_of_test


othello = Othello()
othello.play()
# othello.test(
#     depth=5,
# )
