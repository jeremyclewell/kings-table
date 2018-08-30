# -*- coding: utf-8 -*-
import numpy as np
import pudb
from easyAI import TwoPlayersGame, DictTT

black_pieces = [[0,3],[1,3],[3,0],[3,1],[3,5],[3,6],[5,3],[6,3]]
white_pieces = [[2,3],[3,2],[3,3],[3,4],[4,3]]
king = [3,3]
throne = np.array([[3,3]])
corners = np.array([[0,0],[0,6],[6,0],[6,6]])
pieces = [black_pieces, white_pieces + [king]]
BLACK = 1
WHITE = 2

class Game(TwoPlayersGame):
    """
    """

    def __init__(self, players, board_size = (7, 7)):
        self.players = players
        self.board_size = board_size
        self.board = np.zeros(board_size, dtype = int)
        for piece in black_pieces:
            self.board[piece[0]][piece[1]] = 1
        for piece in white_pieces:
            self.board[piece[0]][piece[1]] = 2
        self.king = np.array([5,5])
        self.nplayer = 1 # player 1 starts.

    def pieceIsKing(piece):
        return piece == king

    def validMoveFilter(boardSlice, piece):
        return corner in boardSlice

    def possible_moves_for_piece(self, piece):
        v_moves = []
        column = self.board[piece[0]]
        v_mask = np.ma.masked_where(column != 0, column)
        v_slices = np.ma.notmasked_contiguous(v_mask)
        v_slices = [slice for slice in v_slices if slice.stop == piece[1] or piece[1]+1 == slice.start]
        if len(v_slices) != 0:
            v_moves = range(np.amin(v_slices).start, np.amax(v_slices).stop)
            if piece[1] in v_moves:
                v_moves.remove(piece[1])
            v_moves = [[piece[0], val] for val in v_moves]

        h_moves = []
        row = self.board[:, piece[1]]
        h_mask = np.ma.masked_where(row != 0, row)
        h_slices = np.ma.notmasked_contiguous(h_mask)
        h_slices = [slice for slice in h_slices if slice.start == piece[0]+1 or piece[0]-1 == slice.stop]
        if len(h_slices) != 0:
            h_moves = range(np.amin(h_slices).start, np.amax(h_slices).stop)
            if piece[0] in h_moves:
                h_moves.remove(piece[0])
            h_moves = [[val, piece[1]] for val in h_moves]

        restricted_squares = throne
        if piece is not king:
            restricted_squares = np.concatenate((throne, corners), axis=0)

        moves = [(piece, move) for move in h_moves + v_moves if move not in restricted_squares]
        return moves

    def get_piece(self, coord):
        try:
            return self.board[coord[0]][coord[1]]
        except:
            return None

    def get_pieces(self, player):
        pieces = np.where(self.board == player)
        return np.dstack(pieces)[0]

    def capture(self, position):
        directions = [np.array([0,1]),np.array([0,-1]),np.array([1,0]),np.array([-1,0])]
        for direction in directions:
            target = direction + position
            if self.get_piece(target) == self.nopponent:
                if self.get_piece(direction + target) == self.nplayer or \
                        any(np.equal(corners, direction + target).all(1)) or \
                        any(np.equal(throne, direction + target).all(1)):
                    self.board[target[0]][target[1]] = 0

    def possible_moves(self):
        moves = []
        pieces = self.get_pieces(self.nplayer)
        if self.nmove % 3:
            pieces = pieces[::-1]
        for piece in pieces:
            moves.extend(self.possible_moves_for_piece(piece))
        if len(moves) == 0:
            pudb.set_trace()
        return moves

    def make_move(self, move):
        current_pos = move[0]
        next_pos = move[1]
        self.board[current_pos[0]][current_pos[1]] = 0
        self.board[next_pos[0]][next_pos[1]] = self.nplayer
        if (self.king == current_pos).all():
            self.king = next_pos
        self.capture(next_pos)

    def show(self):
        print('\n' + '\n'.join(['  1 2 3 4 5 6 7'] +
              ['ABCDEFG'[k] +
               ' ' + ' '.join([['∙', '⚫️', '⚪️', '👑'][self.board[k, i]]
               for i in range(self.board_size[0])])
               for k in range(self.board_size[1])] + ['']))

    def lose(self):
        if self.nplayer == BLACK:
            self.has_lost = (self.king == corners).any()
        else:
            self.has_lost = self.get_pieces(WHITE).size == 0

            # if not (self.king == self.get_pieces(WHITE)).any():
            #     return True
        return self.has_lost

    def scoring(self):
        if not self.has_lost:
            material = len(self.get_pieces(BLACK))/2./len(self.get_pieces(WHITE))
            # king_to_corner = min([np.linalg.norm(np.array(self.king)-corner) for corner in corners])
            # attackers_to_king = np.array([np.linalg.norm(np.array(self.king)-piece) for piece in self.get_pieces(BLACK)]).mean()
        # return king_to_corner + material**10 - attackers_to_king
        # return -attackers_to_king + king_to_corner
            return material
            # return material
        else:
            return -100

    def is_over(self):
        return self.lose()

    def ttentry(self):
        return "".join([".0X"[i] for i in self.board.flatten()])

if __name__ == "__main__":
    from easyAI import AI_Player, Negamax

    #ai_algo = Negamax(3, None , tt = DictTT())
    ai_algo = Negamax(5, None)
    game = Game([AI_Player(ai_algo), AI_Player(ai_algo)])
    game.play()
    print("player %d loses" % (game.nplayer))
