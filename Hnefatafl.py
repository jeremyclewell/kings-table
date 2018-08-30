# -*- coding: utf-8 -*-
import numpy as np
import pudb
from easyAI import TwoPlayersGame, DictTT

black_pieces = [[0,3],[0,4],[0,5],[0,6],[0,7],[1,5],[3,0],[3,10],[4,0],[4,10],[5,0],
                [5,1],[5,9],[5,10],[6,0],[6,10],[7,0],[7,10],[9,5],[10,3],[10,4],
                [10,5],[10,6],[10,7]]
white_pieces = [[3,5],[4,4],[4,5],[4,6],[5,3],[5,4],[5,5],[5,6],[5,7],[6,4],[6,5],[6,6],[7,5]]
# black_pieces_2 = [[5,1],[5,9],[5,10],[6,0],[6,10],[7,0],[7,10]]
# white_pieces_2 = [[5,5],[6,5],[6,6],[7,5]]
black_pieces_2 = [[5,3],[5,2],[5,7],[5,9],[2,3],[2,2],[2,7],[2,9]]
white_pieces_2 = [[5,5],[6,5],[6,6],[7,5]]
king = [5,5]
throne = np.array([[5,5]])
corners = np.array([[0,0],[0,10],[10,0],[10,10]])
pieces = [black_pieces, white_pieces + [king]]
BLACK = 1
WHITE = 2

class Game(TwoPlayersGame):
    """
    """

    def __init__(self, players, board_size = (11, 11)):
        self.players = players
        self.board_size = board_size
        self.board = np.zeros(board_size, dtype = int)
        for piece in black_pieces_2:
            self.board[piece[0]][piece[1]] = 1
        for piece in white_pieces_2:
            self.board[piece[0]][piece[1]] = 2
        self.king = np.array([5,5])
        self.nplayer = 1 # player 1 starts.

    def possible_moves_for_piece(self, piece):
        v_moves = []
        # pudb.set_trace()
        v_mask = np.ma.masked_where(self.board[:, piece[1]] != 0, self.board[:, piece[1]])
        v_slices = np.ma.notmasked_contiguous(v_mask)
        try:
            v_slice = [slice for slice in v_slices if slice.start <= piece[0]+1 and piece[0]-1 <= slice.stop][0]
        except TypeError:
            v_slice = v_slices
        except IndexError:
            v_slice = None
        if v_slice is not None:
            v_moves = range(v_slice.start, v_slice.stop)
            if piece[0] in v_moves:
                v_moves.remove(piece[0])
            v_moves = [[val, piece[1]] for val in v_moves]

        h_moves = []
        h_mask = np.ma.masked_where(self.board[piece[0]] != 0, self.board[piece[0]])
        h_slices = np.ma.notmasked_contiguous(h_mask)
        try:
            h_slice = [slice for slice in h_slices if slice.start <= piece[1]+1 and piece[1]-1 <= slice.stop][0]
        except TypeError:
            h_slice = h_slices
        except IndexError:
            h_slice = None
        if h_slice is not None:
            h_moves = range(h_slice.start, h_slice.stop)
            if piece[1] in h_moves:
                h_moves.remove(piece[1])
            h_moves = [[piece[0], val] for val in h_moves]
        return [(piece, move) for move in h_moves + v_moves if move not in corners + throne]

    def get_piece(self, coord):
        try:
            return self.board[coord[0]][coord[1]]
        except:
            return None

    def get_pieces(self, player):
        pieces = np.where(self.board == player)
        return np.dstack(pieces)[0]

    def capture(self, move):
        directions = [np.array([0,1]),np.array([0,-1]),np.array([1,0]),np.array([-1,0])]
        for direction in directions:
            target = direction + move
            if self.get_piece(target) == self.nopponent:
                if self.get_piece(direction + target) == self.nplayer or \
                        (direction + target) in corners or \
                        (direction + target) in throne:
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
        print('\n' + '\n'.join(['  1 2 3 4 5 6 7 8 9 10 11'] +
              ['ABCDEFGHIJK'[k] +
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
            return -(material**10)
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
