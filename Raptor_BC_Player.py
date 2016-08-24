""" Marcus Pham, Michael Omori
Raptor_BC_Player.py
Baroque Chess
April 30, 2016
"""
from copy import deepcopy
import heapq
from random import choice

BLACK = 0
WHITE = 1

INIT_TO_CODE = {'p': 2, 'P': 3, 'c': 4, 'C': 5, 'l': 6, 'L': 7, 'i': 8, 'I': 9,
                'w': 10, 'W': 11, 'k': 12, 'K': 13, 'f': 14, 'F': 15, '-': 0}

CODE_TO_INIT = {0: '-', 2: 'p', 3: 'P', 4: 'c', 5: 'C', 6: 'l', 7: 'L', 8: 'i', 9: 'I',
                10: 'w', 11: 'W', 12: 'k', 13: 'K', 14: 'f', 15: 'F'}

PIECE_VALUES = {'p': 1000, 'P': 1000, 'c': 2900, 'C': 2900, 'l': 4300, 'L': 4300, 'i': 5300, 'I': 5300,
                'w': 3100, 'W': 3100, 'k': 300000, 'K': 300000, 'f': 8200, 'F': 8200, '-': 0}


def introduce():
    return '''
My name is RAPTOR. I was designed by 
Marcus Pham (mtpham13) and Michael Omori (omorim).
I can smell fear, and I eat Kings for lunch.
'''


def nickname():
    return 'RAPTOR'


def who(piece): return piece % 2


def prepare(player2Nickname):
    pass


def parse(bs):  # bs is board string
    """Translate a board string into the list of lists representation."""
    b = [[0, 0, 0, 0, 0, 0, 0, 0] for r in range(8)]
    rs9 = bs.split("\n")
    rs8 = rs9[1:]  # eliminate the empty first item.
    for iy in range(8):
        rss = rs8[iy].split(' ')
        for jx in range(8):
            b[iy][jx] = INIT_TO_CODE[rss[jx]]
    return b


INITIAL = parse('''
c l i w k i l f
p p p p p p p p
- - - - - - - -
- - - - - - - -
- - - - - - - -
- - - - - - - -
P P P P P P P P
F L I W K I L C
''')


def random_move(state):
    #print('legalMoves')
    #print(legalMoves(state))
    move = []
    while move == []:
        move = choice(legalMoves(state))
    return move


def makeMove(current_state, current_remark, time_limit=1):
    # new_state = changeState(current_state, chooseMove(current_state))
    #move = random_move(current_state)
    move = chooseMove(current_state)
    #print(move)
    new_state = changeState(current_state, move)
    new_remark = "Your move."
    #print(move)
    #print(new_state)
    return [[move, new_state], new_remark]


def chooseMove(state):
    """Evaluates and chooses to play the best move found in the given time."""
    'Returns the state resulting from playing that move.'
    # Iterative Deepening Search
    # + alpha-beta pruning
    for depth in range(20):
        found = mini_max(state, depth, 1)
        if found != None:
            return found


# each tree node has an interval (a, b) in which solution must lie
# prune off any node, if a >= b
# copy the value when going down
# fill in beta value if on min, alpha value if on max
# alpha is the lower bound, beta is the upper bound
# flip alpha and beta when going up
# perform static evaluation of every successor and order them best-first before doing recursive call


def mini_max(state, depth, time):
    ordered_states = []
    if depth == 0 or time == 1:
        return staticEval(state)
    for child in neighbors(state):
        heapq.heappush(ordered_states, child)  # order states
    if state.whose_move == 1:
        v = -100000
        for child in ordered_states:
            b = mini_max(child, depth - 1, 1)
            if b > v:
                v = b
        return v
    if state.whose_move == 0:
        v = 1000000
        for child in ordered_states:
            b = mini_max(child, depth - 1, 1)
            if b < v:
                v = b
        return v


def legalMoves(state):
    # Returns a list of all legal moves for a given state.
    moveList = []
    for r in range(0, 8):
        for c in range(0, 8):
            piece = CODE_TO_INIT[state.board[r][c]]
            if piece != '-':
                is_white_piece = piece.isupper()
                if (state.whose_move and is_white_piece) or \
                        (not state.whose_move and not is_white_piece):
                    frozen = freezeCheck(state.board, r, c, state.whose_move)
                    if not frozen:
                        getMoves(state.board, piece, r, c, moveList)
    return moveList


# Takes the current state and a legal move and returns an updated
# state after making the given move.
# A move is formatted like: "p34-64" or "I55-54"
def changeState(state, move):
    new_state = deepcopy(state)
    piece = move[0]
    old_r = int(move[1])
    old_c = int(move[2])
    new_r = int(move[4])
    new_c = int(move[5])
    # Make the move
    #print(old_r)
    #print(old_c)
    new_state.board[old_r][old_c] = 0
    new_state.board[new_r][new_c] = INIT_TO_CODE[piece]
    # Handle special capture cases
    handleSpecialCaptures(new_state, piece, old_r, old_c, new_r, new_c)
    new_state.whose_move ^= 1
    # print(new_state)
    return new_state


# Handles special captures (Captures not automatically handled by moving piece to destination)
# Returns whether the given move results in a capture from the given state.
def handleSpecialCaptures(state, piece, old_r, old_c, new_r, new_c):
    board = state.board
    if piece.lower() == 'p':
        return pincherCapture(board, new_r, new_c)
    elif piece.lower() == 'i':
        captured = pincherCapture(board, new_r, new_c, True) or \
                   withdrawerCapture(board, old_r, old_c, new_r, new_c, True) or \
                   coordinatorCapture(board, new_r, new_c, state.whose_move, True)
        return captured
    elif piece.lower() == 'w':
        return withdrawerCapture(board, old_r, old_c, new_r, new_c)
    elif piece.lower() == 'c':
        return coordinatorCapture(board, new_r, new_c, state.whose_move)
    else:
        return False


def pincherCapture(board, r, c, is_imitator=False):
    # check adjacent squares, check if enemy, and then check one over in same direction is friendly piece
    # r = 1, c = 1
    #adj_r = 0, adj_c = 0
    #adj_r = 2, adj_c = 2
    for i in range(-1, 2, 2):
        adj_r = r + i
        adj_c = c + i
        if 0 <= adj_r <= 7 and 0 <= adj_c <= 7:
            if isOppositePiece(board, r, c, r, adj_c):
                if board[r][adj_c + i] % 2 == board[r][c]:
                    if is_imitator:
                        imitator = CODE_TO_INIT[board[r][c]]
                        return adjacentCapture(board, r, adj_c, r, c, imitator)
                    return adjacentCapture(board, r, adj_c, r, c)
            if isOppositePiece(board, r, c, adj_r, c):
                if board[r][adj_c + i] % 2 == board[r][c]:
                    if is_imitator:
                        imitator = CODE_TO_INIT[board[r][c]]
                        return adjacentCapture(board, adj_r, c, r, c, imitator)
                    return adjacentCapture(board, adj_r, c, r, c)
    return False


def withdrawerCapture(board, r1, c1, r2, c2, is_imitator=False):
    """ Takes a board, and two sets of coordinates. Can also handle
    Imitators posing as Withdrawers.
    Captures (removes) the piece 1 square adjacent to (r1,c1) in the
    opposite direction of (r2,c2) if it is an opposite-colored piece.
    Returns True if a capture was made, False otherwise."""
    if is_imitator:
        imitator = CODE_TO_INIT[[board[r1][c1]]]
        return adjacentCapture(board, r1, c1, r2, c2, imitator)
    return adjacentCapture(board, r1, c1, r2, c2)


def coordinatorCapture(board, r, c, whose_move, is_imitator=False):
    king_r = -1
    king_c = -1
    for i in range(8):
        for j in range(8):
            if board[i][j] == (12 + whose_move):  # Found the King
                king_r = i
                king_c = j
                break
                # Check for breaking out of both loops
    can_capture1 = isOppositePiece(board, r, c, king_r, c)
    can_capture2 = isOppositePiece(board, r, c, r, king_c)
    if is_imitator:
        can_capture1 = can_capture1 and (board[king_r][c] == 5 or board[king_r][c] == 6)
        can_capture2 = can_capture2 and (board[r][king_c] == 5 or board[r][king_c] == 6)
    if can_capture1:
        board[king_r][c] = 0
    if can_capture2:
        board[r][king_c] = 0
    return can_capture1 or can_capture2


# Takes a board, and two sets of coordinates.
# Captures (removes) the piece 1 square adjacent to (r1,c1) in the
# opposite direction of (r2,c2) if it is an opposite-colored piece.
# Returns True if a capture was made, False otherwise.
def adjacentCapture(board, r1, c1, r2, c2, imitator_piece=None):
    left = -1 if (r2 - r1 < 0) else 1
    down = -1 if (c2 - c1 < 0) else 1
    cap_r = r1 - left
    cap_c = c1 - down
    if isOppositePiece(board, r1, c1, r2, c2):
        if imitator_piece and CODE_TO_INIT[board[cap_r][cap_c]] != imitator_piece:
            return False
        board[cap_r][cap_c] = 0
        return True
    return False


# Returns whether two coordinates house opposite-colored pieces
def isOppositePiece(board, r1, c1, r2, c2):
    return board[r1][c1] != 0 and board[r2][c2] != 0 and board[r1][c1] % 2 != board[r2][c2] % 2


# Takes a board, piece, and piece coordinates
# Returns a list of all legal moves for that piece, including captures.
def getMoves(board, piece, r, c, moveList=[]):
    # capture moves included
    if piece.lower() == 'p':
        # print('pawn moves')
        getRookMoves(piece, board, r, c, moveList)
    # Freezer cannot capture
    elif piece.lower() == 'f':
        getQueenMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'l':
        # print("leaper moves")
        getQueenMoves(piece, board, r, c, moveList)
    elif piece.lower() == 'i':
        getQueenMoves(piece, board, r, c, moveList)
        # Can capture adjacent enemy King
        for i in range(-1, 2):
            if 0 <= r + i < 8 and 0 <= c + i < 8:
                adj_king = board[r + i][c + i] == 12 or board[r + i][c + i] == 13
                if isOppositePiece(board, r, c, r + i, c + i) and adj_king:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r + i) + str(c + i))
                    # capture moves included
    elif piece.lower() == 'w':
        # print('width')
        getQueenMoves(piece, board, r, c, moveList)
    # capture moves included
    elif piece.lower() == 'k':
        # print('killer king')
        getKingMoves(board, r, c, moveList)
    # capture moves included
    elif piece.lower() == 'c':
        # print('coordinator')
        getQueenMoves(piece, board, r, c, moveList)
    # print(moveList)
    return moveList


"""
# Takes current board, position of leaper (r,c), a horizontal and
# vertical direction, and move list
# Adds the leaper moves in the given direction to the move list.
def leaperHelp(board, r, c, horizontal, vertical, moveList, is_imitator=False):
    for i in range(1, 8):  # 7x
        new_row = r + vertical * i
        new_file = c + horizontal * i
        if 0 <= new_file < 8 and 0 <= new_row < 8:
            if board[new_row][new_file] == 0:  # movements
                moveList.append(CODE_TO_INIT[board[r][c]] + str(r) + str(c) + '-' + str(new_row) + str(new_file))
            # checking captures
            elif 1 < r < 6 and 1 < c < 6 and board[r][c] % 2 != board[file_move][rank_move] % 2 and \
                    board[r + horizontal * (i + 1)][c + vertical * (i + 1)] == 0:
                if is_imitator:
                    can_capture_leaper = board[file_move][rank_move] == 6 or board[file_move][rank_move] == 7
                    if can_capture_leaper:
                        moveList.append(
                            CODE_TO_INIT[board[r][c]] + str(r) + str(c) + '-' + str(r + horizontal * (i + 1)) + str(
                                c + vertical * (i + 1)))
                else:
                    moveList.append(
                        CODE_TO_INIT[board[r][c]] + str(r) + str(c) + '-' + str(r + horizontal * (i + 1)) + str(
                            c + vertical * (i + 1)))
"""


def getKingMoves(board, r, c, moveList):
    # Takes board, piece coordinates, and movelist
    # Appends available king moves to the list of moves.
    for i in range(3):
        for j in range(3):
            rank = r - 1 + i
            file = c - 1 + j
            if 0 <= rank <= 7 and 0 <= file <= 7 \
                    and board[r][c] % 2 != board[rank][file] % 2:  # King cannot move into friendlies
                moveList.append(CODE_TO_INIT[board[r][c]] + str(r) + str(c) + '-' + str(rank) + str(file))


# Takes board, piece coordinates, and movelist
# Appends available Rook-like moves at the given coordinates to the list of moves.
def getRookMoves(piece, board, r, c, moveList):
    # r = 6
    # c = 0
    up = True
    down = True
    right = True
    left = True
    for i in range(1, 8):
        # if we hit a piece, stop or if we hit the edge
        if r + i <= 7 and up:  # going up
            if board[r + i][c] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r + i) + str(c))
            elif piece.lower() == 'l' and board[r+i+1][c] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r+i][c]].lower() == 'l' and board[r+i+1][c] == 0:
                if r + i <= 6:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r + i + 1) + str(c))
            else:
                up = False
        if r - i >= 0 and down:  # going down
            if board[r - i][c] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r - i) + str(c))
            elif piece.lower() == 'l' and board[r - i - 1][c] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r-i][c]].lower() == 'l' and board[r - i - 1][
                        c] == 0:
                if r - i >= -1:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r - i - 1) + str(c))
            else:
                down = False
        if c + i <= 7 and right:  # going right
            if board[r][c + i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r) + str(c + i))
            elif piece.lower() == 'l' and board[r][c + i + 1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r][c+i]].lower() == 'l' and board[r][c+i+1] == 0:
                if c + i <= 6:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r) + str(c+i+1))
            else:
                right = False
        if c - i >= 0 and left:  # going left
            if board[r][c - i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r) + str(c - i))
            elif piece.lower() == 'l' and board[r][c-i-1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r][c-i]].lower() == 'l' and board[r][c-i-1] == 0:
                if c - i >= -1:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r) + str(c-i-1))
            else:
                left = False


def getBishopMoves(piece, board, r, c, moveList):
    # Takes board, piece coordinates, and move list
    # Appends available Bishop-like moves at the given coordinates to the list of moves
    ne = True
    nw = True
    se = True
    sw = True
    for i in range(1, 8):
        # if we hit a piece, stop or if we hit the edge
        if r + i <= 7 and c + i <= 7 and ne:  # going up and right
            if board[r + i][c + i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r + i) + str(c + i))
            elif piece.lower() == 'l' and board[r+i+1][c + i + 1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r + i][c+i]].lower() == 'l' and board[r + i + 1][c+i+1] == 0:
                if r + i <= 6 and c + i <= 6:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r+i+1) + str(c + i + 1))
            else:
                ne = False
        if r - i >= 0 and c - i >= 0 and sw:  # going down and left
            if board[r - i][c - i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r - i) + str(c - i))
            elif piece.lower() == 'l' and board[r-i-1][c - i - 1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r - i][c-i]].lower() == 'l' and \
                    board[r-i-1][c-i-1] == 0:
                if r - i >= -1 and c - i >= -1:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r-i-1) + str(c - i - 1))
            else:
                sw = False
        if c + i <= 7 and r - i >= 0 and se:  # going right and down
            if board[r - i][c + i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r - i) + str(c + i))
            elif piece.lower() == 'l' and board[r-i-1][c + i + 1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r - i][c+i]].lower() == 'l' and board[r-i-1][
                        c+i+1] == 0:
                if r - i >= -1 and c + i <= 6:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r-i-1) + str(c + i + 1))
            else:
                se = False
        if c - i >= 0 and r + i <= 7 and nw:  # going left   and up
            if board[r + i][c - i] == 0:
                moveList.append(piece + str(r) + str(c) + '-' + str(r + i) + str(c - i))
            elif piece.lower() == 'l' and board[r+i+1][c - i - 1] == 0 \
                    or piece.lower() == 'i' and CODE_TO_INIT[board[r + i][c-i]].lower() == 'l' and board[r + i + 1][
                        c-i-1] == 0:
                if r + i <= 6 and c - i >= -1:
                    moveList.append(piece + str(r) + str(c) + '-' + str(r+i+1) + str(c - i - 1))
            else:
                nw = False


# Takes board, piece coordinates, and move list
# Appends available Queen-like moves at the given coordinate to the list of moves
def getQueenMoves(piece, board, r, c, moveList):
    getRookMoves(piece, board, r, c, moveList)
    getBishopMoves(piece, board, r, c, moveList)


# Takes board, piece coordinates, and boolean white-to-move
# Returns whether the piece at the given coordinates is frozen.
def freezeCheck(board, r, c, white_to_move):
    opp_freezer = 14 if white_to_move else 15  # 'F' or 'f'
    is_freezer = (white_to_move and board[r][c] == 14) or \
                 (not white_to_move and board[r][c] == 15)
    opp_imitator = 9 if white_to_move else 8  # 'I' or 'i'
    for i in range(3):
        for j in range(3):
            rank = r - 1 + i
            file = c - 1 + j
            if 0 <= rank <= 7 and 0 <= file <= 7:
                if board[rank][file] == opp_freezer or (is_freezer and board[rank][file] == opp_imitator):
                    return True
    return False


# Returns a list of all neighboring states for a given state.


def neighbors(state):
    result = []
    copy_state = deepcopy(state)
    for move in legalMoves(state):
        result.append(changeState(copy_state, move))
    return result


def freeze_penalty(state, freezer_color, freezer_locationx, freezer_locationy):
    # accounted in piece activity
    # Freezer: Immobilized pieces lose fourth of their value, kings get penalty of -2500
    # checks all adjacent squares
    bonus = 0
    for i in range(freezer_locationx - 1, freezer_locationx + 2):  # horizontal
        if i in range(0, 8) and i != freezer_locationx:  # checking if legal square and doesn't match freezer location
            for j in range(freezer_locationy - 1, freezer_locationy + 2):  # vertical
                if j in range(0, 8) and j != freezer_locationy:  # checking if legal square
                    if state.board[i][j] != 0:
                        if state.board[i][j] % freezer_color != 0:
                            if state.board[i][j] == 12 or state.board[i][j] == 13:
                                bonus -= 2500
                            else:
                                bonus -= PIECE_VALUES[CODE_TO_INIT[state.board[i][j]]] / 4
    if freezer_color == 0:  # if black piece
        return -1 * bonus
    else:
        return bonus


def coordinator_bonus(state, x, y):
    # 1000 / 14 for every enemy piece on line with king
    bonus = 0
    for i in range(0, 8):
        if y != i:
            if isOppositePiece(state.board, x, y, x, i):
                bonus += 1000 / 14
        if x != i:
            if isOppositePiece(state.board, x, y, i, y):
                bonus += 1000 / 14
    return bonus


def w_bonus(state, w_color, w_locationx, w_locationy):
    # Withdrawer: Gains bonus proportional to values of pieces next to it and if it can withdraw
    # max bonus of 1200
    bonus = 0
    for i in range(w_locationx - 1, w_locationx + 2):  # horizontal
        if i in range(0, 8) and i != w_locationx:  # checking if legal square and not square piece is on
            for j in range(w_locationy - 1, w_locationy + 2):  # vertical
                if j in range(0, 8) and j != w_locationy:  # checking if legal square
                    if state.board[i][j] != 0:  # if adjacent piece
                        new_state = deepcopy(state)
                        if withdrawerCapture(new_state.board, w_locationx, w_locationy, i, j):
                            bonus += PIECE_VALUES[state.board(i, j)] / 20
    if w_color == 0:
        return -1 * bonus
    else:
        return bonus


# checks one piece at r, c
def pinch(state, r, c):
    bonus = 0
    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            if 0 <= r + i < 8 and 0 <= r + j < 8:
                if state.board[r + i][r + j] != 0:  # if not empty square
                    if isOppositePiece(state.board, r, c, r + i, r + j):
                        bonus += (PIECE_VALUES[CODE_TO_INIT[state.board[r + i][r + j]]] -
                                  PIECE_VALUES[CODE_TO_INIT[state.board[r][c]]]) / 70
    return bonus


def leaper_bonus(state, color, leaper_x, leaper_y):
    # what are specific rules for leaper?
    # Leaper: bonus if there is an empty square to leap to and capture them
    # max is 1000
    largest_piece_value = 0
    for move in getMoves(state.board, 'l', leaper_x, leaper_y, ):
        if move[1] == leaper_x:  # moving vertically
            if move[2] > leaper_y:  # jumping up or right
                if PIECE_VALUES[state.board[move[1], move[2] - 1]] > largest_piece_value:
                    largest_piece_value = PIECE_VALUES[state.board[move[1], move[2] - 1]]
            else:  # jumping down or left
                if PIECE_VALUES[state.board[move[1], move[2] + 1]] > largest_piece_value:
                    largest_piece_value = PIECE_VALUES[state.board[move[1], move[2] + 1]]
    if color == 0:
        return -1 * largest_piece_value / 8
    else:
        return largest_piece_value


def king_safety(state, x_location, y_location):
    # King: loses value when no pieces are around it, more pieces around it the better
    # ranges from -1000 to 1000
    ks = 0
    # white king safety
    for i in range(round(x_location / 3), round(x_location / 3) + 3):  # horizontal
        for j in range(round(y_location / 3), round(y_location / 3) + 3):  # vertical
            # if square is adjacent
            if (y_location - j) + (x_location - i) == 1 or (y_location - j) + (x_location - i) == 2:
                if state.board[i][j] != 0:
                    if state.board[i][j] % 2 == 0:  # black piece
                        ks -= 500
                    else:
                        ks += 500
        return ks


def staticEval(state):
    # even numbers map to black pieces
    material_diff = 0
    king_eval = 0
    f_penalty = 0
    queen_bonus = 0
    l_bonus = 0
    pa = 0
    pincher_bonus = 0
    for r in range(8):
        for c in range(8):
            piece = state.board[r][c]
            if piece == 12:
                king_eval -= king_safety(state, r, c)
            if piece == 13:
                king_eval += king_safety(state, r, c)
            if piece != 0:  # if it's not an empty square
                # pincher bonus
                pincher_bonus += pinch(state, r, c)

                # checking number of squares each piece can move to, l is index
                # max of 2000
                copy_state = deepcopy(state)
                num_moves = len(legalMoves(copy_state))
                copy_state.whose_move ^= 1
                num_moves2 = len(legalMoves(copy_state))
                if state.whose_move:
                    pa = 40 * (num_moves - num_moves2)
                else:
                    pa = 40 * (num_moves2 - num_moves)
                material_diff += PIECE_VALUES[CODE_TO_INIT[piece]]
                if piece == 14 or piece == 15:
                    f_penalty += freeze_penalty(state, piece, r, c)
                    # check freezer bonus
                elif piece == 10 or piece == 11:
                    queen_bonus += w_bonus(state, piece, r, c)
                    # check with-drawer bonus
                elif piece == 6 or piece == 7:
                    l_bonus += leaper_bonus(state, piece, r, c)
                    # check leaper bonus
    return material_diff
            #+ king_eval + f_penalty + queen_bonus + l_bonus + pa + pincher_bonus


class BC_state:
    def __init__(self, old_board=INITIAL, whose_move=WHITE):
        new_board = [r[:] for r in old_board]
        self.board = new_board
        self.whose_move = whose_move
        self.state_eval = staticEval(self)

    def __repr__(self):
        s = ''
        for r in range(8):
            for c in range(8):
                s += CODE_TO_INIT[self.board[r][c]] + " "
            s += "\n"
        if self.whose_move == WHITE:
            s += "WHITE's move"
        else:
            s += "BLACK's move"
        s += "\n"
        return s

    def __lt__(self, other):
        return self.state_eval < other.state_eval


def test_starting_board():
    init_state = BC_state(INITIAL, WHITE)
    print(init_state)

# test_starting_board()
