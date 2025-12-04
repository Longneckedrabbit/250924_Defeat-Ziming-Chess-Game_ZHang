"""
中国象棋规则和走法逻辑
"""

class ChessRules:
    def __init__(self):
        self.piece_moves = {
            'king': self.get_king_moves,
            'advisor': self.get_advisor_moves,
            'elephant': self.get_elephant_moves,
            'horse': self.get_horse_moves,
            'rook': self.get_rook_moves,
            'cannon': self.get_cannon_moves,
            'pawn': self.get_pawn_moves
        }

    def get_valid_moves(self, board, row, col):
        """获取棋子的所有合法移动"""
        piece = board[row][col]
        if not piece:
            return []

        piece_type = piece['type']
        color = piece['color']

        if piece_type in self.piece_moves:
            return self.piece_moves[piece_type](board, row, col, color)

        return []

    def get_king_moves(self, board, row, col, color):
        """获取帅/将的合法移动"""
        moves = []

        # 确定活动范围
        if color == 'red':
            min_row, max_row = 7, 9
        else:
            min_row, max_row = 0, 2

        min_col, max_col = 3, 5

        # 四个方向
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 检查是否在九宫格内
            if (min_row <= new_row <= max_row and
                min_col <= new_col <= max_col):

                # 检查目标位置
                if self.is_valid_move(board, row, col, new_row, new_col, color):
                    moves.append((new_row, new_col))

        # 检查将帅对面
        enemy_king_pos = self.find_enemy_king(board, color)
        if enemy_king_pos:
            enemy_row, enemy_col = enemy_king_pos
            # 如果在同一列且中间没有棋子
            if col == enemy_col and self.is_clear_path(board, row, enemy_row, col):
                # 可以直接吃掉对方的将/帅
                moves.append((enemy_row, enemy_col))

        return moves

    def get_advisor_moves(self, board, row, col, color):
        """获取士/仕的合法移动"""
        moves = []

        # 确定活动范围（九宫格）
        if color == 'red':
            min_row, max_row = 7, 9
        else:
            min_row, max_row = 0, 2

        min_col, max_col = 3, 5

        # 斜着走
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 检查是否在九宫格内
            if (min_row <= new_row <= max_row and
                min_col <= new_col <= max_col):

                if self.is_valid_move(board, row, col, new_row, new_col, color):
                    moves.append((new_row, new_col))

        return moves

    def get_elephant_moves(self, board, row, col, color):
        """获取象/相的合法移动"""
        moves = []

        # 确定活动范围（不能过河）
        if color == 'red':
            min_row, max_row = 5, 9
        else:
            min_row, max_row = 0, 4

        # 走田字
        directions = [(2, 2), (2, -2), (-2, 2), (-2, -2)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 检查是否在己方半场
            if min_row <= new_row <= max_row and 0 <= new_col <= 8:
                # 检查是否被塞象眼
                eye_row, eye_col = row + dr // 2, col + dc // 2
                if board[eye_row][eye_col] is None:
                    if self.is_valid_move(board, row, col, new_row, new_col, color):
                        moves.append((new_row, new_col))

        return moves

    def get_horse_moves(self, board, row, col, color):
        """获取马的合法移动"""
        moves = []

        # 马走日字
        directions = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]

        # 马腿位置
        leg_positions = [
            (1, 0), (1, 0), (-1, 0), (-1, 0),
            (0, 1), (0, -1), (0, 1), (0, -1)
        ]

        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row + dr, col + dc

            # 检查是否在棋盘内
            if 0 <= new_row <= 9 and 0 <= new_col <= 8:
                # 检查是否被绊马腿
                leg_row, leg_col = row + leg_positions[i][0], col + leg_positions[i][1]
                if board[leg_row][leg_col] is None:
                    if self.is_valid_move(board, row, col, new_row, new_col, color):
                        moves.append((new_row, new_col))

        return moves

    def get_rook_moves(self, board, row, col, color):
        """获取车的合法移动"""
        moves = []

        # 横向移动
        for dc in [-1, 1]:
            for i in range(1, 9):
                new_row, new_col = row, col + i * dc
                if 0 <= new_col <= 8:
                    if board[new_row][new_col] is None:
                        moves.append((new_row, new_col))
                    else:
                        if board[new_row][new_col]['color'] != color:
                            moves.append((new_row, new_col))
                        break
                else:
                    break

        # 纵向移动
        for dr in [-1, 1]:
            for i in range(1, 10):
                new_row, new_col = row + i * dr, col
                if 0 <= new_row <= 9:
                    if board[new_row][new_col] is None:
                        moves.append((new_row, new_col))
                    else:
                        if board[new_row][new_col]['color'] != color:
                            moves.append((new_row, new_col))
                        break
                else:
                    break

        return moves

    def get_cannon_moves(self, board, row, col, color):
        """获取炮的合法移动"""
        moves = []

        # 横向移动
        for dc in [-1, 1]:
            jumped = False
            for i in range(1, 9):
                new_row, new_col = row, col + i * dc
                if 0 <= new_col <= 8:
                    if board[new_row][new_col] is None:
                        if not jumped:
                            moves.append((new_row, new_col))
                    else:
                        if not jumped:
                            jumped = True
                        else:
                            if board[new_row][new_col]['color'] != color:
                                moves.append((new_row, new_col))
                            break
                else:
                    break

        # 纵向移动
        for dr in [-1, 1]:
            jumped = False
            for i in range(1, 10):
                new_row, new_col = row + i * dr, col
                if 0 <= new_row <= 9:
                    if board[new_row][new_col] is None:
                        if not jumped:
                            moves.append((new_row, new_col))
                    else:
                        if not jumped:
                            jumped = True
                        else:
                            if board[new_row][new_col]['color'] != color:
                                moves.append((new_row, new_col))
                            break
                else:
                    break

        return moves

    def get_pawn_moves(self, board, row, col, color):
        """获取兵/卒的合法移动"""
        moves = []

        if color == 'red':
            # 红兵向上走
            if row > 0:
                if self.is_valid_move(board, row, col, row - 1, col, color):
                    moves.append((row - 1, col))

            # 过河后可以横走
            if row <= 4:
                for dc in [-1, 1]:
                    new_row, new_col = row, col + dc
                    if 0 <= new_col <= 8:
                        if self.is_valid_move(board, row, col, new_row, new_col, color):
                            moves.append((new_row, new_col))
        else:
            # 黑卒向下走
            if row < 9:
                if self.is_valid_move(board, row, col, row + 1, col, color):
                    moves.append((row + 1, col))

            # 过河后可以横走
            if row >= 5:
                for dc in [-1, 1]:
                    new_row, new_col = row, col + dc
                    if 0 <= new_col <= 8:
                        if self.is_valid_move(board, row, col, new_row, new_col, color):
                            moves.append((new_row, new_col))

        return moves

    def is_valid_move(self, board, from_row, from_col, to_row, to_col, color):
        """检查移动是否合法"""
        target_piece = board[to_row][to_col]

        # 目标位置不能是己方棋子
        if target_piece and target_piece['color'] == color:
            return False

        return True

    def find_enemy_king(self, board, color):
        """找到敌方将/帅的位置"""
        enemy_color = 'black' if color == 'red' else 'red'

        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece and piece['type'] == 'king' and piece['color'] == enemy_color:
                    return (row, col)

        return None

    def is_clear_path(self, board, from_row, to_row, col):
        """检查路径是否清晰（无棋子阻挡）"""
        min_row, max_row = min(from_row, to_row), max(from_row, to_row)

        for row in range(min_row + 1, max_row):
            if board[row][col] is not None:
                return False

        return True

    def is_in_check(self, board, color):
        """检查是否被将军"""
        # 找到己方将/帅的位置
        king_pos = None
        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece and piece['type'] == 'king' and piece['color'] == color:
                    king_pos = (row, col)
                    break

        if not king_pos:
            return False

        king_row, king_col = king_pos

        # 检查是否有敌方棋子可以攻击将/帅
        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece and piece['color'] != color:
                    moves = self.get_valid_moves(board, row, col)
                    if (king_row, king_col) in moves:
                        return True

        return False

    def is_checkmate(self, board, color):
        """检查是否被将死"""
        if not self.is_in_check(board, color):
            return False

        # 检查是否有任何合法移动可以解除将军
        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece and piece['color'] == color:
                    moves = self.get_valid_moves(board, row, col)
                    for move_row, move_col in moves:
                        # 模拟移动
                        original_piece = board[move_row][move_col]
                        board[move_row][move_col] = piece
                        board[row][col] = None

                        # 检查是否还在被将军
                        still_in_check = self.is_in_check(board, color)

                        # 恢复棋盘
                        board[row][col] = piece
                        board[move_row][move_col] = original_piece

                        if not still_in_check:
                            return False

        return True

    def make_move(self, board, from_row, from_col, to_row, to_col):
        """执行移动并返回新的棋盘状态"""
        # 创建新棋盘
        new_board = [row[:] for row in board]

        # 执行移动
        piece = new_board[from_row][from_col]
        new_board[to_row][to_col] = piece
        new_board[from_row][from_col] = None

        return new_board

    def get_all_legal_moves(self, board, color):
        """获取指定颜色所有棋子的合法移动"""
        moves = []

        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece and piece['color'] == color:
                    piece_moves = self.get_valid_moves(board, row, col)
                    for move_row, move_col in piece_moves:
                        moves.append({
                            'from': (row, col),
                            'to': (move_row, move_col),
                            'piece': piece
                        })

        return moves

    def evaluate_board(self, board):
        """简单评估棋盘局面"""
        piece_values = {
            'king': 10000,
            'advisor': 20,
            'elephant': 20,
            'horse': 40,
            'rook': 90,
            'cannon': 45,
            'pawn': 10
        }

        red_score = 0
        black_score = 0

        for row in range(10):
            for col in range(9):
                piece = board[row][col]
                if piece:
                    value = piece_values.get(piece['type'], 0)
                    if piece['color'] == 'red':
                        red_score += value
                    else:
                        black_score += value

        return red_score - black_score