from flask import Flask, render_template, request, jsonify
import json
import logging
from datetime import datetime
import os
import numpy as np
from collections import defaultdict
from chess_rules import ChessRules
from game_logger import GameLogger
from deep_learning_model import PatternRecognitionModel, ChessPositionEncoder
from progress_tracker import ProgressTracker

app = Flask(__name__)

# 基础配置
app.config['SECRET_KEY'] = 'chinese-chess-ai-learning'
app.config['LOG_DIR'] = 'logs'
app.config['DATA_DIR'] = 'data'

# 创建必要的目录
os.makedirs(app.config['LOG_DIR'], exist_ok=True)
os.makedirs(app.config['DATA_DIR'], exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(app.config['LOG_DIR'], 'game.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 象棋棋盘状态类
class ChessBoard:
    def __init__(self):
        self.rules = ChessRules()
        self.board = self.initialize_board()
        self.current_player = 'red'  # 红方先行
        self.move_history = []
        self.game_state = 'playing'

    def initialize_board(self):
        """初始化棋盘"""
        board = [[None for _ in range(9)] for _ in range(10)]

        # 红方棋子（下方）
        # 车
        board[9][0] = {'type': 'rook', 'color': 'red'}
        board[9][8] = {'type': 'rook', 'color': 'red'}
        # 马
        board[9][1] = {'type': 'horse', 'color': 'red'}
        board[9][7] = {'type': 'horse', 'color': 'red'}
        # 相
        board[9][2] = {'type': 'elephant', 'color': 'red'}
        board[9][6] = {'type': 'elephant', 'color': 'red'}
        # 士
        board[9][3] = {'type': 'advisor', 'color': 'red'}
        board[9][5] = {'type': 'advisor', 'color': 'red'}
        # 帅
        board[9][4] = {'type': 'king', 'color': 'red'}
        # 炮
        board[7][1] = {'type': 'cannon', 'color': 'red'}
        board[7][7] = {'type': 'cannon', 'color': 'red'}
        # 兵
        for i in range(0, 9, 2):
            board[6][i] = {'type': 'pawn', 'color': 'red'}

        # 黑方棋子（上方）
        # 车
        board[0][0] = {'type': 'rook', 'color': 'black'}
        board[0][8] = {'type': 'rook', 'color': 'black'}
        # 马
        board[0][1] = {'type': 'horse', 'color': 'black'}
        board[0][7] = {'type': 'horse', 'color': 'black'}
        # 象
        board[0][2] = {'type': 'elephant', 'color': 'black'}
        board[0][6] = {'type': 'elephant', 'color': 'black'}
        # 士
        board[0][3] = {'type': 'advisor', 'color': 'black'}
        board[0][5] = {'type': 'advisor', 'color': 'black'}
        # 将
        board[0][4] = {'type': 'king', 'color': 'black'}
        # 炮
        board[2][1] = {'type': 'cannon', 'color': 'black'}
        board[2][7] = {'type': 'cannon', 'color': 'black'}
        # 卒
        for i in range(0, 9, 2):
            board[3][i] = {'type': 'pawn', 'color': 'black'}

        return board

    def get_board_state(self):
        """获取棋盘状态"""
        return {
            'board': self.board,
            'current_player': self.current_player,
            'game_state': self.game_state,
            'move_history': self.move_history
        }

    def make_move(self, from_row, from_col, to_row, to_col):
        """执行移动"""
        if self.is_valid_move(from_row, from_col, to_row, to_col):
            # 记录移动
            move_data = {
                'from': (from_row, from_col),
                'to': (to_row, to_col),
                'piece': self.board[from_row][from_col],
                'captured': self.board[to_row][to_col],
                'player': self.current_player
            }
            self.move_history.append(move_data)

            # 执行移动
            self.board[to_row][to_col] = self.board[from_row][from_col]
            self.board[from_row][from_col] = None

            # 切换玩家
            self.current_player = 'black' if self.current_player == 'red' else 'red'

            # 检查游戏状态
            self.check_game_state()

            return True
        return False

    def is_valid_move(self, from_row, from_col, to_row, to_col):
        """检查移动是否合法"""
        piece = self.board[from_row][from_col]
        if not piece:
            return False

        if piece['color'] != self.current_player:
            return False

        valid_moves = self.rules.get_valid_moves(self.board, from_row, from_col)
        return (to_row, to_col) in valid_moves

    def check_game_state(self):
        """检查游戏状态"""
        # 检查是否被将死
        if self.rules.is_checkmate(self.board, 'red'):
            self.game_state = 'black_wins'
        elif self.rules.is_checkmate(self.board, 'black'):
            self.game_state = 'red_wins'
        elif self.is_draw():
            self.game_state = 'draw'

    def is_draw(self):
        """检查是否平局"""
        # 简单的平局判断：双方只剩下将帅
        pieces = []
        for row in self.board:
            for piece in row:
                if piece:
                    pieces.append(piece)

        if len(pieces) == 2 and all(p['type'] == 'king' for p in pieces):
            return True
        return False

    def get_all_legal_moves(self, color):
        """获取指定颜色所有合法移动"""
        return self.rules.get_all_legal_moves(self.board, color)

# 基础AI类（小白水平）
class BasicAI:
    def __init__(self):
        self.player_name = "张子鸣"
        self.learning_data = defaultdict(list)
        self.game_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.rules = ChessRules()
        self.difficulty = 'beginner'  # AI难度等级

    def get_move(self, game_board):
        """获取AI的移动（结合学习系统）"""
        legal_moves = self.get_legal_moves(game_board)
        if not legal_moves:
            return None

        import random
        # 根据难度选择移动策略
        if self.difficulty == 'beginner':
            # 小白水平：完全随机
            return random.choice(legal_moves)
        elif self.difficulty == 'novice':
            # 初学者：优先吃子
            capture_moves = [m for m in legal_moves if m['captured']]
            if capture_moves and random.random() < 0.7:
                return random.choice(capture_moves)
            return random.choice(legal_moves)
        else:
            # 使用学习系统提供的建议
            learning_suggestion = learning_system.get_ai_move_suggestion(
                game_board.board, legal_moves
            )
            if learning_suggestion:
                return learning_suggestion
            else:
                # 回退到简单评估
                return self.get_best_move(game_board, legal_moves)

    def get_legal_moves(self, game_board):
        """获取所有合法移动"""
        return game_board.get_all_legal_moves('black')

    def get_best_move(self, game_board, legal_moves):
        """使用简单评估选择最佳移动"""
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            score = self.evaluate_move(game_board, move)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_move(self, game_board, move):
        """评估移动的价值"""
        from_row, from_col = move['from']
        to_row, to_col = move['to']
        captured = move['captured']

        # 基础分数
        score = 0

        # 吃子得分
        if captured:
            piece_values = {
                'king': 10000,
                'advisor': 20,
                'elephant': 20,
                'horse': 40,
                'rook': 90,
                'cannon': 45,
                'pawn': 10
            }
            score += piece_values.get(captured['type'], 0)

        # 位置得分（简单评估）
        score += self.evaluate_position(to_row, to_col, move['piece'])

        return score

    def evaluate_position(self, row, col, piece):
        """评估棋子位置的价值"""
        position_scores = {
            'pawn': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [6, 6, 6, 6, 6, 6, 6, 6, 6],
                [8, 8, 8, 8, 8, 8, 8, 8, 8],
                [10, 10, 10, 10, 10, 10, 10, 10, 10],
                [12, 12, 12, 12, 12, 12, 12, 12, 12],
                [14, 14, 14, 14, 14, 14, 14, 14, 14],
                [16, 16, 16, 16, 16, 16, 16, 16, 16],
                [18, 18, 18, 18, 18, 18, 18, 18, 18]
            ],
            'horse': [
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [4, 6, 6, 6, 6, 6, 6, 6, 4],
                [4, 6, 8, 8, 8, 8, 8, 6, 4],
                [4, 6, 8, 10, 10, 10, 8, 6, 4],
                [4, 6, 8, 10, 12, 10, 8, 6, 4],
                [4, 6, 8, 10, 12, 10, 8, 6, 4],
                [4, 6, 8, 10, 10, 10, 8, 6, 4],
                [4, 6, 8, 8, 8, 8, 8, 6, 4],
                [4, 6, 6, 6, 6, 6, 6, 6, 4],
                [4, 4, 4, 4, 4, 4, 4, 4, 4]
            ]
        }

        piece_type = piece['type']
        if piece_type in position_scores:
            return position_scores[piece_type][row][col]
        return 0

    def learn_from_game(self, game_data):
        """从对局中学习"""
        self.game_count += 1

        # 基础学习逻辑
        winner = game_data.get('winner')
        if winner == 'ai':
            self.wins += 1
        elif winner == 'human':
            self.losses += 1
        else:
            self.draws += 1

        # 更新AI难度
        self.update_difficulty()

        # 记录学习数据
        self.learning_data['games'].append({
            'game_id': self.game_count,
            'result': winner,
            'moves': game_data.get('moves', []),
            'timestamp': datetime.now().isoformat()
        })

    def update_difficulty(self):
        """根据对局数更新AI难度"""
        if self.game_count > 1000:
            self.difficulty = 'master'
        elif self.game_count > 500:
            self.difficulty = 'advanced'
        elif self.game_count > 200:
            self.difficulty = 'intermediate'
        elif self.game_count > 100:
            self.difficulty = 'amateur'
        elif self.game_count > 50:
            self.difficulty = 'novice'
        else:
            self.difficulty = 'beginner'

    def get_stats(self):
        """获取AI统计信息"""
        total_games = self.wins + self.losses + self.draws
        win_rate = (self.wins / total_games * 100) if total_games > 0 else 0

        return {
            'games_played': self.game_count,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': round(win_rate, 1),
            'difficulty': self.difficulty,
            'difficulty_text': {
                'beginner': '小白',
                'novice': '初学者',
                'amateur': '业余',
                'intermediate': '中级',
                'advanced': '高级',
                'master': '大师'
            }.get(self.difficulty, '未知')
        }

# 学习系统类
class LearningSystem:
    def __init__(self):
        self.patterns = defaultdict(list)
        self.strategy_weights = {}
        self.move_predictions = {}
        self.learning_rate = 0.01

        # 初始化深度学习模型
        self.pattern_model = PatternRecognitionModel()
        self.position_encoder = ChessPositionEncoder()

        # 尝试加载已学习的模式
        self.load_learning_data()

    def analyze_game(self, game_data):
        """分析对局数据"""
        moves = game_data.get('moves', [])
        winner = game_data.get('winner')

        # 使用深度学习模型分析游戏
        self.pattern_model.analyze_game(game_data)

        # 传统模式分析
        for i, move in enumerate(moves):
            self.analyze_move_pattern(move, i)

        # 更新策略
        self.update_strategy(winner)

        # 保存学习数据
        self.save_learning_data()

        logging.info(f"游戏分析完成，学习模式数: {len(self.pattern_model.get_learning_summary())}")

    def analyze_move_pattern(self, move, position):
        """分析单个移动模式"""
        pattern_key = f"{move['from']}_{move['to']}_{move['piece']['type']}"
        self.patterns[pattern_key].append({
            'position': position,
            'result': move.get('result', 'unknown')
        })

    def update_strategy(self, game_result):
        """更新策略权重"""
        # 根据对局结果更新策略
        if game_result == 'ai':
            # AI获胜，加强当前策略
            for pattern in self.patterns:
                self.strategy_weights[pattern] = self.strategy_weights.get(pattern, 0.5) * 1.1
        elif game_result == 'human':
            # AI失败，减弱当前策略
            for pattern in self.patterns:
                self.strategy_weights[pattern] = self.strategy_weights.get(pattern, 0.5) * 0.9

    def predict_human_move(self, board_state, legal_moves):
        """预测张子鸣的下一步移动"""
        # 使用深度学习模型预测
        move_scores = self.pattern_model.predict_move(board_state, legal_moves)

        # 结合传统模式匹配
        for move in legal_moves:
            pattern_key = f"{move['from']}_{move['to']}_{move['piece']['type']}"
            if pattern_key in self.patterns:
                pattern_strength = len(self.patterns[pattern_key])
                move_scores[move] = move_scores.get(move, 0) + pattern_strength * 0.1

        return move_scores

    def get_ai_move_suggestion(self, board_state, legal_moves):
        """为AI提供移动建议"""
        # 预测人类可能的移动
        human_move_predictions = self.predict_human_move(board_state, legal_moves)

        # 为AI选择最佳防守或进攻策略
        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            score = 0

            # 基础评估
            if move['captured']:
                piece_values = {
                    'king': 10000, 'advisor': 20, 'elephant': 20,
                    'horse': 40, 'rook': 90, 'cannon': 45, 'pawn': 10
                }
                score += piece_values.get(move['captured']['type'], 0)

            # 基于人类预测的防守策略
            for human_move, human_score in human_move_predictions.items():
                if self.is_counter_move(move, human_move):
                    score += human_score * 10  # 防守价值

            # 基于学习的策略权重
            pattern_key = f"{move['from']}_{move['to']}_{move['piece']['type']}"
            strategy_score = self.strategy_weights.get(pattern_key, 0.5)
            score += strategy_score

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def is_counter_move(self, ai_move, human_move):
        """判断AI移动是否是对人类移动的有效应对"""
        # 简化的实现：检查是否在同一区域
        ai_to = ai_move['to']
        human_to = human_move['to']

        # 如果在同一行或列附近，认为是有效的应对
        return abs(ai_to[0] - human_to[0]) <= 2 and abs(ai_to[1] - human_to[1]) <= 2

    def get_learning_progress(self):
        """获取学习进度"""
        model_summary = self.pattern_model.get_learning_summary()

        return {
            'total_patterns': len(self.patterns),
            'model_patterns': model_summary['total_patterns'],
            'model_type': model_summary['model_type'],
            'learning_progress': model_summary['learning_progress'],
            'strategy_weights_count': len(self.strategy_weights),
            'ai_improvement_level': self.calculate_ai_level()
        }

    def calculate_ai_level(self):
        """计算AI改进水平"""
        total_patterns = len(self.patterns) + self.pattern_model.get_learning_summary()['total_patterns']
        if total_patterns < 50:
            return 0.1  # 小白水平
        elif total_patterns < 200:
            return 0.3  # 初学者
        elif total_patterns < 500:
            return 0.5  # 中级
        elif total_patterns < 1000:
            return 0.7  # 高级
        else:
            return 0.9  # 大师水平

    def save_learning_data(self):
        """保存学习数据"""
        try:
            # 保存模式模型
            pattern_file = os.path.join(app.config['DATA_DIR'], 'patterns.json')
            self.pattern_model.save_patterns(pattern_file)

            # 保存传统模式
            traditional_file = os.path.join(app.config['DATA_DIR'], 'traditional_patterns.json')
            with open(traditional_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'patterns': self.patterns,
                    'strategy_weights': self.strategy_weights
                }, f, ensure_ascii=False, indent=2)

            logging.info("学习数据保存成功")

        except Exception as e:
            logging.error(f"保存学习数据失败: {e}")

    def load_learning_data(self):
        """加载学习数据"""
        try:
            # 加载模式模型
            pattern_file = os.path.join(app.config['DATA_DIR'], 'patterns.json')
            self.pattern_model.load_patterns(pattern_file)

            # 加载传统模式
            traditional_file = os.path.join(app.config['DATA_DIR'], 'traditional_patterns.json')
            if os.path.exists(traditional_file):
                with open(traditional_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.patterns = defaultdict(list, data.get('patterns', {}))
                    self.strategy_weights = data.get('strategy_weights', {})

            logging.info("学习数据加载成功")

        except Exception as e:
            logging.warning(f"加载学习数据失败: {e}")

# 全局变量
game_board = ChessBoard()
ai_player = BasicAI()
learning_system = LearningSystem()
game_logger = GameLogger(app.config['LOG_DIR'])
progress_tracker = ProgressTracker(app.config['DATA_DIR'])

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/game/start', methods=['POST'])
def start_game():
    """开始新游戏"""
    global game_board
    game_board = ChessBoard()

    # 开始游戏日志
    game_id = game_logger.start_new_game()
    logger.info(f"新游戏开始: {game_id}")

    # 记录学习事件
    game_logger.log_learning_event(
        'game_started',
        {'game_id': game_id, 'ai_level': ai_player.difficulty},
        importance=1.0
    )

    return jsonify({
        'status': 'success',
        'game_id': game_id,
        'game_state': game_board.get_board_state()
    })

@app.route('/api/game/move', methods=['POST'])
def make_move():
    """处理移动请求"""
    start_time = datetime.now()
    data = request.get_json()
    from_pos = data.get('from')
    to_pos = data.get('to')

    # 解析位置
    from_row, from_col = map(int, from_pos.split(','))
    to_row, to_col = map(int, to_pos.split(','))

    # 获取移动前的棋子信息
    piece = game_board.board[from_row][from_col]
    captured = game_board.board[to_row][to_col]

    # 执行玩家移动
    success = game_board.make_move(from_row, from_col, to_row, to_col)

    if not success:
        return jsonify({
            'status': 'error',
            'message': '非法移动'
        }), 400

    # 记录玩家移动
    move_number = len(game_board.move_history)
    thinking_time = (datetime.now() - start_time).total_seconds()
    game_logger.log_move(
        from_pos=(from_row, from_col),
        to_pos=(to_row, to_col),
        piece=piece,
        captured=captured,
        player='human',
        move_number=move_number,
        board_state=game_board.board,
        thinking_time=thinking_time
    )

    logger.info(f"玩家移动: {from_pos} -> {to_pos}")

    # 检查游戏是否结束
    if game_board.game_state != 'playing':
        game_end_time = datetime.now()
        game_duration = (game_end_time - start_time).total_seconds()

        # 记录游戏结束
        game_logger.end_game(
            winner='human' if game_board.game_state == 'red_wins' else 'ai',
            end_reason=game_board.game_state,
            final_board=game_board.board,
            game_duration=game_duration
        )

        return jsonify({
            'status': 'success',
            'game_over': True,
            'winner': 'human' if game_board.game_state == 'red_wins' else 'ai',
            'game_state': game_board.get_board_state()
        })

    # AI回应
    ai_start_time = datetime.now()
    ai_move = ai_player.get_move(game_board)
    ai_thinking_time = (datetime.now() - ai_start_time).total_seconds()

    if ai_move:
        # 执行AI移动
        ai_from_row, ai_from_col = ai_move['from']
        ai_to_row, ai_to_col = ai_move['to']
        ai_piece = game_board.board[ai_from_row][ai_from_col]
        ai_captured = game_board.board[ai_to_row][ai_to_col]

        game_board.make_move(ai_from_row, ai_from_col, ai_to_row, ai_to_col)

        # 记录AI移动
        game_logger.log_move(
            from_pos=(ai_from_row, ai_from_col),
            to_pos=(ai_to_row, ai_to_col),
            piece=ai_piece,
            captured=ai_captured,
            player='ai',
            move_number=len(game_board.move_history),
            board_state=game_board.board,
            thinking_time=ai_thinking_time
        )

        # 记录AI思考过程
        ai_thinking_process = {
            'moves_considered': len(ai_player.get_legal_moves(game_board)),
            'time_spent': ai_thinking_time,
            'difficulty_level': ai_player.difficulty,
            'evaluation_method': 'basic' if ai_player.difficulty == 'beginner' else 'advanced'
        }

        game_logger.log_ai_thinking(
            thinking_process=ai_thinking_process,
            move_chosen=ai_move,
            evaluation_scores={'move_value': ai_player.evaluate_move(game_board, ai_move)},
            alternatives=ai_player.get_legal_moves(game_board)[:5]  # 前5个备选
        )

        logger.info(f"AI移动: {ai_move['from']} -> {ai_move['to']} (思考时间: {ai_thinking_time:.2f}秒)")

        # 检查游戏是否结束
        if game_board.game_state != 'playing':
            game_end_time = datetime.now()
            game_duration = (game_end_time - start_time).total_seconds()

            # AI学习
            learning_data = {
                'winner': 'ai' if game_board.game_state == 'black_wins' else 'human',
                'moves': game_board.move_history,
                'game_duration': game_duration,
                'total_moves': len(game_board.move_history)
            }

            ai_player.learn_from_game(learning_data)

            # 使用学习系统分析游戏
            learning_system.analyze_game(learning_data)

            # 更新进度追踪
            winner = 'ai' if game_board.game_state == 'black_wins' else 'human'
            progress_tracker.update_game_result(winner, learning_data)

            # 记录学习事件
            game_logger.log_learning_event(
                'game_completed',
                {
                    'winner': learning_data['winner'],
                    'game_duration': game_duration,
                    'total_moves': learning_data['total_moves'],
                    'ai_performance': {
                        'avg_thinking_time': ai_thinking_time,
                        'difficulty': ai_player.difficulty,
                        'learning_progress': ai_player.game_count
                    }
                },
                importance=2.0
            )

            # 记录游戏结束
            game_logger.end_game(
                winner='ai' if game_board.game_state == 'black_wins' else 'human',
                end_reason=game_board.game_state,
                final_board=game_board.board,
                game_duration=game_duration
            )

            return jsonify({
                'status': 'success',
                'game_over': True,
                'winner': 'ai' if game_board.game_state == 'black_wins' else 'human',
                'ai_move': {
                    'from': f"{ai_from_row},{ai_from_col}",
                    'to': f"{ai_to_row},{ai_to_col}"
                },
                'game_state': game_board.get_board_state()
            })

    return jsonify({
        'status': 'success',
        'ai_move': {
            'from': f"{ai_move['from'][0]},{ai_move['from'][1]}",
            'to': f"{ai_move['to'][0]},{ai_move['to'][1]}"
        } if ai_move else None,
        'game_state': game_board.get_board_state()
    })

@app.route('/api/learning/stats', methods=['GET'])
def get_learning_stats():
    """获取学习统计信息"""
    ai_stats = ai_player.get_stats()
    learning_progress = game_logger.get_ai_learning_progress()

    stats = {
        'games_played': ai_stats['games_played'],
        'wins': ai_stats['wins'],
        'losses': ai_stats['losses'],
        'draws': ai_stats['draws'],
        'win_rate': ai_stats['win_rate'],
        'patterns_learned': len(learning_system.patterns),
        'ai_level': ai_stats['difficulty_text'],
        'ai_difficulty': ai_stats['difficulty'],
        'learning_events': learning_progress['learning_events'],
        'patterns_discovered': learning_progress['patterns_discovered'],
        'avg_thinking_depth': learning_progress['avg_thinking_depth']
    }
    return jsonify(stats)

@app.route('/api/game/history', methods=['GET'])
def get_game_history():
    """获取游戏历史"""
    limit = request.args.get('limit', 10, type=int)
    history = game_logger.get_game_history(limit)
    return jsonify({
        'status': 'success',
        'history': history
    })

@app.route('/api/learning/export', methods=['POST'])
def export_logs():
    """导出日志"""
    try:
        exported_files = game_logger.export_logs()
        return jsonify({
            'status': 'success',
            'message': '日志导出成功',
            'files': exported_files
        })
    except Exception as e:
        logger.error(f"导出日志失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'导出失败: {str(e)}'
        }), 500

@app.route('/api/learning/progress', methods=['GET'])
def get_detailed_learning_progress():
    """获取详细的学习进度"""
    try:
        learning_progress = learning_system.get_learning_progress()
        model_summary = learning_system.pattern_model.get_learning_summary()

        progress_data = {
            'learning_progress': learning_progress,
            'model_summary': model_summary,
            'ai_stats': ai_player.get_stats(),
            'game_history': game_logger.get_game_history(5)  # 最近5局游戏
        }

        return jsonify(progress_data)
    except Exception as e:
        logger.error(f"获取学习进度失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

@app.route('/api/learning/patterns', methods=['GET'])
def get_learned_patterns():
    """获取学习到的模式"""
    try:
        patterns = {
            'opening_patterns': dict(learning_system.pattern_model.opening_preferences),
            'tactical_patterns': dict(learning_system.pattern_model.tactical_patterns),
            'positional_patterns': dict(learning_system.pattern_model.positional_preferences),
            'traditional_patterns': dict(learning_system.patterns),
            'strategy_weights': dict(learning_system.strategy_weights)
        }

        return jsonify({
            'status': 'success',
            'patterns': patterns
        })
    except Exception as e:
        logger.error(f"获取模式失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

@app.route('/api/progress/summary', methods=['GET'])
def get_progress_summary():
    """获取进度总结"""
    try:
        summary = progress_tracker.get_progress_summary()
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        logger.error(f"获取进度总结失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

@app.route('/api/progress/chart-data', methods=['GET'])
def get_progress_chart_data():
    """获取进度图表数据"""
    try:
        chart_data = progress_tracker.get_skill_progress_chart_data()
        return jsonify({
            'status': 'success',
            'chart_data': chart_data
        })
    except Exception as e:
        logger.error(f"获取图表数据失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

@app.route('/api/progress/milestones', methods=['GET'])
def get_progress_milestones():
    """获取里程碑信息"""
    try:
        milestones = progress_tracker.get_milestone_timeline()
        return jsonify({
            'status': 'success',
            'milestones': milestones
        })
    except Exception as e:
        logger.error(f"获取里程碑失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

@app.route('/api/progress/report', methods=['GET'])
def get_progress_report():
    """获取进度报告"""
    try:
        report = progress_tracker.generate_progress_report()
        return jsonify({
            'status': 'success',
            'report': report
        })
    except Exception as e:
        logger.error(f"获取进度报告失败: {e}")
        return jsonify({
            'status': 'error',
            'message': f'获取失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)