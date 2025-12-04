"""
游戏日志系统 - 记录每步棋和AI思考过程
"""

import json
import logging
from datetime import datetime
import os
from typing import Dict, List, Any
from collections import defaultdict

class GameLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.current_game_id = None
        self.game_log = {}
        self.move_logs = []
        self.ai_thinking_logs = []
        self.learning_logs = []

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志文件
        self.game_log_file = os.path.join(log_dir, 'games.jsonl')
        self.move_log_file = os.path.join(log_dir, 'moves.jsonl')
        self.ai_thinking_log_file = os.path.join(log_dir, 'ai_thinking.jsonl')
        self.learning_log_file = os.path.join(log_dir, 'learning.jsonl')
        self.analysis_log_file = os.path.join(log_dir, 'analysis.jsonl')

        # 初始化日志文件
        self._init_log_files()

    def _init_log_files(self):
        """初始化日志文件"""
        for log_file in [self.game_log_file, self.move_log_file,
                        self.ai_thinking_log_file, self.learning_log_file,
                        self.analysis_log_file]:
            if not os.path.exists(log_file):
                with open(log_file, 'w', encoding='utf-8') as f:
                    pass  # 创建空文件

    def start_new_game(self, game_id: str = None):
        """开始新游戏日志"""
        if game_id is None:
            game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_game_id = game_id
        self.game_log = {
            'game_id': game_id,
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'ai_thinking': [],
            'learning_events': [],
            'game_state': 'started'
        }

        self.move_logs = []
        self.ai_thinking_logs = []
        self.learning_logs = []

        # 记录游戏开始
        self._write_log(self.game_log_file, {
            'type': 'game_start',
            'game_id': game_id,
            'timestamp': datetime.now().isoformat(),
            'data': self.game_log
        })

        logging.info(f"新游戏开始: {game_id}")
        return game_id

    def log_move(self, from_pos: tuple, to_pos: tuple, piece: dict,
                 captured: dict, player: str, move_number: int,
                 board_state: list, thinking_time: float = None):
        """记录移动"""
        move_data = {
            'game_id': self.current_game_id,
            'move_number': move_number,
            'from': from_pos,
            'to': to_pos,
            'piece': piece,
            'captured': captured,
            'player': player,
            'timestamp': datetime.now().isoformat(),
            'thinking_time': thinking_time,
            'board_state': self._serialize_board(board_state)
        }

        self.move_logs.append(move_data)
        self.game_log['moves'].append(move_data)

        # 写入移动日志
        self._write_log(self.move_log_file, {
            'type': 'move',
            'timestamp': datetime.now().isoformat(),
            'data': move_data
        })

        logging.info(f"移动记录: {player} {piece['type']} {from_pos} -> {to_pos}")

    def log_ai_thinking(self, thinking_process: dict, move_chosen: dict,
                       evaluation_scores: dict = None, alternatives: list = None):
        """记录AI思考过程"""
        thinking_data = {
            'game_id': self.current_game_id,
            'timestamp': datetime.now().isoformat(),
            'thinking_process': thinking_process,
            'move_chosen': move_chosen,
            'evaluation_scores': evaluation_scores or {},
            'alternatives_considered': alternatives or [],
            'thinking_depth': len(thinking_process.get('moves_considered', [])),
            'time_spent': thinking_process.get('time_spent', 0)
        }

        self.ai_thinking_logs.append(thinking_data)
        self.game_log['ai_thinking'].append(thinking_data)

        # 写入AI思考日志
        self._write_log(self.ai_thinking_log_file, {
            'type': 'ai_thinking',
            'timestamp': datetime.now().isoformat(),
            'data': thinking_data
        })

        logging.info(f"AI思考记录: 考虑了{len(thinking_data['alternatives_considered'])}个选项")

    def log_learning_event(self, event_type: str, event_data: dict,
                          importance: float = 1.0):
        """记录学习事件"""
        learning_data = {
            'game_id': self.current_game_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': event_data,
            'importance': importance
        }

        self.learning_logs.append(learning_data)
        self.game_log['learning_events'].append(learning_data)

        # 写入学习日志
        self._write_log(self.learning_log_file, {
            'type': 'learning',
            'timestamp': datetime.now().isoformat(),
            'data': learning_data
        })

        logging.info(f"学习事件: {event_type} - 重要性: {importance}")

    def log_pattern_discovery(self, pattern: dict, confidence: float,
                             context: dict = None):
        """记录模式发现"""
        pattern_data = {
            'game_id': self.current_game_id,
            'timestamp': datetime.now().isoformat(),
            'pattern': pattern,
            'confidence': confidence,
            'context': context or {}
        }

        # 写入分析日志
        self._write_log(self.analysis_log_file, {
            'type': 'pattern_discovery',
            'timestamp': datetime.now().isoformat(),
            'data': pattern_data
        })

        logging.info(f"模式发现: {pattern.get('type', 'unknown')} - 置信度: {confidence:.2f}")

    def end_game(self, winner: str, end_reason: str, final_board: list,
                 game_duration: float):
        """结束游戏日志"""
        self.game_log.update({
            'end_time': datetime.now().isoformat(),
            'winner': winner,
            'end_reason': end_reason,
            'game_duration': game_duration,
            'total_moves': len(self.move_logs),
            'final_board': self._serialize_board(final_board),
            'game_state': 'ended'
        })

        # 记录游戏结束
        self._write_log(self.game_log_file, {
            'type': 'game_end',
            'game_id': self.current_game_id,
            'timestamp': datetime.now().isoformat(),
            'data': self.game_log
        })

        logging.info(f"游戏结束: {winner} 获胜, 原因: {end_reason}, 时长: {game_duration:.2f}秒")

        # 生成游戏总结
        self._generate_game_summary()

        return self.game_log

    def _generate_game_summary(self):
        """生成游戏总结"""
        total_moves = len(self.move_logs)
        ai_moves = [m for m in self.move_logs if m['player'] == 'ai']
        human_moves = [m for m in self.move_logs if m['player'] == 'human']

        summary = {
            'game_id': self.current_game_id,
            'total_moves': total_moves,
            'ai_moves': len(ai_moves),
            'human_moves': len(human_moves),
            'captures': len([m for m in self.move_logs if m['captured']]),
            'learning_events': len(self.learning_logs),
            'ai_thinking_depth_avg': sum(t['thinking_depth'] for t in self.ai_thinking_logs) / len(self.ai_thinking_logs) if self.ai_thinking_logs else 0,
            'key_patterns_identified': self._extract_key_patterns(),
            'improvement_suggestions': self._generate_improvement_suggestions()
        }

        # 写入总结日志
        self._write_log(self.analysis_log_file, {
            'type': 'game_summary',
            'timestamp': datetime.now().isoformat(),
            'data': summary
        })

        logging.info(f"游戏总结生成: {summary}")

    def _extract_key_patterns(self):
        """提取关键模式"""
        patterns = []

        # 分析移动模式
        move_sequences = []
        for i in range(len(self.move_logs) - 2):
            sequence = [
                self.move_logs[i]['piece']['type'],
                self.move_logs[i+1]['piece']['type'],
                self.move_logs[i+2]['piece']['type']
            ]
            move_sequences.append(sequence)

        # 找出常见的移动序列
        from collections import Counter
        sequence_counts = Counter(tuple(seq) for seq in move_sequences)
        for seq, count in sequence_counts.items():
            if count >= 2:  # 出现至少2次的模式
                patterns.append({
                    'type': 'move_sequence',
                    'sequence': list(seq),
                    'frequency': count,
                    'description': f"移动序列模式: {'-'.join(seq)}"
                })

        return patterns

    def _generate_improvement_suggestions(self):
        """生成改进建议"""
        suggestions = []

        # 分析AI表现
        ai_moves = [m for m in self.move_logs if m['player'] == 'ai']
        if ai_moves:
            # 分析思考深度
            avg_thinking_depth = sum(t['thinking_depth'] for t in self.ai_thinking_logs) / len(self.ai_thinking_logs) if self.ai_thinking_logs else 0
            if avg_thinking_depth < 3:
                suggestions.append({
                    'type': 'thinking_depth',
                    'description': '建议增加AI思考深度，考虑更多可能的移动',
                    'priority': 'medium'
                })

        # 分析学习效率
        learning_events = [e for e in self.learning_logs if e['importance'] > 0.8]
        if len(learning_events) < len(self.move_logs) * 0.1:
            suggestions.append({
                'type': 'learning_efficiency',
                'description': '学习事件较少，建议增加模式识别频率',
                'priority': 'low'
            })

        return suggestions

    def get_game_history(self, limit: int = 10):
        """获取游戏历史"""
        history = []
        try:
            with open(self.game_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('type') == 'game_end':
                        history.append(data['data'])
                        if len(history) >= limit:
                            break
        except FileNotFoundError:
            pass

        return history[-limit:]  # 返回最近的limit条记录

    def get_ai_learning_progress(self):
        """获取AI学习进度"""
        learning_data = {
            'total_games': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'learning_events': 0,
            'patterns_discovered': 0,
            'avg_thinking_depth': 0,
            'improvement_areas': []
        }

        try:
            with open(self.game_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('type') == 'game_end':
                        game_data = data['data']
                        learning_data['total_games'] += 1
                        winner = game_data.get('winner')
                        if winner == 'ai':
                            learning_data['wins'] += 1
                        elif winner == 'human':
                            learning_data['losses'] += 1
                        else:
                            learning_data['draws'] += 1

            # 统计学习事件
            with open(self.learning_log_file, 'r', encoding='utf-8') as f:
                learning_data['learning_events'] = sum(1 for line in f if line.strip())

            # 统计模式发现
            with open(self.analysis_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('type') == 'pattern_discovery':
                        learning_data['patterns_discovered'] += 1

        except FileNotFoundError:
            pass

        # 计算平均思考深度
        thinking_depths = []
        try:
            with open(self.ai_thinking_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    thinking_depths.append(data['data']['thinking_depth'])
        except FileNotFoundError:
            pass

        if thinking_depths:
            learning_data['avg_thinking_depth'] = sum(thinking_depths) / len(thinking_depths)

        return learning_data

    def _write_log(self, log_file: str, log_entry: dict):
        """写入日志文件"""
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logging.error(f"写入日志失败: {e}")

    def _serialize_board(self, board: list) -> list:
        """序列化棋盘状态"""
        serialized = []
        for row in board:
            serialized_row = []
            for piece in row:
                if piece:
                    serialized_row.append(piece)
                else:
                    serialized_row.append(None)
            serialized.append(serialized_row)
        return serialized

    def export_logs(self, export_dir: str = 'export'):
        """导出日志"""
        os.makedirs(export_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        export_files = {
            'games': self.game_log_file,
            'moves': self.move_log_file,
            'ai_thinking': self.ai_thinking_log_file,
            'learning': self.learning_log_file,
            'analysis': self.analysis_log_file
        }

        exported_files = {}
        for name, source_file in export_files.items():
            if os.path.exists(source_file):
                export_file = os.path.join(export_dir, f'{name}_{timestamp}.jsonl')
                import shutil
                shutil.copy2(source_file, export_file)
                exported_files[name] = export_file

        logging.info(f"日志导出完成: {exported_files}")
        return exported_files