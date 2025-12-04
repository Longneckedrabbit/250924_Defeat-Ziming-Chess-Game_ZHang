"""
AI学习进度追踪系统 - 可视化AI的学习和进步情况
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import defaultdict, deque
import logging

class ProgressTracker:
    """AI学习进度追踪器"""

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.progress_file = os.path.join(data_dir, 'progress.json')
        self.milestones_file = os.path.join(data_dir, 'milestones.json')

        # 进度数据
        self.progress_data = {
            'start_time': datetime.now().isoformat(),
            'total_games': 0,
            'ai_wins': 0,
            'human_wins': 0,
            'draws': 0,
            'learning_sessions': 0,
            'skill_progress': [],
            'pattern_discovery': [],
            'performance_metrics': [],
            'milestones_achieved': []
        }

        # 学习里程碑
        self.milestones = {
            'first_game': {'name': '第一局游戏', 'description': '完成与张子鸣的第一局对弈', 'threshold': 1, 'achieved': False},
            'novice_level': {'name': '初学者水平', 'description': 'AI达到初学者水平', 'threshold': 50, 'achieved': False},
            'pattern_recognition': {'name': '模式识别', 'description': 'AI开始识别张子鸣的棋路模式', 'threshold': 10, 'achieved': False},
            'tactical_learning': {'name': '战术学习', 'description': 'AI学习到基本的战术模式', 'threshold': 100, 'achieved': False},
            'strategic_thinking': {'name': '战略思考', 'description': 'AI展现出战略性思考能力', 'threshold': 200, 'achieved': False},
            'amateur_level': {'name': '业余水平', 'description': 'AI达到业余棋手水平', 'threshold': 200, 'achieved': False},
            'advanced_patterns': {'name': '高级模式', 'description': '识别复杂的棋路模式', 'threshold': 300, 'achieved': False},
            'intermediate_level': {'name': '中级水平', 'description': 'AI达到中级棋手水平', 'threshold': 500, 'achieved': False},
            'prediction_ability': {'name': '预测能力', 'description': 'AI能够预测张子鸣的移动', 'threshold': 500, 'achieved': False},
            'advanced_level': {'name': '高级水平', 'description': 'AI达到高级棋手水平', 'threshold': 1000, 'achieved': False},
            'master_level': {'name': '大师水平', 'description': 'AI达到大师水平，能够超越张子鸣', 'threshold': 1500, 'achieved': False},
            'superior_intelligence': {'name': '超越人类', 'description': 'AI在各方面超越张子鸣', 'threshold': 2000, 'achieved': False}
        }

        # 加载现有数据
        self.load_progress_data()
        self.load_milestones()

    def update_game_result(self, winner: str, game_data: Dict):
        """更新游戏结果"""
        self.progress_data['total_games'] += 1
        self.progress_data['learning_sessions'] += 1

        if winner == 'ai':
            self.progress_data['ai_wins'] += 1
        elif winner == 'human':
            self.progress_data['human_wins'] += 1
        else:
            self.progress_data['draws'] += 1

        # 更新技能进度
        self.update_skill_progress(game_data)

        # 检查里程碑
        self.check_milestones()

        # 保存数据
        self.save_progress_data()

    def update_skill_progress(self, game_data: Dict):
        """更新技能进度数据"""
        current_time = datetime.now()

        # 计算技能指标
        total_games = self.progress_data['total_games']
        win_rate = self.progress_data['ai_wins'] / max(total_games, 1)

        # 技能等级评估
        skill_level = self.calculate_skill_level()

        # 学习效率
        learning_efficiency = self.calculate_learning_efficiency(game_data)

        # 模式识别能力
        pattern_recognition = self.calculate_pattern_recognition()

        # 战术理解
        tactical_understanding = self.calculate_tactical_understanding()

        # 预测准确率
        prediction_accuracy = self.calculate_prediction_accuracy()

        skill_entry = {
            'timestamp': current_time.isoformat(),
            'game_number': total_games,
            'skill_level': skill_level,
            'win_rate': win_rate,
            'learning_efficiency': learning_efficiency,
            'pattern_recognition': pattern_recognition,
            'tactical_understanding': tactical_understanding,
            'prediction_accuracy': prediction_accuracy,
            'overall_progress': self.calculate_overall_progress()
        }

        self.progress_data['skill_progress'].append(skill_entry)

        # 只保留最近1000条记录
        if len(self.progress_data['skill_progress']) > 1000:
            self.progress_data['skill_progress'] = self.progress_data['skill_progress'][-1000:]

    def calculate_skill_level(self) -> float:
        """计算技能等级 (0-100)"""
        total_games = self.progress_data['total_games']
        win_rate = self.progress_data['ai_wins'] / max(total_games, 1)

        # 基础分数基于游戏数量
        base_score = min(total_games / 20, 50)  # 每20局游戏得1分，最多50分

        # 胜率分数
        win_rate_score = win_rate * 50

        return base_score + win_rate_score

    def calculate_learning_efficiency(self, game_data: Dict) -> float:
        """计算学习效率 (0-100)"""
        moves = game_data.get('moves', [])
        if not moves:
            return 0

        # 计算移动多样性
        piece_types_used = len(set(move['piece']['type'] for move in moves if move['player'] == 'ai'))
        diversity_score = min(piece_types_used / 7 * 100, 100)  # 7种棋子类型

        # 计算适应性（基于游戏长度）
        ideal_length = 50  # 理想游戏长度
        game_length = len(moves)
        adaptation_score = 100 - abs(game_length - ideal_length) / ideal_length * 100

        return (diversity_score + adaptation_score) / 2

    def calculate_pattern_recognition(self) -> float:
        """计算模式识别能力 (0-100)"""
        try:
            with open(os.path.join(self.data_dir, 'patterns.json'), 'r', encoding='utf-8') as f:
                patterns = json.load(f)

            total_patterns = (len(patterns.get('opening_preferences', {})) +
                           len(patterns.get('tactical_patterns', {})) +
                           len(patterns.get('positional_preferences', {})))

            return min(total_patterns / 5 * 100, 100)  # 每5个模式得1分，最多100分
        except:
            return 0

    def calculate_tactical_understanding(self) -> float:
        """计算战术理解能力 (0-100)"""
        skill_progress = self.progress_data['skill_progress']
        if len(skill_progress) < 5:
            return 0

        # 基于最近5局的胜率变化
        recent_games = skill_progress[-5:]
        win_rate_trend = recent_games[-1]['win_rate'] - recent_games[0]['win_rate']

        # 基于技能等级提升
        skill_trend = recent_games[-1]['skill_level'] - recent_games[0]['skill_level']

        return min((win_rate_trend + skill_trend) * 100 + 50, 100)

    def calculate_prediction_accuracy(self) -> float:
        """计算预测准确率 (0-100)"""
        # 简化实现，实际需要基于预测结果和实际移动的对比
        total_games = self.progress_data['total_games']
        if total_games < 10:
            return 0

        # 基于胜率和游戏数量的综合评估
        win_rate = self.progress_data['ai_wins'] / total_games
        game_factor = min(total_games / 100, 1)

        return win_rate * game_factor * 100

    def calculate_overall_progress(self) -> float:
        """计算总体学习进度 (0-100)"""
        total_games = self.progress_data['total_games']

        if total_games < 50:
            return total_games / 50 * 20  # 小白阶段：0-20%
        elif total_games < 200:
            return 20 + (total_games - 50) / 150 * 30  # 初学者到业余：20-50%
        elif total_games < 500:
            return 50 + (total_games - 200) / 300 * 20  # 业余到中级：50-70%
        elif total_games < 1000:
            return 70 + (total_games - 500) / 500 * 20  # 中级到高级：70-90%
        else:
            return 90 + min((total_games - 1000) / 1000 * 10, 10)  # 高级到大师：90-100%

    def check_milestones(self):
        """检查并更新里程碑"""
        for key, milestone in self.milestones.items():
            if not milestone['achieved']:
                threshold_type = self.get_milestone_threshold_type(key)
                current_value = self.get_milestone_current_value(threshold_type)

                if current_value >= milestone['threshold']:
                    milestone['achieved'] = True
                    milestone['achieved_at'] = datetime.now().isoformat()
                    self.progress_data['milestones_achieved'].append({
                        'milestone': key,
                        'name': milestone['name'],
                        'achieved_at': milestone['achieved_at']
                    })
                    logging.info(f"里程碑达成: {milestone['name']}")

    def get_milestone_threshold_type(self, milestone_key: str) -> str:
        """获取里程碑的阈值类型"""
        if milestone_key in ['first_game', 'novice_level', 'amateur_level', 'intermediate_level', 'advanced_level', 'master_level']:
            return 'games'
        elif milestone_key in ['pattern_recognition', 'tactical_learning', 'advanced_patterns']:
            return 'patterns'
        elif milestone_key in ['strategic_thinking', 'prediction_ability', 'superior_intelligence']:
            return 'skill'
        return 'games'

    def get_milestone_current_value(self, threshold_type: str) -> int:
        """获取里程碑的当前值"""
        if threshold_type == 'games':
            return self.progress_data['total_games']
        elif threshold_type == 'patterns':
            return self.calculate_pattern_recognition() // 20  # 每20%算1个模式
        elif threshold_type == 'skill':
            return int(self.calculate_skill_level())
        return 0

    def get_progress_summary(self) -> Dict:
        """获取进度总结"""
        current_skill = self.calculate_skill_level()
        overall_progress = self.calculate_overall_progress()

        # 计算达到下一个里程碑的进度
        next_milestone = self.get_next_milestone()
        milestone_progress = self.calculate_milestone_progress(next_milestone)

        return {
            'total_games': self.progress_data['total_games'],
            'ai_wins': self.progress_data['ai_wins'],
            'human_wins': self.progress_data['human_wins'],
            'draws': self.progress_data['draws'],
            'win_rate': self.progress_data['ai_wins'] / max(self.progress_data['total_games'], 1),
            'skill_level': current_skill,
            'skill_rank': self.get_skill_rank(current_skill),
            'overall_progress': overall_progress,
            'next_milestone': next_milestone,
            'milestone_progress': milestone_progress,
            'achieved_milestones': len([m for m in self.milestones.values() if m['achieved']]),
            'total_milestones': len(self.milestones),
            'learning_streak': self.calculate_learning_streak(),
            'estimated_completion': self.estimate_completion_time()
        }

    def get_skill_rank(self, skill_level: float) -> str:
        """根据技能等级获取段位"""
        if skill_level < 20:
            return "小白"
        elif skill_level < 35:
            return "初学者"
        elif skill_level < 50:
            return "业余"
        elif skill_level < 65:
            return "中级"
        elif skill_level < 80:
            return "高级"
        elif skill_level < 90:
            return "大师"
        else:
            return "宗师"

    def get_next_milestone(self) -> Dict:
        """获取下一个未达成的里程碑"""
        for milestone in self.milestones.values():
            if not milestone['achieved']:
                return milestone
        return None

    def calculate_milestone_progress(self, milestone: Dict) -> float:
        """计算到下一个里程碑的进度"""
        if not milestone:
            return 100.0

        threshold_type = self.get_milestone_threshold_type(list(self.milestones.keys())[list(self.milestones.values()).index(milestone)])
        current_value = self.get_milestone_current_value(threshold_type)
        return min(current_value / milestone['threshold'] * 100, 100)

    def calculate_learning_streak(self) -> int:
        """计算连续学习天数"""
        if not self.progress_data['skill_progress']:
            return 0

        # 检查最近的学习记录
        current_date = datetime.now().date()
        streak = 0

        for entry in reversed(self.progress_data['skill_progress']):
            entry_date = datetime.fromisoformat(entry['timestamp']).date()
            if (current_date - entry_date).days == streak:
                streak += 1
            else:
                break

        return streak

    def estimate_completion_time(self) -> str:
        """预计完成所有学习的时间"""
        total_games = self.progress_data['total_games']
        overall_progress = self.calculate_overall_progress()

        if overall_progress < 1:
            return "无法估计"

        # 基于当前进度计算
        estimated_total_games = total_games / overall_progress * 100
        remaining_games = estimated_total_games - total_games

        # 假设每天10局游戏
        estimated_days = remaining_games / 10

        if estimated_days < 1:
            return "今天"
        elif estimated_days < 7:
            return f"{int(estimated_days)}天"
        elif estimated_days < 30:
            return f"{int(estimated_days / 7)}周"
        else:
            return f"{int(estimated_days / 30)}个月"

    def get_skill_progress_chart_data(self) -> Dict:
        """获取技能进度图表数据"""
        skill_progress = self.progress_data['skill_progress']
        if not skill_progress:
            return {}

        # 准备图表数据
        chart_data = {
            'labels': [entry['timestamp'] for entry in skill_progress[-50:]],  # 最近50局
            'skill_level': [entry['skill_level'] for entry in skill_progress[-50:]],
            'win_rate': [entry['win_rate'] * 100 for entry in skill_progress[-50:]],
            'learning_efficiency': [entry['learning_efficiency'] for entry in skill_progress[-50:]],
            'pattern_recognition': [entry['pattern_recognition'] for entry in skill_progress[-50:]]
        }

        return chart_data

    def get_milestone_timeline(self) -> List[Dict]:
        """获取里程碑时间线"""
        timeline = []
        for key, milestone in self.milestones.items():
            timeline.append({
                'key': key,
                'name': milestone['name'],
                'description': milestone['description'],
                'threshold': milestone['threshold'],
                'achieved': milestone['achieved'],
                'achieved_at': milestone.get('achieved_at', None)
            })
        return timeline

    def save_progress_data(self):
        """保存进度数据"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存进度数据失败: {e}")

    def load_progress_data(self):
        """加载进度数据"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.progress_data = json.load(f)
        except Exception as e:
            logging.warning(f"加载进度数据失败: {e}")

    def save_milestones(self):
        """保存里程碑数据"""
        try:
            with open(self.milestones_file, 'w', encoding='utf-8') as f:
                json.dump(self.milestones, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存里程碑数据失败: {e}")

    def load_milestones(self):
        """加载里程碑数据"""
        try:
            if os.path.exists(self.milestones_file):
                with open(self.milestones_file, 'r', encoding='utf-8') as f:
                    loaded_milestones = json.load(f)
                    # 合并现有里程碑状态
                    for key, milestone in loaded_milestones.items():
                        if key in self.milestones:
                            self.milestones[key].update(milestone)
        except Exception as e:
            logging.warning(f"加载里程碑数据失败: {e}")

    def generate_progress_report(self) -> str:
        """生成进度报告"""
        summary = self.get_progress_summary()

        report = f"""
=== AI学习进度报告 ===
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 基础统计:
- 总对局数: {summary['total_games']}
- AI胜利: {summary['ai_wins']}
- 张子鸣胜利: {summary['human_wins']}
- 平局: {summary['draws']}
- 胜率: {summary['win_rate']:.1%}

🎯 技能评估:
- 技能等级: {summary['skill_level']:.1f}/100
- 当前段位: {summary['skill_rank']}
- 总体进度: {summary['overall_progress']:.1f}%

🏆 里程碑:
- 已达成: {summary['achieved_milestones']}/{summary['total_milestones']}
- 下一个里程碑: {summary['next_milestone']['name'] if summary['next_milestone'] else '全部完成'}
- 里程碑进度: {summary['milestone_progress']:.1f}%

📈 学习状态:
- 连续学习: {summary['learning_streak']}天
- 预计完成: {summary['estimated_completion']}

💡 改进建议:
{self.generate_improvement_suggestions()}
"""

        return report

    def generate_improvement_suggestions(self) -> str:
        """生成改进建议"""
        suggestions = []
        summary = self.get_progress_summary()

        if summary['win_rate'] < 0.3:
            suggestions.append("- 加强基础战术训练，提高吃子能力")
        if summary['skill_level'] < 30:
            suggestions.append("- 增加对局数量，积累更多经验")
        if summary['overall_progress'] < 50:
            suggestions.append("- 专注于模式识别，学习张子鸣的棋路")
        if summary['learning_streak'] < 3:
            suggestions.append("- 保持规律的对局频率，提高学习效率")

        if not suggestions:
            return "- 当前学习状态良好，继续保持！"

        return "\n".join(suggestions)