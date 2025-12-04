# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Development Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Development Server
```bash
python run.py
```

### Generate Deployment Files
```bash
python deploy.py
```

### Run Tests
```bash
# No specific test command found, but tests would typically be run with:
python -m pytest tests/
```

## 🏗️ High-Level Architecture

### Core Components

1. **Main Application** (`app.py`):
   - Flask web application serving the chess game interface
   - Implements game logic, AI player, and learning system
   - RESTful API endpoints for game control and learning data

2. **Chess Rules Engine** (`chess_rules.py`):
   - Validates moves according to Chinese chess rules
   - Evaluates board positions
   - Manages game state transitions

3. **AI Player System** (`app.py`):
   - BasicAI class with multiple difficulty levels (beginner to master)
   - Move selection based on difficulty and learned patterns
   - Game result learning and difficulty progression

4. **Learning System** (`app.py`):
   - PatternRecognitionModel for deep learning-based pattern recognition
   - Traditional pattern matching and strategy weights
   - Predicts human moves and provides AI counter-strategies
   - Integrates with TensorFlow/PyTorch for neural network capabilities

5. **Game Logger** (`game_logger.py`):
   - Records detailed game logs and AI thinking processes
   - Tracks learning events and pattern discoveries
   - Exports data for analysis

6. **Progress Tracker** (`progress_tracker.py`):
   - Monitors AI learning progress and skill development
   - Generates reports and milestone tracking
   - Provides chart data for visualization

### Data Flow

1. **Game Initialization**:
   - User starts game via `/api/game/start`
   - ChessBoard initializes standard Chinese chess setup
   - GameLogger starts new game log

2. **Player Move**:
   - User submits move via `/api/game/move`
   - ChessRules validates move legality
   - Game state updated and logged
   - AI player calculates response move
   - LearningSystem analyzes patterns and adjusts strategies

3. **AI Learning**:
   - Game results fed to BasicAI for difficulty adjustment
   - LearningSystem analyzes complete games for patterns
   - Pattern data saved to disk for persistence
   - ProgressTracker updates skill metrics

### Deployment Architecture

1. **Containerized Deployment**:
   - Docker configuration with separate services for app and nginx
   - Nginx reverse proxy with SSL termination
   - Gunicorn WSGI server for Flask application

2. **Production Configuration**:
   - Environment-based configuration management
   - SSL certificate support (Let's Encrypt compatible)
   - SystemD service files for traditional deployment

### Key Directories

- `/templates` - HTML templates for web interface
- `/static` - CSS, JavaScript, and image assets
- `/data` - Persistent learning data and patterns
- `/logs` - Game logs and system logs
- `/ssl` - SSL certificates (development only)