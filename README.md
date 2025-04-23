# Speech Emotion Recognition System

A real-time system for recognizing emotions and identifying speakers from audio input, built with Python, PyTorch, and FastAPI.

## Features

- Real-time audio processing
- Emotion recognition from speech
- Speaker identification
- RESTful API interface
- Real-time visualization dashboard
- Performance monitoring and logging

## Requirements

- Python 3.9+
- PostgreSQL 15
- Redis 7.0
- NVIDIA GPU (recommended)
- Docker (optional)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Start the development server:
```bash
uvicorn src.api.main:app --reload
```

## Development Setup

### Using Docker

1. Build the development image:
```bash
docker build -t ser-system:dev -f docker/Dockerfile.dev .
```

2. Run the container:
```bash
docker run -it --name ser-dev \
  -v $(pwd):/app \
  -p 8000:8000 \
  ser-system:dev
```

### Database Setup

1. Start PostgreSQL:
```bash
docker run -d --name ser-db \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=ser_db \
  -p 5432:5432 \
  postgres:15
```

2. Run migrations:
```bash
alembic upgrade head
```

## Project Structure

```
speech-emotion-recognition/
├── src/                    # Source code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   │   ├── audio/        # Audio processing
│   │   ├── models/       # ML models
│   │   └── utils/        # Utilities
│   └── frontend/         # Frontend code
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
├── requirements.txt      # Production dependencies
└── requirements-dev.txt  # Development dependencies
```

## Testing

Run the test suite:
```bash
pytest
```

With coverage:
```bash
pytest --cov=src tests/
```

## Documentation

Generate documentation:
```bash
cd docs
make html
```

View at `docs/_build/html/index.html`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 