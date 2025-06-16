# Talia AI Daemon

A daemon service for text classification and summarization using the Gaia model and DistilBERT for Portuguese text.

## Features

- Text summarization using the Gaia model
- Text classification using DistilBERT
- gRPC server over Unix Domain Socket
- Health check endpoint
- Environment-based configuration

## Project Structure

```
talia-ai-daemon/
├── src/
│   ├── __init__.py
│   ├── talia_daemon.py
│   ├── client.py
│   └── proto/
│       ├── talia.proto
│       ├── talia_pb2.py
│       └── talia_pb2_grpc.py
├── tests/
│   └── test_talia_daemon.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Talia-Project/ai-daemon.git
cd talia-ai-daemon
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

Create a `.env` file with the following variables:
```
SOCKET_PATH=/tmp/daemon.sock
```

## Usage

1. Start the daemon:
```bash
python -m src.talia_daemon
```

2. Run the client example:
```bash
python -m src.client
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest tests/
```

## License

MIT License 