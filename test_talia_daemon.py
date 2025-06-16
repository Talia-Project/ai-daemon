# Standard library imports
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

# Third-party imports
import grpc
import pytest
import talia_pb2
import talia_pb2_grpc

# Local application imports
import talia_daemon

# Constants
FLOAT_TOLERANCE = 0.01  # 1% tolerance for float comparisons

# Fixtures
@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before and after each test"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def daemon():
    """Create a daemon instance with proper cleanup"""
    daemon = talia_daemon.TaliaDaemon()
    yield daemon
    # Cleanup
    if hasattr(daemon, 'grpc_server') and daemon.grpc_server:
        try:
            daemon.stop_grpc_server()
        except Exception:
            pass
    if hasattr(daemon, 'running') and daemon.running:
        daemon.stop()

@pytest.fixture
def mock_tokenizer():
    """Mock the tokenizer with proper cleanup"""
    with patch('talia_daemon.AutoTokenizer') as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_model():
    """Mock the model with proper cleanup"""
    with patch('talia_daemon.AutoModelForCausalLM') as mock:
        mock.from_pretrained.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_classifier():
    """Mock the classifier with proper cleanup"""
    with patch('talia_daemon.pipeline') as mock:
        mock.return_value = MagicMock()
        yield mock

@pytest.fixture
def mock_grpc_server():
    """Mock the gRPC server with proper cleanup"""
    with patch('talia_daemon.grpc.server') as mock:
        server_instance = MagicMock()
        mock.return_value = server_instance
        yield mock

class TestTaliaDaemon:
    class TestModelLoading:
        def test_load_model_success(self, daemon, mock_tokenizer, mock_model, mock_classifier):
            """Test successful model loading"""
            daemon.load_model()
            assert daemon.tokenizer is not None, "Tokenizer should be initialized"
            assert daemon.model is not None, "Model should be initialized"
            assert daemon.classifier is not None, "Classifier should be initialized"
            mock_tokenizer.from_pretrained.assert_called_once_with('CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it')
            mock_model.from_pretrained.assert_called_once_with('CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it')
            mock_classifier.assert_called_once_with("text-classification", model="adalbertojunior/distilbert-portuguese-cased")

        def test_load_model_tokenizer_failure(self, daemon, mock_tokenizer, mock_model, mock_classifier):
            """Test tokenizer loading failure"""
            mock_tokenizer.from_pretrained.side_effect = Exception("Tokenizer error")
            with pytest.raises(SystemExit) as exc_info:
                daemon.load_model()
            assert exc_info.value.code == 1, "Should exit with code 1 on tokenizer error"

        def test_load_model_classifier_failure(self, daemon, mock_tokenizer, mock_model, mock_classifier):
            """Test classifier loading failure"""
            mock_classifier.side_effect = Exception("Classifier error")
            with pytest.raises(SystemExit) as exc_info:
                daemon.load_model()
            assert exc_info.value.code == 1, "Should exit with code 1 on classifier error"

        def test_load_model_model_failure(self, daemon, mock_tokenizer, mock_model, mock_classifier):
            """Test model loading failure"""
            mock_model.from_pretrained.side_effect = Exception("Model error")
            with pytest.raises(SystemExit) as exc_info:
                daemon.load_model()
            assert exc_info.value.code == 1, "Should exit with code 1 on model error"

    class TestSummarization:
        def test_summarize_text_success(self, daemon):
            """Test successful text summarization"""
            # Setup tokenizer mock
            tokenizer_mock = MagicMock()
            tokenizer_mock.return_value = {'input_ids': [[1, 2, 3]], 'attention_mask': [[1, 1, 1]]}
            daemon.tokenizer = tokenizer_mock
            
            # Setup model mock
            model_mock = MagicMock()
            model_mock.generate.return_value = [[1, 2, 3, 4]]
            daemon.model = model_mock
            
            # Setup decode mock
            daemon.tokenizer.decode = MagicMock(return_value="Resumo gerado.")
            
            resumo = daemon.summarize_text("Texto de teste.")
            assert resumo == "Resumo gerado.", "Should return the decoded summary"
            tokenizer_mock.assert_called_once_with("Texto de teste.")
            model_mock.generate.assert_called_once()
            daemon.tokenizer.decode.assert_called_once()

        def test_summarize_text_empty_input(self, daemon):
            """Test summarization with empty input"""
            daemon.tokenizer = MagicMock()
            daemon.model = MagicMock()
            resumo = daemon.summarize_text("")
            assert resumo is None, "Should return None for empty input"

        def test_summarize_text_model_error(self, daemon):
            """Test summarization with model error"""
            daemon.tokenizer = MagicMock()
            daemon.model = MagicMock()
            daemon.model.generate.side_effect = Exception("Model error")
            resumo = daemon.summarize_text("Texto de teste.")
            assert resumo is None, "Should return None on model error"

        def test_summarize_text_tokenizer_error(self, daemon):
            """Test summarization with tokenizer error"""
            daemon.tokenizer = MagicMock()
            daemon.model = MagicMock()
            daemon.tokenizer.side_effect = Exception("Tokenizer error")
            resumo = daemon.summarize_text("Texto de teste.")
            assert resumo is None, "Should return None on tokenizer error"

    class TestClassification:
        def test_classify_text_success(self, daemon):
            """Test successful text classification"""
            daemon.classifier = MagicMock()
            daemon.classifier.return_value = [{'label': 'classe1', 'score': 0.95}]
            result = daemon.classify_text("Texto de teste.", ["classe1", "classe2"])
            assert result is not None, "Should return a result"
            assert result['label'] == 'classe1', "Should return correct label"
            assert abs(result['score'] - 0.95) <= FLOAT_TOLERANCE, f"Score should be within {FLOAT_TOLERANCE} of 0.95"
            daemon.classifier.assert_called_once_with("Texto de teste.")

        def test_classify_text_invalid_class(self, daemon):
            """Test classification with invalid class"""
            daemon.classifier = MagicMock()
            daemon.classifier.return_value = [{'label': 'classe3', 'score': 0.95}]
            result = daemon.classify_text("Texto de teste.", ["classe1", "classe2"])
            assert result is None, "Should return None for invalid class"

        def test_classify_text_empty_classes(self, daemon):
            """Test classification with empty classes list"""
            daemon.classifier = MagicMock()
            result = daemon.classify_text("Texto de teste.", [])
            assert result is None, "Should return None for empty classes list"

        def test_classify_text_classifier_error(self, daemon):
            """Test classification with classifier error"""
            daemon.classifier = MagicMock()
            daemon.classifier.side_effect = Exception("Classifier error")
            result = daemon.classify_text("Texto de teste.", ["classe1"])
            assert result is None, "Should return None on classifier error"

        def test_classify_text_empty_input(self, daemon):
            """Test classification with empty input"""
            daemon.classifier = MagicMock()
            result = daemon.classify_text("", ["classe1"])
            assert result is None, "Should return None for empty input"

    class TestGRPCServer:
        def test_start_grpc_server_success(self, daemon, mock_grpc_server):
            """Test successful gRPC server start"""
            with patch('talia_daemon.talia_pb2_grpc.add_TaliaServiceServicer_to_server') as mock_add_servicer:
                daemon.tokenizer = MagicMock()
                daemon.model = MagicMock()
                daemon.classifier = MagicMock()
                daemon.start_grpc_server()
                mock_grpc_server.assert_called_once()
                mock_add_servicer.assert_called_once()
                mock_grpc_server.return_value.add_insecure_port.assert_called_once_with(f'unix://{talia_daemon.SOCKET_PATH}')
                mock_grpc_server.return_value.start.assert_called_once()

        def test_start_grpc_server_existing_socket(self, daemon, mock_grpc_server):
            """Test gRPC server start with existing socket"""
            with patch('os.path.exists', return_value=True), \
                 patch('os.remove') as mock_remove, \
                 patch('talia_daemon.talia_pb2_grpc.add_TaliaServiceServicer_to_server'):
                daemon.tokenizer = MagicMock()
                daemon.model = MagicMock()
                daemon.classifier = MagicMock()
                daemon.start_grpc_server()
                mock_remove.assert_called_once_with(talia_daemon.SOCKET_PATH)

        def test_start_grpc_server_socket_error(self, daemon, mock_grpc_server):
            """Test gRPC server start with socket error"""
            with patch('os.path.exists', return_value=True), \
                 patch('os.remove', side_effect=OSError("Permission denied")), \
                 patch('talia_daemon.talia_pb2_grpc.add_TaliaServiceServicer_to_server'):
                daemon.tokenizer = MagicMock()
                daemon.model = MagicMock()
                daemon.classifier = MagicMock()
                with pytest.raises(SystemExit) as exc_info:
                    daemon.start_grpc_server()
                assert exc_info.value.code == 1, "Should exit with code 1 on socket error"

        def test_stop_grpc_server(self, daemon, mock_grpc_server):
            """Test successful gRPC server stop"""
            daemon.grpc_server = MagicMock()
            daemon.stop_grpc_server()
            daemon.grpc_server.stop.assert_called_once_with(0)

        def test_stop_grpc_server_not_running(self, daemon):
            """Test stopping non-running gRPC server"""
            daemon.grpc_server = None
            daemon.stop_grpc_server()  # Should not raise any exception

    class TestEnvironment:
        def test_socket_path_env(self):
            """Test socket path from environment variable"""
            with patch.dict(os.environ, {"SOCKET_PATH": "/tmp/test.sock"}):
                import importlib
                importlib.reload(talia_daemon)
                assert talia_daemon.SOCKET_PATH == "/tmp/test.sock", "Should use socket path from environment"

        def test_socket_path_default(self):
            """Test default socket path"""
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                importlib.reload(talia_daemon)
                assert talia_daemon.SOCKET_PATH == "/tmp/daemon.sock", "Should use default socket path"

    class TestServiceServicer:
        def test_summarize_success(self, daemon):
            """Test successful summarization via gRPC"""
            servicer = talia_daemon.TaliaServiceServicer(daemon)
            daemon.summarize_text = MagicMock(return_value="Resumo teste")
            request = talia_pb2.SummarizeRequest(text="Texto teste")
            response = servicer.Summarize(request, MagicMock())
            assert response.success, "Should indicate success"
            assert response.summary == "Resumo teste", "Should return correct summary"
            assert not response.error_message, "Should not have error message"

        def test_summarize_failure(self, daemon):
            """Test failed summarization via gRPC"""
            servicer = talia_daemon.TaliaServiceServicer(daemon)
            daemon.summarize_text = MagicMock(return_value=None)
            request = talia_pb2.SummarizeRequest(text="Texto teste")
            response = servicer.Summarize(request, MagicMock())
            assert not response.success, "Should indicate failure"
            assert not response.summary, "Should not have summary"
            assert response.error_message == "Summarization failed", "Should have correct error message"

        def test_classify_success(self, daemon):
            """Test successful classification via gRPC"""
            servicer = talia_daemon.TaliaServiceServicer(daemon)
            daemon.classify_text = MagicMock(return_value={'label': 'classe1', 'score': 0.95})
            request = talia_pb2.ClassifyRequest(
                text="Texto teste",
                possible_classes=["classe1", "classe2"]
            )
            response = servicer.Classify(request, MagicMock())
            assert response.success, "Should indicate success"
            assert response.predicted_class == "classe1", "Should return correct class"
            assert abs(response.confidence - 0.95) <= FLOAT_TOLERANCE, f"Confidence should be within {FLOAT_TOLERANCE} of 0.95"
            assert not response.error_message, "Should not have error message"

        def test_classify_failure(self, daemon):
            """Test failed classification via gRPC"""
            servicer = talia_daemon.TaliaServiceServicer(daemon)
            daemon.classify_text = MagicMock(return_value=None)
            request = talia_pb2.ClassifyRequest(
                text="Texto teste",
                possible_classes=["classe1", "classe2"]
            )
            response = servicer.Classify(request, MagicMock())
            assert not response.success, "Should indicate failure"
            assert not response.predicted_class, "Should not have predicted class"
            assert response.confidence == 0.0, "Should have zero confidence"
            assert response.error_message == "Classification failed", "Should have correct error message"

        def test_health_check(self, daemon):
            """Test health check via gRPC"""
            servicer = talia_daemon.TaliaServiceServicer(daemon)
            request = talia_pb2.HealthCheckRequest()
            response = servicer.HealthCheck(request, MagicMock())
            assert response.healthy, "Should indicate healthy"
            assert response.status == "ok", "Should have ok status"

    def test_stop_sets_running_false(self, daemon):
        """Test daemon stop sets running flag to false"""
        daemon.running = True
        daemon.stop()
        assert not daemon.running, "Running flag should be set to false"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 