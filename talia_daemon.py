#!/usr/bin/env python3
import daemon as python_daemon
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import grpc
import os
import signal
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import talia_pb2
import talia_pb2_grpc
from dotenv import load_dotenv

# Load environment variables from .env or talia_daemon.env
load_dotenv(".env")
load_dotenv("talia_daemon.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('talia_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('talia_daemon')

SOCKET_PATH = os.getenv('SOCKET_PATH', '/tmp/daemon.sock')

class TaliaServiceServicer(talia_pb2_grpc.TaliaServiceServicer):
    def __init__(self, daemon):
        self.daemon = daemon

    def Summarize(self, request, context):
        text = request.text
        logger.info(f"gRPC Summarize request: {text}")
        try:
            summary = self.daemon.summarize_text(text)
            return talia_pb2.SummarizeResponse(summary=summary or '', success=summary is not None, error_message='' if summary else 'Summarization failed')
        except Exception as e:
            logger.error(f"gRPC Summarize error: {str(e)}")
            return talia_pb2.SummarizeResponse(summary='', success=False, error_message=str(e))

    def Classify(self, request, context):
        text = request.text
        possible_classes = request.possible_classes
        logger.info(f"gRPC Classify request: {text} with classes {possible_classes}")
        try:
            result = self.daemon.classify_text(text, possible_classes)
            if result:
                return talia_pb2.ClassifyResponse(
                    predicted_class=result['label'],
                    confidence=result['score'],
                    success=True,
                    error_message=''
                )
            return talia_pb2.ClassifyResponse(
                predicted_class='',
                confidence=0.0,
                success=False,
                error_message='Classification failed'
            )
        except Exception as e:
            logger.error(f"gRPC Classify error: {str(e)}")
            return talia_pb2.ClassifyResponse(
                predicted_class='',
                confidence=0.0,
                success=False,
                error_message=str(e)
            )

    def HealthCheck(self, request, context):
        logger.info("gRPC HealthCheck request")
        return talia_pb2.HealthCheckResponse(healthy=True, status="ok")

class TaliaDaemon:
    def __init__(self):
        self.running = True
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.grpc_server = None
        
    def load_model(self):
        try:
            logger.info("Loading Gaia model (CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it)...")
            self.tokenizer = AutoTokenizer.from_pretrained('CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it')
            self.model = AutoModelForCausalLM.from_pretrained('CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it')
            logger.info("Loading DistilBERT classifier...")
            self.classifier = pipeline("text-classification", model="adalbertojunior/distilbert-portuguese-cased")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            sys.exit(1)

    def summarize_text(self, texto):
        try:
            prompt = "Resuma o seguinte texto em uma frase:"
            inputs = self.tokenizer(prompt + "\n" + texto, return_tensors='pt')
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            resumo = self.tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Resumo para '{texto}': {resumo}")
            return resumo
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return None

    def classify_text(self, text, possible_classes):
        try:
            # First get the raw classification
            result = self.classifier(text)[0]
            
            # If the predicted label is not in possible_classes, return None
            if result['label'] not in possible_classes:
                logger.warning(f"Predicted class {result['label']} not in possible classes {possible_classes}")
                return None
                
            logger.info(f"Classification for '{text}': {result}")
            return result
        except Exception as e:
            logger.error(f"Error classifying text: {str(e)}")
            return None

    def start_grpc_server(self):
        # Remove old socket if exists
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        self.grpc_server = grpc.server(ThreadPoolExecutor())
        talia_pb2_grpc.add_TaliaServiceServicer_to_server(TaliaServiceServicer(self), self.grpc_server)
        self.grpc_server.add_insecure_port(f'unix://{SOCKET_PATH}')
        self.grpc_server.start()
        logger.info(f"gRPC server started on unix://{SOCKET_PATH}")

    def stop_grpc_server(self):
        if self.grpc_server:
            self.grpc_server.stop(0)
            logger.info("gRPC server stopped")
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

    def run(self):
        self.load_model()
        self.start_grpc_server()
        logger.info("Talia Daemon started (Gaia summarization and DistilBERT classification)")
        try:
            while self.running:
                time.sleep(1)
        finally:
            self.stop_grpc_server()

    def stop(self):
        logger.info("Stopping Talia Daemon...")
        self.running = False

def main():
    daemon_instance = TaliaDaemon()
    
    # Handle signals
    def signal_handler(signum, frame):
        daemon_instance.stop()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run as daemon
    with python_daemon.DaemonContext():
        daemon_instance.run()

if __name__ == "__main__":
    main() 