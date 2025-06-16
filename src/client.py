import grpc
import talia_pb2
import talia_pb2_grpc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")
load_dotenv("talia_daemon.env")

SOCKET_PATH = os.getenv('SOCKET_PATH', '/tmp/daemon.sock')

def main():
    # Connect to the gRPC server over UDS
    channel = grpc.insecure_channel(f'unix://{SOCKET_PATH}')
    stub = talia_pb2_grpc.TaliaServiceStub(channel)

    # Health check
    print("\n=== Health Check ===")
    health_response = stub.HealthCheck(talia_pb2.HealthCheckRequest())
    print(f"Health: {health_response.healthy}, Status: {health_response.status}")

    # Summarize
    print("\n=== Summarization ===")
    text = "Reunião com cliente amanhã às 14h para discutir o novo projeto de desenvolvimento de software."
    response = stub.Summarize(talia_pb2.SummarizeRequest(text=text))
    print(f"Success: {response.success}")
    print(f"Summary: {response.summary}")
    if not response.success:
        print(f"Error: {response.error_message}")

    # Classify
    print("\n=== Classification ===")
    text = "Reunião com cliente amanhã"
    possible_classes = ["reunião", "tarefa", "lembrete", "outro"]
    response = stub.Classify(talia_pb2.ClassifyRequest(
        text=text,
        possible_classes=possible_classes
    ))
    print(f"Success: {response.success}")
    if response.success:
        print(f"Predicted class: {response.predicted_class}")
        print(f"Confidence: {response.confidence:.2f}")
    else:
        print(f"Error: {response.error_message}")

if __name__ == "__main__":
    main() 