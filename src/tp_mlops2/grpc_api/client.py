
import grpc

from tp_mlops2.auth import crear_token
from tp_mlops2.grpc_api import scoring_pb2, scoring_pb2_grpc

SAMPLE_REQUEST = {
    "timestamp": "2018-06-15T14:00:00",
    "temp_madrid": 28.0,
    "temp_barcelona": 25.0,
    "temp_valencia": 27.0,
    "temp_seville": 32.0,
    "temp_bilbao": 20.0,
    "load_24h_ago": 30000.0,
    "load_168h_ago": 29500.0,
}


def _auth_metadata():
    token = crear_token("alan", "analista")
    return (("authorization", f"Bearer {token}"),)


def predict_unary(stub):
    request = scoring_pb2.PredictRequest(**SAMPLE_REQUEST)
    response = stub.Predict(request, metadata=_auth_metadata())
    print(
        f"Predicción unary: {response.predicted_load_mw:.1f} MW "
        f"(modelo {response.model_version})"
    )



def predict_batch(stub, n=5):
    requests = [scoring_pb2.PredictRequest(**SAMPLE_REQUEST) for _ in range(n)]
    batch_request = scoring_pb2.BatchPredictRequest(requests=requests)

    print(f"Predicción batch (server-streaming, {n} items):")
    for response in stub.PredictBatch(batch_request, metadata=_auth_metadata()):
        print(f"  {response.predicted_load_mw:.1f} MW")


if __name__ == "__main__":
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = scoring_pb2_grpc.ScoringStub(channel)
        predict_unary(stub)
        predict_batch(stub)
