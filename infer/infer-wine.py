import json
import os

import requests


def main() -> None:
    url = os.getenv("INFER_URL", "http://localhost:8080/v2/models/mlflow-model/infer")
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 13],
                "datatype": "FP64",
                "data": [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]],
            }
        ]
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()