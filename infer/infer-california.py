import json
import os

import requests


def main() -> None:
    url = os.getenv("INFER_URL", "http://localhost:8080/v2/models/mlflow-model/infer")
    payload = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 8],
                "datatype": "FP64",
                "data": [[8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]],
            }
        ]
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()