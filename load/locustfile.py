"""Load test: mostly single predictions, some batches, an occasional health check."""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLES = json.loads((Path(__file__).parent / "samples.json").read_text())


class ApiUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def predict(self):
        self.client.post("/predict", json=random.choice(SAMPLES), name="/predict")

    @task(1)
    def predict_batch(self):
        records = random.sample(SAMPLES, 50)
        self.client.post("/predict/batch", json={"records": records}, name="/predict/batch")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
