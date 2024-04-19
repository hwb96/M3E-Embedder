import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import requests
import json
from fastapi import FastAPI, Depends, HTTPException
from starlette.requests import Request
import time
from starlette.responses import Response
import uvicorn
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# 跨域问题
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SenMatch:
    def __init__(self) -> None:
        # pass
        self.model = SentenceTransformer("./m3e-base", device="cuda")

    def sen_embedding(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings

    def sen_one_embedding(self, sentence):
        embedding = self.model.encode([sentence])
        return embedding


sm = SenMatch()


@app.post("/m3e/embedding")
async def route_order(req: Request):
    if req.headers["Content-Type"] == "application/json":
        item = quertItem(**await req.json())
        torch.cuda.empty_cache()
        data = sm.sen_embedding(item.query)
        emdata = {"embedding": data.tolist()}

    return Response(content=json.dumps(emdata))


if __name__ == "__main__":
    uvicorn.run(app="embedding:app", host="0.0.0.0", port=10201, workers=1)
