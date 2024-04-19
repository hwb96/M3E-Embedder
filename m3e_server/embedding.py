import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import requests
import json
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
import time
from starlette.responses import Response
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED  # HTTP状态码
import numpy as np  # 用于数值计算的库

from typing import List
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()


class quertItem(BaseModel):
    query: List[str] = None


# class onequertItem(BaseModel):
#    input: str
# 定义嵌入请求的数据模型
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


# 定义嵌入响应的数据模型
class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


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

        response_start_time = time.time()
        torch.cuda.empty_cache()
        data = sm.sen_embedding(item.query)
        emdata = {"embedding": data.tolist()}
        response_end_time = time.time()

    return Response(content=json.dumps(emdata))


@app.post("/v1/embeddings111")
async def route_order(req: Request):
    if req.headers["Content-Type"] == "application/json":

        item = req.json()
        # for i in item:
        #    print(i)
        sentences = item.get("sentences", [])

        print(sentences)
        print(333, item["input"])

        # print(item,type(item),type(item.query))
        response_start_time = time.time()
        # rjson = json.loads(item.json(), strict=False, encoding="utf-8")
        # print(f"传入参数：{str(rjson)}")
        torch.cuda.empty_cache()
        data = sm.sen_one_embedding(item.input)
        # print(111111,type(data))
        emdata = {"data": [{"embedding": data.tolist()}]}
        response_end_time = time.time()
        # print("result：", res)
        # print(f"模型响应时间：{str(response_end_time - response_start_time)}")

    return Response(content=json.dumps(emdata))


# 验证token的函数
async def verify_token(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header:
        token_type, _, token = auth_header.partition(" ")
        if token_type.lower() == "bearer" and token == "sk-xx":  # 这里配置你的token
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest, token: bool = Depends(verify_token)
):
    # 计算嵌入向量和tokens数量
    # embeddings = [embeddings_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    # embeddings = [
    #     expand_features(embedding, 1536) if len(embedding) < 1536 else embedding
    #     for embedding in embeddings
    # ]
    embeddings = sm.sen_embedding(request.input)
    # embeddings={"embedding":data.tolist()}
    # embeddings=data.tolist()
    # Min-Max normalization 归一化
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    # total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": 100,
        },
    }

    return response


if __name__ == "__main__":
    # print(sm.sen_embedding(["你好"]))
    uvicorn.run(app="embedding:app", host="0.0.0.0", port=10201, workers=1)
