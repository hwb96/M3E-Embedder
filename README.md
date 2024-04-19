# Docker 容器部署指南 - m3e-embedding-server

虽然在[MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard)嵌入模型榜单上，m3e模型的排名不如2023年那么靠前，但是在我们文旅领域大模型+本地知识库的项目里，我们实验了bge,gte等诸多模型，其他的模型虽然分更高，但是相关性在我们的实验中召回情况不甚理想，而m3e依然是表现最好的那个，您可能需要在您的知识库进行验证。虽然本文档以m3e为主，对于其他模型，代码依然具有通用性。

本文档将指导您如何使用 Docker 构建和部署 m3e-embedding-server 应用。

## 前提条件

- 确保您的系统已经安装了 Docker。
- 确保您的 Docker 版本支持 GPU 并已正确配置。
- 确保下载m3e嵌入模型[moka-ai/m3e-base · Hugging Face](https://huggingface.co/moka-ai/m3e-base)到`m3e-base`目录下。

## 步骤 1: 构建 Docker 镜像

1. 打开终端或命令提示符。

2. 切换到包含 Dockerfile 的目录。

3. 执行以下命令构建 Docker 镜像：

   ```
   docker build -t m3e-embedding-server:latest .
   ```

   这里的`m3e-embedding-server`是镜像的名称，`latest`是标签，`.`表示当前目录。

## 步骤 2: 查看构建的 Docker 镜像

1. 使用以下命令查看已构建的 Docker 镜像列表：

   ```
   docker images
   ```

2. 确认您的 `m3e-embedding-server` 镜像已成功列出。

## 步骤 3: 运行 Docker 容器

1. 使用以下命令启动并运行 Docker 容器：

   ```
   docker run --gpus all -itd -p 10175:10201 --dns=8.8.8.8 --name your-image-name m3e-embedding-server:latest
   ```

   将 `your-image-name` 替换为您想要给容器命名的名称。

2. 确保将宿主机的端口 `10175` 映射到容器的端口 `10201`，并`10175`替换为你想要的端口。

## 步骤 4: 查看容器日志

1. 如果需要查看容器的日志，可以使用以下命令：

   ```
   docker logs --name your-image-name
   ```

   将 `your-image-name`替换为您为容器指定的名称。

## 步骤 5: 测试服务

1. 确认容器正在运行后，您可以使用以下请求测试服务：

   ```
   POST http://ip:port/m3e/embedding
   {
     "query": ["你的名字是什么"]
   }
   ```

   您应该收到一个包含 768 维向量的 JSON 响应。

## 注意事项

- 确保 Dockerfile 中的 `requirements.txt` 文件和 `m3e_server` 目录的路径正确无误。
- Dockerfile里的`leekltw/cuda11.3.1-runtime-ubuntu20.04-python3.10.8`是我在DockerHub上找的非官方镜像，如有需要，访问[nvidia/cuda - Docker Image | Docker Hub](https://hub.docker.com/r/nvidia/cuda)下载英伟达官方镜像。

