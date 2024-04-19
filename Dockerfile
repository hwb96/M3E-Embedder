# 使用当前的基础镜像
FROM leekltw/cuda11.3.1-runtime-ubuntu20.04-python3.10.8

# 设置环境变量，比如时区为北京/上海时间等
ENV TZ=Asia/Shanghai

# 设置镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

# 复制当前目录下的requirements.txt文件到工作目录
COPY ./requirements.txt /app/

# 使用pip安装requirements.txt中的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制当前目录下的所有文件到工作目录
COPY ./m3e_server /app/

# 暴露端口，这里假设你的应用使用10201-10203端口
EXPOSE 10201-10203

# 设置容器启动时执行的命令
# CMD ["nohup", "python", "-u", "embedding.py", ">", "/app/embedding.log", "2>&1"]
CMD ["python", "-u", "embedding.py"]