import sys
from openai import OpenAI
import base64
import cv2
import time

# QUESTION = "What is the content of each image?"
QUESTION = "描述每个视频上的内容是什么?"
# Modify OpenAI's API key and API base to use 9n-transformer's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:1800/v1"
urls = [openai_api_base]
count = 1

import requests
import json

def get_model_id():
    try:
        # 发送 HTTP 请求
        response = requests.get(f"{openai_api_base}/models")
        response.raise_for_status()  # 检查请求是否成功

        # 解析 JSON 数据
        data = response.json()

        # 提取第一个模型的 ID
        model_id = data['data'][0]['id']
        print(f"find model: {model_id}")
        return model_id

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"JSON 解析错误: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e}")
        return None

for openai_api_base in urls:
    print(f"post {openai_api_base}")

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
        max_retries = 0,
        timeout = 60*100,
    )


    # Function to encode the video
    def encode_video(file_path):
        if "jfs" in file_path:
            img = requests.get("http://img10.360buyimg.local/da/" + file_path, timeout=5).content
            encoded_string = base64.b64encode(img)
        else:
            with open(file_path, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read())
        return encoded_string.decode("utf-8")

    # webm格式不支持，报错
    video_path = '/workspace/volume/gxt-1/source_xllm/xllm/demo.mp4'

    # Getting the base64 string
    base64_video = encode_video(video_path)

    # text = '简单介绍下图片内容.'
    text = QUESTION
    message = {
        "role": "user",
        "content": [
             {
                "type": "video_url",
                 "video_url": {
                     "url": f"data:video/mp4;base64,{base64_video}"
                },
           },
            {
                "type": "text",
                "text": text
            },

        ],
    }


    stream=False
    for idx in range(count):
        st = time.time()
        chat_completion = client.chat.completions.create(
            messages=[message],
            model=get_model_id(),
            stream=stream,
            max_tokens=256,
            temperature=0.0
        )
        print(f" cost is {time.time()-st}")

        print(f"Chat completion results: {chat_completion}")
        if stream:
            for chunk in chat_completion:
                print(chunk)
        else:
            print("-" * 50)
            generated_text = chat_completion.choices[0].message.content
            print(generated_text)
            print("-" * 50)
