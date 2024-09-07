from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import requests
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
import time
import psutil
import os

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 메모리확인용
process = psutil.Process(os.getpid())

# 데이터 로드 (훈련 데이터가 필요)
train_df = pd.read_csv("../data/train.csv")

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)  # 데이터를 섞음

# 훈련 및 검증 데이터 분할
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1,  # 10%는 검증 데이터로 사용
    random_state=42  # 재현성을 위한 랜덤 시드
)

# 벡터화 레이어 설정
max_vocab_length = 10000
max_length = 23  # 모델이 기대하는 입력 시퀀스 길이로 수정

# 벡터화 레이어 새로 정의
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

# 훈련 텍스트에 벡터화 레이어 적응시키기
text_vectorizer.adapt(train_sentences)

# 텍스트를 텐서로 변환하는 함수
def convert_text_to_tensor(text):
    test_series = pd.Series([text])
    test_tensor = tf.constant(test_series)
    return test_tensor

# 텐서를 벡터화된 리스트로 변환하는 함수
def vector_to_list(tensor):
    input_vector = text_vectorizer(tensor)    
    input_vector_list = input_vector.numpy().tolist()
    return input_vector_list

class TextInput(BaseModel):
    text: str
 
@app.post("/predict")
async def predict(input: TextInput):

    # 시작 시간 기록
    start_time = time.time()

    try:
        input_text = input.text
        text_tensor = convert_text_to_tensor(input_text)
        list_vector = vector_to_list(text_tensor)

        # 벡터화된 시퀀스 길이가 max_length에 맞춰져 있는지 확인
        assert len(list_vector[0]) == max_length, "Input sequence length mismatch"

        ten = tf.constant(list_vector[0])
        ten = tf.expand_dims(ten, axis=0)

        print(ten)
        print(ten.shape)

        # 예측 API 호출 준비
        url = "http://localhost:8501/v1/models/lstm_model:predict"
        payload = {"instances": ten.numpy().tolist()}  # 이미 리스트이므로 별도 직렬화 불필요
        headers = {"Content-Type": "application/json"}

        # 요청 보내기
        response = requests.post(url, json=payload, headers=headers)

                # 종료 시간 및 리소스 사용량 기록
        end_time = time.time()  # 종료 시간 기록
        execution_time = end_time - start_time  # 실행 시간 계산

        # 메모리 사용량 계산
        memory_usage_bytes = process.memory_info().rss  # RSS(Resident Set Size)는 프로세스가 사용하는 물리 메모리의 양입니다.
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)  # 바이트를 메가바이트로 변환

        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Memory Usage: {memory_usage_mb:.2f} MB")

        # 응답 출력
        print(response.json())
        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))