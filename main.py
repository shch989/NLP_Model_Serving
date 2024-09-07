from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 데이터 로드 (훈련 데이터가 필요)
train_df = pd.read_csv("./data/train.csv")

# Shuffle training dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)  # 데이터를 섞음

# 훈련 및 검증 데이터 분할
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    train_df_shuffled["text"].to_numpy(),
    train_df_shuffled["target"].to_numpy(),
    test_size=0.1,  # 10%는 검증 데이터로 사용
    random_state=42  # 재현성을 위한 랜덤 시드
)

# 모델 로드
max_vocab_length = 10000
max_length = 15

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

# 훈련 텍스트에 벡터화 레이어 적응시키기
text_vectorizer.adapt(train_sentences)

# 모델 로드
model_2 = tf.keras.models.load_model("./lstm/20240906_tf.h5", custom_objects={"TextVectorization": text_vectorizer})

# 모델에서 올바른 이름의 TextVectorization 레이어 가져오기
text_vectorizer = model_2.get_layer("text_vectorization_1")
text_vectorizer.adapt(train_sentences)  # 훈련 데이터를 사용해 레이어 초기화

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    try:
        # 입력 텍스트를 pandas Series로 변환
        test_series = pd.Series([input.text])

        # pandas Series를 Tensor로 변환
        test_tensor = tf.constant(test_series)

        # 예측 수행
        model_2_pred_probs = model_2.predict(test_tensor)
        model_2_preds = tf.squeeze(tf.round(model_2_pred_probs[0][0]))

        value = model_2_preds.numpy()
        
        return {"input_text": input.text, "predicted_probability": float(model_2_pred_probs[0][0]), "result": int(value)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))