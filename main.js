import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs'; 
import { parse } from 'csv-parse/sync';  // CSV 파일을 처리하기 위한 모듈

const app = express();
app.use(express.json());

// CSV 파일 로드 함수
const loadCSV = (filePath) => {
    const data = fs.readFileSync(filePath);  // 파일 읽기
    const records = parse(data, { columns: true });  // CSV 데이터 파싱
    return records;
};

// 모델 로드 함수
let model;
const loadModel = async () => {
  model = await tf.loadLayersModel('file://./saved_model_lstm/model.json');  // 모델 로드
  console.log('모델이 로드되었습니다.');
};

// CSV 데이터 로드
const data = loadCSV('./data/train.csv');

// 어휘 사전 만들기
const createVocabulary = (data, vocabSize) => {
    const wordCounts = {};
    data.forEach(sentence => {
      const words = sentence.toLowerCase().split(' ');
      words.forEach(word => {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      });
    });
  
    const sortedWords = Object.keys(wordCounts).sort((a, b) => wordCounts[b] - wordCounts[a]);
    return sortedWords.slice(0, vocabSize).reduce((vocab, word, index) => {
      vocab[word] = index + 1;
      return vocab;
    }, {});
};

const trainData = data.map(item => item.text);
const maxVocabSize = 10000;

// 어휘 사전 생성
const vocabulary = createVocabulary(trainData, maxVocabSize);

const maxSequenceLength = 15;  // 벡터화 시 사용할 최대 길이

// 텍스트 벡터화 함수
const vectorizeText = (text, vocab, maxLength) => {
  const words = text.toLowerCase().split(' ');
  let vector = words.map(word => vocab[word] && vocab[word] < 10000 ? vocab[word] : 0);  // 사전에 없는 단어는 0으로 처리

  if (vector.length > maxLength) {
    vector = vector.slice(0, maxLength);
  } else {
    while (vector.length < maxLength) {
      vector.push(0);  // 패딩 추가
    }
  }
  return vector;
};

// 예측 API 엔드포인트
app.post('/predict', async (req, res) => {
    const startTime = performance.now(); 
    const { text } = req.body;

    if (!text) {
        return res.status(400).send('텍스트 입력이 필요합니다.');
    }

    try {
        // 입력 텍스트를 벡터화
        const inputVector = vectorizeText(text, vocabulary, maxSequenceLength);
        const inputTensor = tf.tensor2d([inputVector], [1, maxSequenceLength]);  // 입력 텐서 생성

        // 예측 수행
        const prediction = model.predict(inputTensor);
        const result = prediction.dataSync()[0];  // 결과 값 추출

        res.json({
        text,
        prediction: result > 0.5 ? 'Disaster' : 'Normal',
        confidence: result
        });

        const endTime = performance.now();
        const executionTimeMs = endTime - startTime;
        const executionTimeSec = executionTimeMs / 1000;

        const endMemoryUsage = process.memoryUsage().rss;
        const memoryUsageMb = endMemoryUsage / (1024 * 1024)

        console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
        console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);
    } catch (err) {
        res.status(500).send('예측 중 오류가 발생했습니다.');
    }
});

// 서버 시작 및 모델 로드
const PORT = 3000;
app.listen(PORT, async () => {
  await loadModel();  // 서버 시작 시 모델 로드
  console.log(`서버가 ${PORT} 포트에서 실행 중입니다.`);
});