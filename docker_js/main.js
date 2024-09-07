import express from 'express';
import axios from 'axios';  // HTTP 요청을 보내기 위한 라이브러리
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

// CSV 데이터 로드
const data = loadCSV('../data/train.csv');

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

const trainData = data.map(item => item.text);
const maxVocabSize = 10000;

// 어휘 사전 생성
const vocabulary = createVocabulary(trainData, maxVocabSize);
const maxSequenceLength = 23;  // 벡터화 시 사용할 최대 길이

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
    const inputTensor = [inputVector];  // TensorFlow Serving에 보낼 수 있는 형식으로 배열 형태로 준비

    // TensorFlow Serving으로 POST 요청 보내기
    const response = await axios.post('http://localhost:8501/v1/models/lstm_model:predict', {
      instances: inputTensor  // 입력 데이터를 instances 배열에 담아 요청
    });

    const prediction = response.data.predictions[0][0];  // 예측 결과 추출

    res.json({
      text,
      prediction: prediction > 0.5 ? 'Disaster' : 'Normal',
      confidence: prediction
    });

    const endTime = performance.now();
    const executionTimeMs = endTime - startTime;
    const executionTimeSec = executionTimeMs / 1000;

    const endMemoryUsage = process.memoryUsage().rss;
    const memoryUsageMb = endMemoryUsage / (1024 * 1024);

    console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
    console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);
  } catch (err) {
    console.error('예측 중 오류 발생:', err.message);
    res.status(500).send('예측 중 오류가 발생했습니다.');
  }
});

// 서버 시작
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`서버가 ${PORT} 포트에서 실행 중입니다.`);
});
