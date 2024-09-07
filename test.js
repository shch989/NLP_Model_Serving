import * as tf from '@tensorflow/tfjs-node';  // TensorFlow.js 사용
import fs from 'fs'; 
import { parse } from 'csv-parse/sync';

// CSV 파일 로드 함수
const loadCSV = (filePath) => {
    const data = fs.readFileSync(filePath);  // 파일 읽기
    const records = parse(data, { columns: true });  // CSV 데이터 파싱
    return records;
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

// 텍스트 벡터화 함수 수정 (사전에 없는 단어는 0으로 처리, 인덱스가 10000 이상으로 나오지 않게)
const vectorizeText = (text, vocab, maxLength) => {
  const words = text.toLowerCase().split(' ');
  let vector = words.map(word => vocab[word] && vocab[word] < 10000 ? vocab[word] : 0);  // 사전에 없는 단어는 0으로 처리

  if (vector.length > maxLength) {
    vector = vector.slice(0, maxLength);  // 길이를 맞추기 위해 자름
  } else {
    while (vector.length < maxLength) {
      vector.push(0);  // 패딩 추가
    }
  }
  return vector;
};

// 데이터 준비
const trainData = data.map(item => item.text);
const trainLabels = data.map(item => parseInt(item.target));

// 파라미터 설정
const maxVocabSize = 10000;
const maxSequenceLength = 15;

// 어휘 사전 생성
const vocabulary = createVocabulary(trainData, maxVocabSize);

// 텍스트 벡터화
const trainSentences = trainData.map(sentence => vectorizeText(sentence, vocabulary, maxSequenceLength));
const xs = tf.tensor2d(trainSentences, [trainSentences.length, maxSequenceLength]);  // 벡터로 변환된 텍스트 데이터
const ys = tf.tensor2d(trainLabels, [trainLabels.length, 1]);  // 라벨 데이터

// 모델 구성
const model = tf.sequential();

// 임베딩 레이어
model.add(tf.layers.embedding({
  inputDim: maxVocabSize,
  outputDim: 128,  // 임베딩 차원 크기
  inputLength: maxSequenceLength
}));

// GRU 레이어 추가
model.add(tf.layers.gru({ units: 64 }));  // LSTM 대신 GRU 사용

// 출력 레이어 (이진 분류)
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// 모델 컴파일
model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});

// 모델 훈련 후 모델 저장
model.fit(xs, ys, {
    epochs: 5,  // 에포크 수
    batchSize: 32,
    validationSplit: 0.1  // 10% 데이터로 검증
  }).then(async (info) => {
    console.log('훈련 완료');
    console.log('최종 정확도:', info.history.acc);
    
    // 모델을 로컬 파일 시스템에 저장
    await model.save('file://./saved_model_gru');  // GRU 기반 모델을 저장할 경로 지정
    console.log('모델이 저장되었습니다.');
  });
  
// 모델 요약 출력
model.summary();