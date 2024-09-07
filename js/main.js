const express = require('express');
const axios = require('axios');

const app = express();
const port = 3000;

app.use(express.json());

// 로컬 서버에서 외부 API로 POST 요청을 보내는 엔드포인트
app.post('/text', async (req, res) => {
    const startTime = performance.now(); 
    const { text } = req.body;

    if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Invalid input: text must be a string' });
    }

    try {
        // 외부 API로 POST 요청을 보냄
        const response = await axios.post('http://localhost:8000/predict', { text });

        const endTime = performance.now();
        const executionTimeMs = endTime - startTime;
        const executionTimeSec = executionTimeMs / 1000;

        const endMemoryUsage = process.memoryUsage().rss;
        const memoryUsageMb = endMemoryUsage / (1024 * 1024)

        console.log(`Execution time: ${executionTimeSec.toFixed(4)} seconds`);
        console.log(`Memory usage: ${memoryUsageMb.toFixed(2)} MB`);

        // 외부 API로부터 받은 응답을 클라이언트에 전달
        res.json(response.data);
    } catch (error) {
        console.error('Error sending POST request:', error);
        res.status(500).json({ error: 'Failed to send POST request' });
    }
});

// 서버 시작
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
