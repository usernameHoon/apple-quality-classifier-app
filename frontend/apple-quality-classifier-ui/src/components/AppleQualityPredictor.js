import React, { useState } from 'react';

const gradeInfo = {
  '특': { size: '큼', color: '높음', defect: '없음', gloss: '우수' },
  '보통': { size: '중간', color: '낮음', defect: '있음', gloss: '중간' },
  '보통 이하': { size: '작음', color: '낮음', defect: '있음', gloss: '낮음' }
};

const AppleQualityPredictor = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('예측 요청 실패:', error);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-8 bg-white rounded-xl shadow-md mt-10">
      <h2 className="text-2xl font-bold text-center mb-6">🍏 사과 품질 테스트</h2>

      <input type="file" accept="image/*" onChange={handleFileChange} className="mb-6 text-base" />

      {previewUrl && (
        <img
          src={previewUrl}
          alt="preview"
          className="w-80 h-80 object-cover mx-auto mb-6 rounded border"
        />
      )}

      <button
        onClick={handleSubmit}
        className="w-full bg-blue-500 text-white py-3 px-4 text-lg rounded hover:bg-blue-600"
      >
        예측하기
      </button>

      {result && (
        <div className="mt-8 text-center">
          <p className="text-xl font-semibold mb-2">
            품질 예측 결과: <span className="text-green-600 text-2xl">{result.label_korean}</span>
          </p>
          <p className="text-base text-gray-700 mb-6">
            신뢰도: {(result.confidence * 100).toFixed(2)}%
          </p>

          <table className="w-full border text-base">
            <thead>
              <tr className="bg-gray-100">
                <th className="border px-4 py-2">품질 등급</th>
                <th className="border px-4 py-2">크기</th>
                <th className="border px-4 py-2">색상 균일도</th>
                <th className="border px-4 py-2">표면 결점</th>
                <th className="border px-4 py-2">광택</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(gradeInfo).map(([grade, info]) => (
                <tr
                  key={grade}
                  className={result.label_korean === grade ? 'bg-yellow-100 font-semibold' : ''}
                >
                  <td className="border px-4 py-2">{grade}</td>
                  <td className="border px-4 py-2">{info.size}</td>
                  <td className="border px-4 py-2">{info.color}</td>
                  <td className="border px-4 py-2">{info.defect}</td>
                  <td className="border px-4 py-2">{info.gloss}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AppleQualityPredictor;
