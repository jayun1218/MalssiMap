"use client";

import React, { useState, useRef } from "react";
import { UploadCloud, FileAudio, Loader2, MapPin, Mic, Square, X } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from "recharts";

type AnalysisResult = {
  transcript: string;
  confidence: number;
  result: {
    gyeongsang: number;
    jeolla: number;
    chungcheong: number;
    standard: number;
  };
};

const regionLabels: Record<string, string> = {
  gyeongsang: "경상도",
  jeolla: "전라도",
  chungcheong: "충청도",
  standard: "표준어",
};

const colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981"];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  // 파일 업로드 핸들러
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  // 마이크 녹음 핸들러
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        // webm 확장자로 저장 (Whisper는 ffmpeg를 통해 대부분의 오디오 포맷 지원)
        const audioFile = new File([audioBlob], "마이크_녹음.webm", { type: "audio/webm" });
        setFile(audioFile);
        stream.getTracks().forEach(track => track.stop()); // 마이크 사용 종료
      };

      mediaRecorder.start();
      setIsRecording(true);
      setFile(null); // 기존 파일 초기화
    } catch (err) {
      console.error(err);
      alert("마이크 접근 권한을 허용해주세요.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // 파일 취소 핸들러
  const handleClearFile = () => {
    setFile(null);
    setResult(null);
  };

  // 분석 API 호출
  const handleAnalyze = async () => {
    if (!file) return;

    setIsLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("서버 응답 오류");
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  // 차트 데이터 가공
  const chartData = result
    ? Object.entries(result.result).map(([key, value]) => ({
        name: regionLabels[key],
        value: Math.round(value * 100),
      }))
    : [];

  return (
    <main className="min-h-screen bg-slate-950 text-white font-sans selection:bg-blue-500/30">
      <div className="max-w-4xl mx-auto px-6 py-12">
        {/* Header Section */}
        <header className="text-center mb-16 space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-blue-500/10 rounded-2xl mb-4 shadow-[0_0_30px_rgba(59,130,246,0.3)]">
            <MapPin className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-400">
            말씨 지도 AI
          </h1>
          <p className="text-slate-400 text-lg max-w-xl mx-auto leading-relaxed">
            당신의 목소리에 담긴 지역적 특징을 분석합니다. 파일을 업로드하거나 마이크로 직접 말해보세요.
          </p>
        </header>

        <div className="space-y-8">
          {/* Input Section (Grid) */}
          {!file && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {/* File Upload Zone */}
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
                className="
                  group cursor-pointer overflow-hidden
                  border-2 border-dashed border-slate-800 bg-slate-900/50 hover:bg-slate-800/50 hover:border-slate-600
                  rounded-3xl p-10 transition-all duration-300 ease-in-out
                  flex flex-col items-center justify-center text-center space-y-4 h-64
                "
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="audio/*"
                  className="hidden"
                />
                <div className="p-4 bg-slate-800 rounded-full group-hover:scale-110 group-hover:bg-blue-500/20 transition-all duration-300">
                  <UploadCloud className="w-8 h-8 text-slate-400 group-hover:text-blue-400 transition-colors" />
                </div>
                <div>
                  <p className="text-lg font-semibold text-white mb-1">
                    클릭 또는 드래그하여 파일 업로드
                  </p>
                  <p className="text-slate-500 text-sm">
                    WAV, MP3, M4A 등
                  </p>
                </div>
              </div>

              {/* Mic Record Zone */}
              <div
                onClick={isRecording ? stopRecording : startRecording}
                className={`
                  cursor-pointer overflow-hidden
                  border-2 border-dashed rounded-3xl p-10 transition-all duration-300 ease-in-out
                  flex flex-col items-center justify-center text-center space-y-4 h-64
                  ${isRecording 
                    ? "border-red-500/50 bg-red-500/10 shadow-[0_0_30px_rgba(239,68,68,0.2)]" 
                    : "border-slate-800 bg-slate-900/50 hover:bg-slate-800/50 hover:border-slate-600"
                  }
                `}
              >
                <div className={`
                  p-4 rounded-full transition-all duration-300
                  ${isRecording 
                    ? "bg-red-500 animate-pulse scale-110" 
                    : "bg-slate-800 hover:scale-110 hover:bg-purple-500/20"
                  }
                `}>
                  {isRecording ? (
                    <Square className="w-8 h-8 text-white fill-white" />
                  ) : (
                    <Mic className="w-8 h-8 text-slate-400 hover:text-purple-400 transition-colors" />
                  )}
                </div>
                <div>
                  <p className={`text-lg font-semibold mb-1 ${isRecording ? "text-red-400" : "text-white"}`}>
                    {isRecording ? "녹음 중... 클릭하여 완료" : "마이크로 직접 녹음하기"}
                  </p>
                  <p className="text-slate-500 text-sm">
                    {isRecording ? "목소리를 선명하게 들려주세요" : "클릭하면 녹음이 시작됩니다"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Selected File State */}
          {file && (
            <div className="animate-in zoom-in-95 duration-300 flex flex-col items-center justify-center bg-slate-900/80 border border-blue-500/30 rounded-3xl p-8 relative shadow-[0_0_40px_rgba(59,130,246,0.15)]">
              <button 
                onClick={handleClearFile}
                className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors"
                title="다른 파일 선택"
              >
                <X className="w-5 h-5" />
              </button>
              
              <div className="p-5 bg-blue-500/20 rounded-full mb-4">
                <FileAudio className="w-12 h-12 text-blue-400" />
              </div>
              <p className="text-2xl font-bold text-white mb-2">{file.name}</p>
              <p className="text-slate-400 mb-8">
                {(file.size / (1024 * 1024)).toFixed(2)} MB • 오디오 준비 완료
              </p>
              
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className={`
                  px-10 py-4 rounded-full font-bold text-lg
                  transition-all duration-300 ease-out
                  flex items-center space-x-3
                  ${isLoading
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-500 to-indigo-600 text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.6)] hover:scale-105"
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>AI 모델이 방언을 분석하고 있습니다...</span>
                  </>
                ) : (
                  <>
                    <MapPin className="w-6 h-6" />
                    <span>이 목소리 분석하기</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="mt-16 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <h2 className="text-2xl font-bold mb-6 flex items-center space-x-3">
              <span className="w-2 h-8 bg-gradient-to-b from-blue-400 to-purple-500 rounded-full"></span>
              <span>분석 결과 리포트</span>
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Transcript Card */}
              <div className="bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-blue-500/10 transition-colors"></div>
                <h3 className="text-slate-400 font-medium mb-6 text-sm uppercase tracking-widest flex items-center">
                  <Mic className="w-4 h-4 mr-2" />
                  인식된 문장 (STT)
                </h3>
                <p className="text-3xl font-medium leading-relaxed text-white">
                  "{result.transcript}"
                </p>
              </div>

              {/* Chart Card */}
              <div className="bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-purple-500/10 transition-colors"></div>
                <h3 className="text-slate-400 font-medium mb-6 text-sm uppercase tracking-widest flex items-center">
                  <MapPin className="w-4 h-4 mr-2" />
                  방언 유사도
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                      <XAxis type="number" hide />
                      <YAxis 
                        dataKey="name" 
                        type="category" 
                        axisLine={false} 
                        tickLine={false} 
                        tick={{ fill: '#94a3b8', fontSize: 14, fontWeight: 500 }}
                      />
                      <Tooltip 
                        cursor={{ fill: 'rgba(255,255,255,0.02)' }}
                        contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', backdropFilter: 'blur(8px)', borderColor: '#1e293b', borderRadius: '16px', color: '#fff' }}
                        itemStyle={{ color: '#fff', fontWeight: 600 }}
                        formatter={(value: number) => [`${value}%`, '일치율']}
                      />
                      <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={28}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
