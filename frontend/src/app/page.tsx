"use client";

import React, { useState, useRef, useEffect } from "react";
import { UploadCloud, FileAudio, Loader2, MapPin, Mic, Square, X, History, Clock, Sparkles } from "lucide-react";
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

type HistoryItem = {
  id: number;
  transcript: string;
  gyeongsang: number;
  jeolla: number;
  chungcheong: number;
  standard: number;
  created_at: string;
};

const regionLabels: Record<string, string> = {
  gyeongsang: "경상도",
  jeolla: "전라도",
  chungcheong: "충청도",
  standard: "표준어",
};

const colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981"];

// 칭호 부여 로직
function getGamifiedTitle(result: AnalysisResult["result"]) {
  const scores = [
    { name: "경상도", score: result.gyeongsang },
    { name: "전라도", score: result.jeolla },
    { name: "충청도", score: result.chungcheong },
    { name: "표준어", score: result.standard }
  ];
  
  const top = scores.sort((a, b) => b.score - a.score)[0];
  const percent = top.score * 100;
  
  if (top.name === "경상도") {
    if (percent >= 80) return { title: "🔥 불타는 찐 갱상도 네이티브", color: "from-red-500 to-orange-500", glow: "shadow-[0_0_50px_rgba(239,68,68,0.4)]" };
    if (percent >= 50) return { title: "🌶️ 짭짤한 갱상도 바이브", color: "from-orange-400 to-amber-500", glow: "shadow-[0_0_40px_rgba(245,158,11,0.3)]" };
    return { title: "🌱 갱상도 향 1% 첨가", color: "from-amber-200 to-orange-300", glow: "shadow-[0_0_30px_rgba(252,211,77,0.2)]" };
  }
  if (top.name === "전라도") {
    if (percent >= 80) return { title: "😎 참말로 징한 전라도 네이티브", color: "from-purple-600 to-indigo-600", glow: "shadow-[0_0_50px_rgba(147,51,234,0.4)]" };
    if (percent >= 50) return { title: "🍠 구수한 전라도 바이브", color: "from-purple-400 to-indigo-500", glow: "shadow-[0_0_40px_rgba(168,85,247,0.3)]" };
    return { title: "🌱 전라도 향 1% 첨가", color: "from-purple-300 to-indigo-400", glow: "shadow-[0_0_30px_rgba(216,180,254,0.2)]" };
  }
  if (top.name === "충청도") {
    if (percent >= 80) return { title: "🪨 여유만만 충청도 양반", color: "from-emerald-500 to-teal-500", glow: "shadow-[0_0_50px_rgba(16,185,129,0.4)]" };
    if (percent >= 50) return { title: "🍵 은은한 충청도 바이브", color: "from-emerald-400 to-teal-400", glow: "shadow-[0_0_40px_rgba(52,211,153,0.3)]" };
    return { title: "🌱 충청도 향 1% 첨가", color: "from-emerald-200 to-teal-300", glow: "shadow-[0_0_30px_rgba(110,231,183,0.2)]" };
  }
  // 표준어
  if (percent >= 80) return { title: "🤖 서울깍쟁이 AI 아나운서", color: "from-blue-500 to-cyan-500", glow: "shadow-[0_0_50px_rgba(59,130,246,0.4)]" };
  if (percent >= 50) return { title: "🏙️ 매끄러운 서울 바이브", color: "from-blue-400 to-cyan-400", glow: "shadow-[0_0_40px_rgba(96,165,250,0.3)]" };
  return { title: "🌱 서울말 향 1% 첨가", color: "from-blue-200 to-cyan-300", glow: "shadow-[0_0_30px_rgba(191,219,254,0.2)]" };
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [histories, setHistories] = useState<HistoryItem[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  useEffect(() => {
    fetchHistories();
  }, []);

  const fetchHistories = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/history");
      if (res.ok) {
        const data = await res.json();
        setHistories(data);
      }
    } catch (error) {
      console.error("히스토리 로드 실패:", error);
    }
  };

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
        const audioFile = new File([audioBlob], "마이크_녹음.webm", { type: "audio/webm" });
        setFile(audioFile);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setFile(null);
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

  const handleClearFile = () => {
    setFile(null);
    setResult(null);
  };

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
      fetchHistories();
    } catch (error) {
      console.error(error);
      alert("분석 중 오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = result
    ? Object.entries(result.result).map(([key, value]) => ({
        name: regionLabels[key],
        value: Math.round(value * 100),
      }))
    : [];

  const titleData = result ? getGamifiedTitle(result.result) : null;

  return (
    <main className="min-h-screen bg-slate-950 text-white font-sans selection:bg-blue-500/30 pb-20 overflow-x-hidden">
      {/* Background ambient glow */}
      <div className="fixed top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/20 blur-[120px] pointer-events-none"></div>
      <div className="fixed bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-purple-900/20 blur-[120px] pointer-events-none"></div>

      <div className="max-w-4xl mx-auto px-6 py-12 relative z-10">
        {/* Header Section */}
        <header className="text-center mb-16 space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-slate-800/50 rounded-2xl mb-4 border border-slate-700/50 backdrop-blur-xl">
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
                  border border-slate-800 bg-slate-900/50 hover:bg-slate-800/80 hover:border-slate-600
                  rounded-3xl p-10 transition-all duration-300 ease-in-out
                  flex flex-col items-center justify-center text-center space-y-4 h-64
                  backdrop-blur-sm
                "
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="audio/*"
                  className="hidden"
                />
                <div className="p-4 bg-slate-800/80 rounded-full group-hover:scale-110 group-hover:bg-blue-500/20 transition-all duration-300">
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
                  cursor-pointer overflow-hidden backdrop-blur-sm
                  border rounded-3xl p-10 transition-all duration-300 ease-in-out
                  flex flex-col items-center justify-center text-center space-y-4 h-64
                  ${isRecording 
                    ? "border-red-500/50 bg-red-500/10 shadow-[0_0_30px_rgba(239,68,68,0.2)]" 
                    : "border-slate-800 bg-slate-900/50 hover:bg-slate-800/80 hover:border-slate-600"
                  }
                `}
              >
                <div className={`
                  p-4 rounded-full transition-all duration-300
                  ${isRecording 
                    ? "bg-red-500 animate-pulse scale-110" 
                    : "bg-slate-800/80 hover:scale-110 hover:bg-purple-500/20"
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
          {file && !result && (
            <div className="animate-in zoom-in-95 duration-300 flex flex-col items-center justify-center bg-slate-900/60 backdrop-blur-md border border-slate-700/50 rounded-3xl p-10 relative">
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
                      : "bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:shadow-[0_0_30px_rgba(59,130,246,0.6)] hover:scale-105"
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    <span>AI가 방언을 분석하고 있습니다...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-6 h-6" />
                    <span>이 목소리 분석하기</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && titleData && (
          <div className="mt-12 animate-in slide-in-from-bottom-8 duration-700 fade-in flex flex-col space-y-8">
            
            {/* Gamification Title Card */}
            <div className={`relative overflow-hidden rounded-3xl p-1 bg-gradient-to-r ${titleData.color} ${titleData.glow} transition-all duration-1000`}>
              <div className="bg-slate-950/90 backdrop-blur-xl rounded-[22px] px-8 py-12 text-center relative z-10 flex flex-col items-center justify-center">
                
                {/* Reset Button */}
                <button 
                  onClick={handleClearFile}
                  className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-full transition-colors"
                  title="다시 하기"
                >
                  <X className="w-5 h-5" />
                </button>

                <p className="text-slate-400 font-semibold tracking-widest text-sm mb-4 uppercase">당신의 사투리 칭호</p>
                <h2 className={`text-5xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r ${titleData.color} pb-2`}>
                  {titleData.title}
                </h2>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Transcript Card */}
              <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800/80 rounded-3xl p-8 shadow-xl relative overflow-hidden group h-full">
                <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl -mr-10 -mt-10"></div>
                <h3 className="text-slate-400 font-medium mb-6 text-sm uppercase tracking-widest flex items-center">
                  <Mic className="w-4 h-4 mr-2" />
                  AI가 들은 문장 (STT)
                </h3>
                <p className="text-2xl md:text-3xl font-medium leading-relaxed text-white">
                  "{result.transcript}"
                </p>
              </div>

              {/* Chart Card */}
              <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800/80 rounded-3xl p-8 shadow-xl relative overflow-hidden group">
                <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-3xl -mr-10 -mt-10"></div>
                <h3 className="text-slate-400 font-medium mb-6 text-sm uppercase tracking-widest flex items-center">
                  <MapPin className="w-4 h-4 mr-2" />
                  사투리 농도 분석
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
                        formatter={(value: number) => [`${value}%`, '농도']}
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

        {/* History Section */}
        {histories.length > 0 && (
          <div className="mt-24 animate-in fade-in slide-in-from-bottom-8 duration-1000">
            <h2 className="text-2xl font-bold mb-8 flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div className="flex items-center space-x-3">
                <History className="w-6 h-6 text-slate-400" />
                <span>최근 분석된 말씨들</span>
              </div>
              <span className="text-sm font-medium text-slate-400 bg-slate-800/50 px-4 py-1.5 rounded-full border border-slate-700/50">
                최근 {histories.length}개
              </span>
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {histories.map((item) => {
                const regionScores = [
                  { name: "경상도", score: item.gyeongsang, color: "text-red-400", bg: "bg-red-500/10", border: "border-red-500/20" },
                  { name: "전라도", score: item.jeolla, color: "text-purple-400", bg: "bg-purple-500/10", border: "border-purple-500/20" },
                  { name: "충청도", score: item.chungcheong, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
                  { name: "표준어", score: item.standard, color: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20" }
                ];
                
                const topRegion = regionScores.sort((a, b) => b.score - a.score)[0];
                const dateStr = new Date(item.created_at).toLocaleString('ko-KR', {
                  month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                });

                return (
                  <div key={item.id} className={`bg-slate-900/40 backdrop-blur-sm border ${topRegion.border} hover:bg-slate-800/60 rounded-2xl p-6 transition-all duration-300 hover:-translate-y-1 group cursor-default shadow-lg`}>
                    <div className="flex justify-between items-start mb-4">
                      <div className={`px-3 py-1 rounded-full text-xs font-bold ${topRegion.bg} ${topRegion.color} flex items-center shadow-sm`}>
                        <span className="mr-1 w-1.5 h-1.5 rounded-full bg-current"></span>
                        {topRegion.name} {Math.round(topRegion.score * 100)}%
                      </div>
                      <div className="flex items-center text-slate-500 text-xs">
                        <Clock className="w-3 h-3 mr-1 opacity-70" />
                        {dateStr}
                      </div>
                    </div>
                    <p className="text-slate-300 font-medium line-clamp-3 group-hover:text-white transition-colors leading-relaxed">
                      "{item.transcript}"
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
