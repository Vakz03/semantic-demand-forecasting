import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  FileSpreadsheet, 
  BarChart3, 
  AlertCircle, 
  RefreshCw, 
  Download, 
  Search, 
  Layers, 
  Package, 
  ArrowRight
} from 'lucide-react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

export default function App() {
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [fileStats, setFileStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [catalog, setCatalog] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [filterRotation, setFilterRotation] = useState("all");
  useEffect(() => {
    let interval;
    if (loading) {
      interval = setInterval(() => setElapsedTime(t => t + 1), 1000);
    } else {
      setElapsedTime(0);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading]);
  const handleFileSelection = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setError(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      if (lines.length > 0) {
        const headers = lines[0].split(',').map(h => h.replace(/["']/g, '').trim());
        const previewRows = lines.slice(1, 6).map(line => {
          return line.split(',').map(cell => cell.replace(/["']/g, '').trim());
        });
        
        const estRows = Math.round(selectedFile.size / Math.max(30, (lines[1]?.length || 50)));
        setFileStats({
          sizeMb: (selectedFile.size / (1024 * 1024)).toFixed(2),
          colCount: headers.length,
          estRows: estRows.toLocaleString()
        });
        setFilePreview({ headers, rows: previewRows });
      }
    };
    reader.readAsText(selectedFile.slice(0, 25600));
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setElapsedTime(0);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict-demand/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const { pronostico, catalogo } = response.data;
      setResults(pronostico);
      setCatalog(catalogo || []);

      const uniqueIds = [...new Set(pronostico.unique_id)];
      if (uniqueIds.length > 0) {
        setSelectedId(uniqueIds[0]);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Error durante la ejecución del modelo. Verifique la estructura del CSV.");
    } finally {
      setLoading(false);
    }
  };
  const getProductInfo = (id) => {
    if (!catalog || catalog.length === 0) return { name: `SKU: ${id}`, details: "" };
    const item = catalog.find(c => c.unique_id === id);
    if (!item) return { name: `SKU: ${id}`, details: "" };

    const nameKey = Object.keys(item).find(k => /des|nom|prod|name/i.test(k) && k !== 'unique_id' && isNaN(Number(item[k])));
    const name = nameKey && item[nameKey] ? String(item[nameKey]) : `SKU ${id}`;
    const otherKeys = Object.keys(item).filter(k => k !== 'unique_id' && k !== nameKey && isNaN(Number(item[k])));
    const details = otherKeys.map(k => item[k]).filter(Boolean).join(" · ");

    return { name, details };
  };
  const seriesSummaries = useMemo(() => {
    if (!results) return [];
    const uniqueIds = [...new Set(results.unique_id)];
    
    return uniqueIds.map(id => {
      let sum = 0;
      let count = 0;
      for (let i = 0; i < results.unique_id.length; i++) {
        if (results.unique_id[i] === id) {
          sum += results.LGBMRegressor[i];
          count++;
        }
      }
      const avg = count > 0 ? sum / count : 0;
      let rotation = "Normal";
      if (avg < 0.5) rotation = "Intermitente";
      else if (avg > 25) rotation = "Alta";

      const info = getProductInfo(id);
      return {
        id,
        name: info.name,
        details: info.details,
        total: Math.round(sum * 10) / 10,
        avg: Math.round(avg * 100) / 100,
        rotation
      };
    });
  }, [results, catalog]);
  const filteredSeries = useMemo(() => {
    return seriesSummaries.filter(item => {
      const matchSearch = item.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          item.id.toLowerCase().includes(searchTerm.toLowerCase());
      const matchRot = filterRotation === "all" || 
                       (filterRotation === "high" && item.rotation === "Alta") ||
                       (filterRotation === "medium" && item.rotation === "Normal") ||
                       (filterRotation === "low" && item.rotation === "Intermitente");
      return matchSearch && matchRot;
    });
  }, [seriesSummaries, searchTerm, filterRotation]);
  const currentSKUData = useMemo(() => {
    if (!results || !selectedId) return { chartData: [], kpis: null };

    const chartData = [];
    let sum = 0;
    const values = [];

    for (let i = 0; i < results.unique_id.length; i++) {
      if (results.unique_id[i] === selectedId) {
        const val = Math.round(results.LGBMRegressor[i] * 100) / 100;
        sum += val;
        values.push(val);
        const upperBand = Math.round((val * 1.25 + 0.5) * 100) / 100;
        const lowerBand = Math.max(0, Math.round((val * 0.75 - 0.2) * 100) / 100);

        chartData.push({
          ds: results.ds[i],
          demanda: val,
          bandaSuperior: upperBand,
          bandaInferior: lowerBand
        });
      }
    }

    const count = chartData.length || 1;
    const avg = sum / count;
    const variance = values.reduce((acc, v) => acc + Math.pow(v - avg, 2), 0) / count;
    const stdDev = Math.sqrt(variance);
    const safetyStock = Math.max(1, Math.round(1.65 * stdDev + (avg * 0.3)));
    const reorderPoint = Math.round(safetyStock + (avg * 3));

    const kpis = {
      totalDemand: Math.round(sum),
      dailyAvg: avg.toFixed(2),
      safetyStock,
      reorderPoint,
      cycleDays: count
    };

    return { chartData, kpis };
  }, [results, selectedId]);
  const handleExportCSV = () => {
    if (!currentSKUData.chartData.length) return;
    const info = getProductInfo(selectedId);
    
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Fecha,SKU,Descripcion,Demanda_Estimada_Uds,Banda_Superior_Uds,Banda_Inferior_Uds,Accion_Logistica\n";
    
    currentSKUData.chartData.forEach(row => {
      const action = row.demanda < 0.5 ? "Monitoreo pasivo" : row.demanda > 20 ? "Reabastecimiento critico" : "Stock normal";
      csvContent += `${row.ds},"${selectedId}","${info.name.replace(/"/g, '""')}",${row.demanda},${row.bandaSuperior},${row.bandaInferior},"${action}"\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `pronostico_sku_${selectedId}_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      <header className="h-14 bg-slate-900 border-b border-slate-800 px-6 flex items-center justify-between text-white sticky top-0 z-30">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 text-white p-1.5 rounded">
            <Layers className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-bold tracking-tight text-sm text-slate-100">SUPPLY CHAIN CONSOLE</span>
            <span className="text-xs text-slate-400 font-mono">v1.2 // Demand Planning Engine</span>
          </div>
        </div>

        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2 text-slate-300 border-r border-slate-800 pr-4">
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500"></span>
            <span>Motor: LightGBM + Qwen2.5</span>
          </div>
          {results && (
            <button
              onClick={() => { setResults(null); setFile(null); setFilePreview(null); }}
              className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded text-xs transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Cargar nuevo dataset
            </button>
          )}
        </div>
      </header>
      {!results ? (
        <main className="flex-1 max-w-6xl w-full mx-auto p-6 md:p-10">
          <div className="mb-6">
            <h1 className="text-xl font-semibold text-slate-900">Ingestión de Registro Histórico</h1>
            <p className="text-xs text-slate-500 mt-1">
              Cargue un archivo CSV transaccional o agregado para inferencia semántica automática y entrenamiento de series temporales.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 bg-white border border-slate-200 rounded p-5 flex flex-col justify-between">
              <div>
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-3">Archivo de Entrada</span>
                
                <label 
                  htmlFor="file-upload"
                  className="border-2 border-dashed border-slate-300 hover:border-blue-500 hover:bg-slate-50/60 rounded p-6 flex flex-col items-center justify-center cursor-pointer transition-colors text-center"
                >
                  <FileSpreadsheet className="w-8 h-8 text-slate-400 mb-2" />
                  <span className="text-xs font-semibold text-slate-700">Seleccionar archivo CSV</span>
                  <span className="text-[11px] text-slate-400 mt-1">Delimitado por comas (.csv)</span>
                  <input 
                    type="file" 
                    id="file-upload" 
                    accept=".csv"
                    onChange={(e) => e.target.files[0] && handleFileSelection(e.target.files[0])}
                    className="hidden" 
                  />
                </label>

                {file && (
                  <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-200/60 font-mono text-[11px]">
                      <span className="text-slate-500">Nombre:</span>
                      <span className="font-semibold text-slate-800 truncate max-w-[150px]">{file.name}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-200/60 font-mono text-[11px]">
                      <span className="text-slate-500">Tamaño:</span>
                      <span className="text-slate-800">{fileStats?.sizeMb} MB</span>
                    </div>
                    <div className="flex justify-between py-1 font-mono text-[11px]">
                      <span className="text-slate-500">Filas est.:</span>
                      <span className="text-slate-800">~{fileStats?.estRows}</span>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-4 p-3 bg-rose-50 border border-rose-200 rounded text-xs text-rose-700 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold block">Error de procesamiento:</span>
                    <span>{error}</span>
                  </div>
                </div>
              )}

              <div className="mt-6">
                {!loading ? (
                  <button
                    onClick={handleUpload}
                    disabled={!file}
                    className={`w-full py-2.5 px-4 rounded text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${
                      file 
                        ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm' 
                        : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                    }`}
                  >
                    <span>Ejecutar Pronóstico</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                ) : (
                  <div className="p-3 bg-slate-900 text-white rounded text-xs border border-slate-800">
                    <div className="flex justify-between text-[11px] text-slate-400 font-mono mb-2">
                      <span>TELEMETRÍA // PROCESANDO</span>
                      <span>{elapsedTime}s / ~80s</span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden mb-3">
                      <div 
                        className="bg-blue-500 h-1.5 transition-all duration-1000 ease-linear" 
                        style={{ width: `${Math.min(95, (elapsedTime / 80) * 100)}%` }}
                      />
                    </div>
                    <span className="text-[11px] text-slate-300 block">
                      Inferencia semántica y entrenamiento de 865 series temporales en paralelo...
                    </span>
                  </div>
                )}
              </div>
            </div>
            <div className="lg:col-span-2 bg-white border border-slate-200 rounded p-5 flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Inspección Preliminar de Columnas
                </span>
                {filePreview && (
                  <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                    {filePreview.headers.length} Columnas Detectadas
                  </span>
                )}
              </div>

              {filePreview ? (
                <div className="overflow-x-auto border border-slate-200 rounded flex-1">
                  <table className="min-w-full text-xs divide-y divide-slate-200">
                    <thead className="bg-slate-50 text-slate-600 font-mono text-[11px]">
                      <tr>
                        {filePreview.headers.map((h, i) => (
                          <th key={i} className="px-3 py-2 text-left font-semibold whitespace-nowrap">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 bg-white font-mono text-[11px] text-slate-700">
                      {filePreview.rows.map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-slate-50/80">
                          {row.map((cell, cIdx) => (
                            <td key={cIdx} className="px-3 py-1.5 whitespace-nowrap">
                              {cell || <span className="text-slate-300">null</span>}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center border border-dashed border-slate-200 rounded p-8 text-center text-slate-400">
                  <Package className="w-8 h-8 text-slate-300 mb-2" />
                  <span className="text-xs">Seleccione un archivo CSV para inspeccionar la estructura de datos.</span>
                  <span className="text-[11px] text-slate-400 mt-1">El previsualizador validará el esquema en cliente sin transferir datos aún.</span>
                </div>
              )}
            </div>
          </div>
        </main>
      ) : (
        <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          <aside className="w-full lg:w-80 bg-white border-r border-slate-200 flex flex-col h-auto lg:h-[calc(100vh-3.5rem)] flex-shrink-0">
            <div className="p-3 border-b border-slate-200 bg-slate-50/50">
              <div className="relative mb-2">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                <input 
                  type="text"
                  placeholder="Buscar SKU o descripción..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-8 pr-3 py-1.5 bg-white border border-slate-300 rounded text-xs focus:outline-none focus:border-blue-500 font-mono"
                />
              </div>
              <div className="flex gap-1 text-[10px] font-medium text-slate-600">
                <button 
                  onClick={() => setFilterRotation("all")}
                  className={`px-2 py-0.5 rounded ${filterRotation === 'all' ? 'bg-slate-900 text-white' : 'bg-slate-200/70 hover:bg-slate-200'}`}
                >
                  Todos ({seriesSummaries.length})
                </button>
                <button 
                  onClick={() => setFilterRotation("high")}
                  className={`px-2 py-0.5 rounded ${filterRotation === 'high' ? 'bg-slate-900 text-white' : 'bg-slate-200/70 hover:bg-slate-200'}`}
                >
                  Alta
                </button>
                <button 
                  onClick={() => setFilterRotation("low")}
                  className={`px-2 py-0.5 rounded ${filterRotation === 'low' ? 'bg-slate-900 text-white' : 'bg-slate-200/70 hover:bg-slate-200'}`}
                >
                  Intermitente
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-y-auto divide-y divide-slate-100">
              {filteredSeries.length > 0 ? (
                filteredSeries.map(item => (
                  <button
                    key={item.id}
                    onClick={() => setSelectedId(item.id)}
                    className={`w-full text-left p-3 text-xs transition-colors flex flex-col gap-1 ${
                      selectedId === item.id 
                        ? 'bg-blue-50/70 border-l-2 border-blue-600' 
                        : 'hover:bg-slate-50'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[11px] font-semibold text-slate-500">{item.id}</span>
                      <span className={`text-[10px] px-1.5 py-0.2 rounded font-medium ${
                        item.rotation === 'Alta' 
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                          : item.rotation === 'Normal'
                            ? 'bg-blue-50 text-blue-700 border border-blue-200'
                            : 'bg-slate-100 text-slate-600 border border-slate-200'
                      }`}>
                        {item.rotation}
                      </span>
                    </div>
                    <span className="font-semibold text-slate-900 truncate">{item.name}</span>
                    <div className="flex justify-between text-[11px] text-slate-500 font-mono mt-0.5">
                      <span>Proy. ciclo:</span>
                      <span className="font-medium text-slate-700">{item.total} uds</span>
                    </div>
                  </button>
                ))
              ) : (
                <div className="p-6 text-center text-xs text-slate-400">
                  No se encontraron SKUs coincidentes.
                </div>
              )}
            </div>
          </aside>
          <main className="flex-1 bg-slate-50 p-6 overflow-y-auto">
            <div className="bg-white border border-slate-200 rounded p-4 mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded border border-slate-200 font-semibold">
                    SKU // {selectedId}
                  </span>
                  {getProductInfo(selectedId).details && (
                    <span className="text-xs text-slate-500">
                      {getProductInfo(selectedId).details}
                    </span>
                  )}
                </div>
                <h2 className="text-lg font-bold text-slate-900 tracking-tight">
                  {getProductInfo(selectedId).name}
                </h2>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleExportCSV}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded text-xs font-semibold shadow-sm transition-colors"
                >
                  <Download className="w-3.5 h-3.5 text-slate-600" />
                  <span>Exportar SKU (.csv)</span>
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className="bg-white border border-slate-200 rounded p-4">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                  Demanda Total Proyectada
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold font-mono text-slate-900">
                    {currentSKUData.kpis?.totalDemand.toLocaleString()}
                  </span>
                  <span className="text-xs text-slate-500">uds</span>
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  Ciclo completo ({currentSKUData.kpis?.cycleDays} días)
                </span>
              </div>

              <div className="bg-white border border-slate-200 rounded p-4">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                  Promedio Diario Estimado
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold font-mono text-slate-900">
                    {currentSKUData.kpis?.dailyAvg}
                  </span>
                  <span className="text-xs text-slate-500">uds/día</span>
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  Tasa de rotación prevista
                </span>
              </div>

              <div className="bg-white border border-slate-200 rounded p-4">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                  Stock de Seguridad Sugerido
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold font-mono text-blue-600">
                    {currentSKUData.kpis?.safetyStock}
                  </span>
                  <span className="text-xs text-slate-500">uds (Buffer)</span>
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  Nivel de servicio meta ~95%
                </span>
              </div>

              <div className="bg-white border border-slate-200 rounded p-4">
                <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
                  Punto de Reorden (ROP)
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold font-mono text-amber-600">
                    {currentSKUData.kpis?.reorderPoint}
                  </span>
                  <span className="text-xs text-slate-500">uds mínimas</span>
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  Considerando 3 días Lead Time
                </span>
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded p-4 mb-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-slate-700" />
                  <span className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
                    Curva de Demanda y Banda de Incertidumbre
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 bg-slate-900 inline-block"></span>
                    <span className="text-slate-600">Pronóstico Diario</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-3 h-2 bg-blue-100 border border-blue-300 inline-block"></span>
                    <span className="text-slate-600">Banda de Confianza (+25%)</span>
                  </div>
                </div>
              </div>

              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={currentSKUData.chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="2 2" stroke="#E2E8F0" vertical={false} />
                    <XAxis 
                      dataKey="ds" 
                      stroke="#94A3B8" 
                      tick={{ fill: '#475569', fontSize: 11, fontFamily: 'monospace' }} 
                      tickMargin={8}
                    />
                    <YAxis 
                      stroke="#94A3B8" 
                      tick={{ fill: '#475569', fontSize: 11, fontFamily: 'monospace' }} 
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#0F172A', 
                        borderColor: '#334155', 
                        borderRadius: '4px', 
                        color: '#F8FAFC',
                        fontSize: '11px',
                        fontFamily: 'monospace'
                      }}
                      itemStyle={{ color: '#93C5FD' }}
                      formatter={(val, name) => [
                        `${val} uds`, 
                        name === 'demanda' ? 'Pronóstico' : 'Banda Superior'
                      ]}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="bandaSuperior" 
                      stroke="none" 
                      fill="#BFDBFE" 
                      fillOpacity={0.4} 
                    />
                    <Line 
                      type="monotone" 
                      dataKey="demanda" 
                      stroke="#0F172A" 
                      strokeWidth={2.5} 
                      dot={{ r: 3, fill: '#0F172A' }}
                      activeDot={{ r: 6, fill: '#2563EB' }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded overflow-hidden">
              <div className="p-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
                  Plan Operativo Diario Desglosado
                </span>
                <span className="text-[11px] font-mono text-slate-500">
                  {currentSKUData.chartData.length} Días de Previsión
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-slate-200 text-xs">
                  <thead className="bg-slate-100 font-mono text-[11px] text-slate-600">
                    <tr>
                      <th className="px-4 py-2 text-left font-semibold">Fecha Planificada</th>
                      <th className="px-4 py-2 text-right font-semibold">Pronóstico Diario</th>
                      <th className="px-4 py-2 text-right font-semibold">Banda Límite Sup.</th>
                      <th className="px-4 py-2 text-left font-semibold pl-8">Acción Logística Sugerida</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono text-xs">
                    {currentSKUData.chartData.map((row, idx) => {
                      const isHigh = row.demanda > 20;
                      const isLow = row.demanda < 0.5;

                      return (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="px-4 py-2.5 text-slate-800">{row.ds}</td>
                          <td className="px-4 py-2.5 text-right font-semibold text-slate-900">{row.demanda.toFixed(2)}</td>
                          <td className="px-4 py-2.5 text-right text-slate-500">{row.bandaSuperior.toFixed(2)}</td>
                          <td className="px-4 py-2.5 pl-8">
                            {isLow ? (
                              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-700 border border-slate-200">
                                Monitoreo pasivo / Demanda esporádica
                              </span>
                            ) : isHigh ? (
                              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-50 text-amber-800 border border-amber-200">
                                Reabastecimiento preventivo prioritario
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-800 border border-emerald-200">
                                Stock de ciclo suficiente
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

          </main>
        </div>
      )}

    </div>
  );
}
