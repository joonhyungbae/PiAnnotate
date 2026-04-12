import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { PieceList } from './components/PieceList';
import { Visualizer } from './components/Visualizer';
import { FingeringCompare } from './components/FingeringCompare';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PieceList />} />
        <Route path="/vis" element={<Visualizer />} />
        <Route path="/fingering/:pieceId" element={<FingeringCompare />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
