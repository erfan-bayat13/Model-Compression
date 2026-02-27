import { Route, Routes } from "react-router-dom";

import { Configure } from "./pages/Configure";
import { Home } from "./pages/Home";
import { Progress } from "./pages/Progress";
import { Results } from "./pages/Results";

function Placeholder({ name }: { name: string }) {
  return (
    <div className="min-h-screen flex items-center justify-center text-gray-400 text-sm">
      {name} — coming soon
    </div>
  );
}

export function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-gray-200 bg-white px-6 py-4">
        <span className="text-sm font-semibold text-gray-900 tracking-tight">
          Model Compression
        </span>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/configure" element={<Configure />} />
          <Route path="/progress/:jobId" element={<Progress />} />
          <Route path="/results/:jobId" element={<Results />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
